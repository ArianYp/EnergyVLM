#!/usr/bin/env python3
"""
End-to-end Energy-Based Distillation for SD3.5 DiT.

Uses official Stable Diffusion 3.5 models:
- Teacher: stabilityai/stable-diffusion-3.5-large (8.1B params)
- Student: stabilityai/stable-diffusion-3.5-medium (2.5B params)

Both are standard diffusers models with the same latent space.
Projector g_phi trained end-to-end with L_align.

Usage:
  python train_distill_e2e_sd3.py --num_steps 10000 --K 4
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from projector import DiT2DINOProjector


def _make_hook(capture_dict: dict, is_final_step: list):
    def hook(module, input, output):
        if is_final_step[0]:
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states)
            enc, hid = output
            capture_dict["hidden"] = hid.detach()

    return hook


def run_teacher_denoising_capture_hidden(
    pipe, latents, prompt_embeds, pooled_embeds, device, capture_dict, is_final_step_ref
):
    """Run teacher denoising, capture hidden states at final step."""
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae

    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final_step_ref))

    try:
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            is_final_step_ref[0] = (i == len(timesteps) - 1)
            latent_model_input = latents  # FlowMatchEulerDiscreteScheduler has no scale_model_input
            timestep = t.expand(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        with torch.no_grad():
            latents_for_decode = (latents / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)
            image = vae.decode(latents_for_decode, return_dict=False)[0]
        pil_images = pipe.image_processor.postprocess(image, output_type="pil")
    finally:
        handle.remove()

    if "hidden" not in capture_dict:
        raise RuntimeError("Failed to capture SD3 hidden states")

    H_T = capture_dict["hidden"]  # [B, seq, dim]
    h_T = H_T.mean(dim=1)
    return h_T, pil_images, latents


def load_coco_captions(coco_path: str) -> list:
    path = Path(coco_path)
    if path.is_dir():
        for ann_file in ["annotations/captions_train2017.json", "annotations/captions_train2014.json"]:
            candidate = path / ann_file
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No COCO captions under {coco_path}")
    with open(path) as f:
        ann = json.load(f)
    return [a["caption"] for a in ann["annotations"]]


def get_prompts(num: int, pool: list | None, seed: int) -> list:
    if pool:
        rng = random.Random(seed)
        return [pool[rng.randint(0, len(pool) - 1)] for _ in range(num)]
    base = [
        "A photo of a cat sitting on a wooden table",
        "A landscape with mountains and a lake at sunset",
        "A bowl of fresh fruit on a kitchen counter",
        "A person walking a dog in the park",
    ]
    rng = random.Random(seed)
    return [base[rng.randint(0, len(base) - 1)] for _ in range(num)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="checkpoints/distill_e2e_sd3")
    parser.add_argument("--teacher_id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--student_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument("--lr_student", type=float, default=1e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--coco_path", default="/data/gpfs/datasets/COCO")
    parser.add_argument("--wandb_project", default="energy-based-dit")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    caption_pool = None
    if args.coco_path and Path(args.coco_path).exists():
        caption_pool = load_coco_captions(args.coco_path)
        print(f"Loaded {len(caption_pool)} COCO captions")

    # --- Load teacher (frozen) ---
    print("Loading teacher (SD3.5 Large)...")
    pipe_teacher = StableDiffusion3Pipeline.from_pretrained(
        args.teacher_id,
        torch_dtype=torch.bfloat16,
        token="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"
    )
    pipe_teacher = pipe_teacher.to(device)
    pipe_teacher.transformer.eval()
    pipe_teacher.vae.eval()
    for p in pipe_teacher.transformer.parameters():
        p.requires_grad = False

    # --- Load student (trainable) ---
    print("Loading student (SD3.5 Medium)...")
    pipe_student = StableDiffusion3Pipeline.from_pretrained(
        args.student_id,
        torch_dtype=torch.float32,
        token="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"
    )
    pipe_student.transformer = pipe_student.transformer.to(device).float()
    pipe_student.transformer.train()
    pipe_student.vae = pipe_teacher.vae
    pipe_student.text_encoder = pipe_teacher.text_encoder
    pipe_student.text_encoder_2 = pipe_teacher.text_encoder_2
    pipe_student.text_encoder_3 = pipe_teacher.text_encoder_3
    pipe_student.tokenizer = pipe_teacher.tokenizer
    pipe_student.tokenizer_2 = pipe_teacher.tokenizer_2
    pipe_student.tokenizer_3 = pipe_teacher.tokenizer_3
    pipe_student.scheduler = pipe_teacher.scheduler

    # --- DINOv2 (frozen) ---
    print("Loading DINOv2...")
    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id)
    dinov2_model.eval()
    dinov2_model = dinov2_model.to(device)
    for p in dinov2_model.parameters():
        p.requires_grad = False

    # --- Projector ---
    dit_dim = pipe_teacher.transformer.config.num_attention_heads * pipe_teacher.transformer.config.attention_head_dim
    dinov2_dim = dinov2_model.config.hidden_size
    projector = DiT2DINOProjector(dit_dim=dit_dim, dinov2_dim=dinov2_dim).to(device)

    # --- Freeze all except student DiT and projector ---
    for c in [pipe_student.vae, pipe_student.text_encoder, pipe_student.text_encoder_2, pipe_student.text_encoder_3]:
        c.eval()
        for p in c.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": pipe_student.transformer.parameters(), "lr": args.lr_student},
            {"params": projector.parameters(), "lr": args.lr_projector},
        ]
    )

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scheduler = pipe_teacher.scheduler
    latent_channels = pipe_teacher.transformer.config.in_channels

    pbar = tqdm(range(args.num_steps), desc="Distillation SD3.5")

    for step in pbar:
        pipe_student.transformer.train()
        projector.train()
        optimizer.zero_grad()

        prompts = get_prompts(1, caption_pool, args.seed + step)
        prompts_K = prompts * args.K

        with torch.no_grad():
            prompt_embeds, _, pooled_embeds, _ = pipe_teacher.encode_prompt(
                prompt=prompts_K,
                prompt_2=prompts_K,
                prompt_3=prompts_K,
                do_classifier_free_guidance=False,
                device=device,
                num_images_per_prompt=1,
            )

        latents = pipe_teacher.prepare_latents(
            args.K,
            latent_channels,
            args.height,
            args.width,
            prompt_embeds.dtype,
            device,
            generator=torch.Generator(device).manual_seed(args.seed + step + 1),
        )

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        capture_dict = {}
        is_final = [False]

        h_T, pil_images, final_latents = run_teacher_denoising_capture_hidden(
            pipe_teacher, latents, prompt_embeds, pooled_embeds, device, capture_dict, is_final
        )
        dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
        pixel_values = dinov2_inputs["pixel_values"].to(device)

        with torch.no_grad():
            dinov2_out = dinov2_model(pixel_values=pixel_values)
            z_DINO = dinov2_out.last_hidden_state[:, 0, :]
            z_DINO = F.normalize(z_DINO, p=2, dim=-1)

        z_DiT = projector(h_T.float())
        L_align = 1.0 - (z_DiT * z_DINO).sum(dim=-1).mean()
        E = 1.0 - (z_DiT * z_DINO).sum(dim=-1)
        q = F.softmax(-E / args.tau, dim=0)

        timesteps = scheduler.timesteps
        t_idx = random.randint(0, len(timesteps) - 1)
        t = timesteps[t_idx]

        noise = torch.randn_like(final_latents, device=device, dtype=torch.float32)
        t_batch = t.expand(args.K) if isinstance(t, torch.Tensor) else torch.tensor([t], dtype=torch.float32, device=device).expand(args.K)
        z_t = scheduler.scale_noise(final_latents.float(), t_batch, noise)
        latent_model_input = z_t  # FlowMatchEulerDiscreteScheduler has no scale_model_input
        timestep = t.expand(args.K)

        with torch.no_grad():
            latent_tea = latent_model_input.to(pipe_teacher.transformer.dtype)
            eps_tea = pipe_teacher.transformer(
                hidden_states=latent_tea,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            eps_tea = eps_tea.float()

        latent_stu = latent_model_input.float()
        eps_stu = pipe_student.transformer(
            hidden_states=latent_stu,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.float(),
            pooled_projections=pooled_embeds.float(),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        diff = eps_stu.float() - eps_tea
        mse_per = (diff ** 2).sum(dim=(1, 2, 3))
        L_distill = (q * mse_per).sum()
        L_KL = (mse_per).sum()
        loss = L_distill + args.lambda_align * L_align + args.beta_kl * L_KL

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(pipe_student.transformer.parameters()) + list(projector.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", align=f"{L_align.item():.4f}")

        if not args.no_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/L_distill": L_distill.item(),
                    "train/L_align": L_align.item(),
                    "train/L_KL": L_KL.item(),
                },
                step=step + 1,
            )

        if (step + 1) % args.save_every == 0:
            torch.save(pipe_student.transformer.state_dict(), Path(args.output_dir) / f"student_step{step+1}.pt")
            torch.save(projector.state_dict(), Path(args.output_dir) / f"projector_step{step+1}.pt")

    torch.save(pipe_student.transformer.state_dict(), Path(args.output_dir) / "student_final.pt")
    torch.save(projector.state_dict(), Path(args.output_dir) / "projector_final.pt")
    print("Done.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
