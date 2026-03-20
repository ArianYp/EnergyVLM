#!/usr/bin/env python3
"""
End-to-end Energy-Based Distillation for DiT.

Trains the projector g_phi jointly with the student during distillation (no pre-phase).
- Teacher: PixArt-XL (frozen)
- Student: Either (a) pretrained PixArt-XL-2-256x256 at 256x256, or (b) smaller DiT (14 layers) at 512x512
- Projector g_phi: trained end-to-end with L_align

Full objective: L = L_distill + λ*L_align + β*L_KL

Usage:
  # Option 1: Pretrained small PixArt at 256
  python train_distill_e2e.py --student_id PixArt-alpha/PixArt-XL-2-256x256

  # Option 2: Manually smaller (14 layers) at 512
  python train_distill_e2e.py --num_steps 10000 --K 4 --tau 0.1 --lambda_align 0.1 --beta_kl 0.1
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from diffusers import PixArtAlphaPipeline
from diffusers.models import PixArtTransformer2DModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from projector import DiT2DINOProjector


# --- DiT hidden state extraction ---

def _make_hook(capture_dict: dict, is_final_step: list):
    """Create a forward hook that captures output only at the final denoising step."""

    def hook(module, input, output):
        if is_final_step[0]:
            hidden = output[0] if isinstance(output, tuple) else output
            capture_dict["hidden"] = hidden.detach()

    return hook


def run_teacher_denoising_capture_hidden(
    pipe,
    latents,
    prompt_embeds,
    prompt_attention_mask,
    added_cond_kwargs,
    device,
    capture_dict,
    is_final_step_ref,
):
    """
    Run teacher denoising loop and capture hidden states at final step.
    Returns: (final_latents, images) - no gradients through teacher.
    """
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae

    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final_step_ref))

    try:
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            is_final_step_ref[0] = (i == len(timesteps) - 1)
            latent_model_input = scheduler.scale_model_input(latents, t)

            if not torch.is_tensor(t):
                t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            else:
                t_tensor = t.unsqueeze(0).to(device) if t.dim() == 0 else t
            t_tensor = t_tensor.expand(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred = transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=t_tensor,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                latent_channels = transformer.config.in_channels
                if transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        with torch.no_grad():
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
    finally:
        handle.remove()

    if "hidden" not in capture_dict:
        raise RuntimeError("Failed to capture DiT hidden states at final step")

    H_T = capture_dict["hidden"]
    h_T = H_T.mean(dim=1)
    return h_T, image, latents


def load_coco_captions(coco_path: str) -> list:
    """Load captions from COCO annotations."""
    path = Path(coco_path)
    if path.is_dir():
        for ann_file in [
            "annotations/captions_train2017.json",
            "annotations/captions_train2014.json",
            "captions_train2017.json",
        ]:
            candidate = path / ann_file
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No COCO captions JSON found under {coco_path}")
    with open(path) as f:
        ann = json.load(f)
    return [a["caption"] for a in ann["annotations"]]


def get_prompts(num_prompts: int, caption_pool: list | None, seed: int) -> list:
    """Sample prompts from caption pool or built-in."""
    pool = caption_pool if caption_pool else []
    if pool:
        rng = random.Random(seed)
        indices = [rng.randint(0, len(pool) - 1) for _ in range(num_prompts)]
        return [pool[i] for i in indices]
    base_prompts = [
        "A photo of a cat sitting on a wooden table",
        "A landscape with mountains and a lake at sunset",
        "A bowl of fresh fruit on a kitchen counter",
        "A person walking a dog in the park",
        "A vintage car parked on a street",
        "A cozy living room with a fireplace",
        "An astronaut floating in space",
        "A delicious pizza with various toppings",
        "A butterfly on a colorful flower",
        "A modern office with large windows",
    ]
    rng = random.Random(seed)
    return [base_prompts[rng.randint(0, len(base_prompts) - 1)] for _ in range(num_prompts)]


def main():
    parser = argparse.ArgumentParser(description="End-to-end energy-based DiT distillation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/distill_e2e")
    parser.add_argument("--teacher_id", type=str, default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--student_id", type=str, default=None,
                        help="Use pretrained small PixArt (e.g. PixArt-alpha/PixArt-XL-2-256x256). "
                             "If set, runs at 256 res; else creates smaller from teacher (student_layers).")
    parser.add_argument("--student_layers", type=int, default=14,
                        help="Number of transformer layers when creating student from teacher (ignored if student_id set).")
    parser.add_argument("--dinov2_id", type=str, default="facebook/dinov2-base")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--K", type=int, default=4, help="Number of samples per prompt for energy weighting")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for Gibbs weights")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="Weight for L_align")
    parser.add_argument("--beta_kl", type=float, default=0.1, help="Weight for L_KL")
    parser.add_argument("--lr_student", type=float, default=1e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--coco_path", type=str, default="/data/gpfs/datasets/COCO")
    parser.add_argument("--wandb_project", type=str, default="energy-based-dit")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    caption_pool = None
    if args.coco_path and Path(args.coco_path).exists():
        caption_pool = load_coco_captions(args.coco_path)
        print(f"Loaded {len(caption_pool)} COCO captions")
    else:
        print("Using built-in prompts")

    # --- Load teacher (frozen) ---
    if args.student_id:
        # Use pretrained small PixArt (e.g. XL-2-256x256): run at 256, teacher must match resolution
        args.height, args.width = 256, 256
        teacher_id = args.student_id  # same model for teacher at 256
        print(f"Using pretrained student {args.student_id} at 256x256 (teacher = same model)")
    else:
        teacher_id = args.teacher_id

    print("Loading teacher...")
    pipe_teacher = PixArtAlphaPipeline.from_pretrained(
        teacher_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_teacher = pipe_teacher.to(device)
    pipe_teacher.transformer.eval()
    pipe_teacher.vae.eval()
    for p in pipe_teacher.transformer.parameters():
        p.requires_grad = False

    # --- Load or create student ---
    if args.student_id:
        # Load pretrained small PixArt (e.g. PixArt-XL-2-256x256)
        print(f"Loading student from {args.student_id}...")
        pipe_student = PixArtAlphaPipeline.from_pretrained(
            args.student_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        pipe_student.transformer = pipe_student.transformer.to(device).float()
        pipe_student.transformer.train()
        pipe_student.vae = pipe_teacher.vae
        pipe_student.text_encoder = pipe_teacher.text_encoder
        pipe_student.tokenizer = pipe_teacher.tokenizer
        pipe_student.scheduler = pipe_teacher.scheduler
    else:
        # Create smaller student from teacher config (fewer layers)
        print(f"Creating student with {args.student_layers} layers (teacher has 28)...")
        teacher_config = pipe_teacher.transformer.config
        student_transformer = PixArtTransformer2DModel(
            num_attention_heads=teacher_config.num_attention_heads,
            attention_head_dim=teacher_config.attention_head_dim,
            in_channels=teacher_config.in_channels,
            out_channels=teacher_config.out_channels,
            num_layers=args.student_layers,
            dropout=teacher_config.dropout,
            norm_num_groups=teacher_config.norm_num_groups,
            cross_attention_dim=teacher_config.cross_attention_dim,
            attention_bias=teacher_config.attention_bias,
            sample_size=teacher_config.sample_size,
            patch_size=teacher_config.patch_size,
            activation_fn=teacher_config.activation_fn,
            num_embeds_ada_norm=teacher_config.num_embeds_ada_norm,
            norm_type=teacher_config.norm_type,
            norm_elementwise_affine=getattr(teacher_config, "norm_elementwise_affine", False),
            norm_eps=getattr(teacher_config, "norm_eps", 1e-6),
            use_additional_conditions=teacher_config.use_additional_conditions,
            interpolation_scale=getattr(teacher_config, "interpolation_scale", None),
            caption_channels=getattr(teacher_config, "caption_channels", None),
        )
        # Load first N layers from teacher for initialization (convert to float32)
        teacher_state = pipe_teacher.transformer.state_dict()
        student_state = student_transformer.state_dict()
        load_dict = {}
        for k, v in teacher_state.items():
            if "transformer_blocks" in k:
                block_idx = int(k.split(".")[1])
                if block_idx >= args.student_layers:
                    continue
            if k in student_state and student_state[k].shape == v.shape:
                load_dict[k] = v.detach().float().clone()
        student_transformer.load_state_dict(load_dict, strict=False)
        student_transformer = student_transformer.to(device)
        student_transformer.train()

        # Build student pipeline (share VAE, text_encoder, scheduler with teacher)
        pipe_student = PixArtAlphaPipeline(
            tokenizer=pipe_teacher.tokenizer,
            text_encoder=pipe_teacher.text_encoder,
            vae=pipe_teacher.vae,
            transformer=student_transformer,
            scheduler=pipe_teacher.scheduler,
        )

    # --- DINOv2 (frozen) ---
    print("Loading DINOv2...")
    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id)
    dinov2_model.eval()
    dinov2_model = dinov2_model.to(device)
    for p in dinov2_model.parameters():
        p.requires_grad = False

    # --- Projector (trainable) ---
    dit_dim = pipe_teacher.transformer.config.num_attention_heads * pipe_teacher.transformer.config.attention_head_dim
    dinov2_dim = dinov2_model.config.hidden_size
    projector = DiT2DINOProjector(dit_dim=dit_dim, dinov2_dim=dinov2_dim).to(device)

    # --- Freeze all except student DiT and projector ---
    pipe_student.vae.eval()
    pipe_student.text_encoder.eval()
    for p in pipe_student.vae.parameters():
        p.requires_grad = False
    for p in pipe_student.text_encoder.parameters():
        p.requires_grad = False
    trainable_params = sum(p.numel() for p in pipe_student.transformer.parameters()) + sum(p.numel() for p in projector.parameters())
    print(f"Trainable: student DiT + projector ({trainable_params:,} params). All others frozen.")

    optimizer = torch.optim.AdamW(
        [
            {"params": pipe_student.transformer.parameters(), "lr": args.lr_student},
            {"params": projector.parameters(), "lr": args.lr_projector},
        ]
    )

    # --- W&B ---
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scheduler = pipe_teacher.scheduler
    latent_channels = pipe_teacher.transformer.config.in_channels

    pbar = tqdm(range(args.num_steps), desc="Distillation")

    for step in pbar:
        pipe_student.transformer.train()
        projector.train()
        optimizer.zero_grad()

        # Sample one prompt, generate K samples
        prompts = get_prompts(1, caption_pool, seed=args.seed + step)
        prompt = prompts[0]
        # Repeat prompt K times
        prompts_K = [prompt] * args.K

        with torch.no_grad():
            (
                prompt_embeds,
                prompt_attention_mask,
                _,
                _,
            ) = pipe_teacher.encode_prompt(
                prompts_K,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
                device=device,
                clean_caption=True,
            )

        # Prepare latents for teacher
        latents = pipe_teacher.prepare_latents(
            args.K,
            latent_channels,
            args.height,
            args.width,
            prompt_embeds.dtype,
            device,
            generator=torch.Generator(device).manual_seed(args.seed + step + 1),
        )

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        sample_size = pipe_teacher.transformer.config.sample_size
        if sample_size in (64, 128, 32):
            resolution = torch.tensor(
                [[args.height, args.width]], dtype=prompt_embeds.dtype, device=device
            ).repeat(args.K, 1)
            aspect_ratio = torch.tensor(
                [[float(args.height / args.width)]], dtype=prompt_embeds.dtype, device=device
            ).repeat(args.K, 1)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        scheduler.set_timesteps(args.num_inference_steps, device=device)

        capture_dict = {}
        is_final_step_ref = [False]

        # Run teacher forward, capture h_T and get final latents/images
        h_T, image_tensor, final_latents = run_teacher_denoising_capture_hidden(
            pipe_teacher,
            latents,
            prompt_embeds,
            prompt_attention_mask,
            added_cond_kwargs,
            device,
            capture_dict,
            is_final_step_ref,
        )

        # DINOv2 features for generated images (frozen)
        pil_images = [
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            for img in image_tensor
        ]
        dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
        pixel_values = dinov2_inputs["pixel_values"].to(device)

        with torch.no_grad():
            dinov2_outputs = dinov2_model(pixel_values=pixel_values)
            z_DINO = dinov2_outputs.last_hidden_state[:, 0, :]
            z_DINO = F.normalize(z_DINO, p=2, dim=-1)

        # Project h_T -> z_DiT (trainable)
        h_T_float = h_T.float()
        z_DiT = projector(h_T_float)

        # L_align: 1 - cos(g_phi(h_T), f(y))
        L_align = 1.0 - (z_DiT * z_DINO).sum(dim=-1).mean()

        # Energy: E_i = 1 - cos(z_DiT, z_DINO)
        E = 1.0 - (z_DiT * z_DINO).sum(dim=-1)

        # Gibbs weights: q_i = exp(-E_i/tau) / Z
        q = F.softmax(-E / args.tau, dim=0)

        # Sample random timestep for distillation
        timesteps = scheduler.timesteps
        t_idx = random.randint(0, len(timesteps) - 1)
        t = timesteps[t_idx]

        # Get z_t by adding noise to z_0 (final_latents) via forward diffusion
        # z_t = alpha_t * z_0 + sigma_t * noise
        noise = torch.randn_like(final_latents, device=device, dtype=torch.float32)
        t_batch = torch.tensor([t], dtype=torch.long, device=device).expand(args.K)
        z_t = scheduler.add_noise(final_latents.float(), noise, t_batch)
        z_t = z_t.to(pipe_student.transformer.dtype)

        latent_model_input = scheduler.scale_model_input(z_t, t)

        # Teacher noise prediction (no grad) - teacher is float16
        with torch.no_grad():
            latent_tea = latent_model_input.to(pipe_teacher.transformer.dtype)
            eps_tea = pipe_teacher.transformer(
                latent_tea,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t_batch,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            eps_tea = eps_tea.float()
            if pipe_teacher.transformer.config.out_channels // 2 == latent_channels:
                eps_tea = eps_tea.chunk(2, dim=1)[0]

        # Student noise prediction (with grad) - ensure float32 for student
        latent_stu = latent_model_input.float()
        eps_stu = pipe_student.transformer(
            latent_stu,
            encoder_hidden_states=prompt_embeds.float(),
            encoder_attention_mask=prompt_attention_mask,
            timestep=t_batch,
            added_cond_kwargs={k: v.float() if torch.is_tensor(v) else v for k, v in added_cond_kwargs.items()},
            return_dict=False,
        )[0]
        if pipe_student.transformer.config.out_channels // 2 == latent_channels:
            eps_stu = eps_stu.chunk(2, dim=1)[0]

        # L_distill: sum_i q_i * ||eps_stu_i - eps_tea_i||^2
        diff = eps_stu.float() - eps_tea.float()
        mse_per_sample = (diff ** 2).sum(dim=(1, 2, 3))
        L_distill = (q * mse_per_sample).sum()

        # L_KL: same as distill for Gaussian (D_KL ∝ ||mu_tea - mu_stu||^2)
        L_KL = (mse_per_sample).sum()

        # Full objective
        loss = L_distill + args.lambda_align * L_align + args.beta_kl * L_KL

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(pipe_student.transformer.parameters()) + list(projector.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            distill=f"{L_distill.item():.4f}",
            align=f"{L_align.item():.4f}",
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/L_distill": L_distill.item(),
                    "train/L_align": L_align.item(),
                    "train/L_KL": L_KL.item(),
                    "train/energy_mean": E.mean().item(),
                    "train/q_entropy": -(q * (q + 1e-8).log()).sum().item(),
                },
                step=step + 1,
            )

        if (step + 1) % args.log_every == 0:
            print(
                f"Step {step+1} | Loss: {loss.item():.4f} | L_distill: {L_distill.item():.4f} | "
                f"L_align: {L_align.item():.4f} | L_KL: {L_KL.item():.4f}"
            )

        if (step + 1) % args.save_every == 0:
            ckpt_dir = Path(args.output_dir)
            torch.save(pipe_student.transformer.state_dict(), ckpt_dir / f"student_step{step+1}.pt")
            torch.save(projector.state_dict(), ckpt_dir / f"projector_step{step+1}.pt")
            print(f"Saved checkpoints to {ckpt_dir}")

    torch.save(pipe_student.transformer.state_dict(), Path(args.output_dir) / "student_final.pt")
    torch.save(projector.state_dict(), Path(args.output_dir) / "projector_final.pt")
    print("Training complete.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
