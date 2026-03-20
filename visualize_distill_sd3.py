#!/usr/bin/env python3
"""
Visualize distilled SD3.5 model: generate images, compute energy for teacher, compare with student.

- Loads distilled student (SD3.5 Medium) from checkpoint
- Generates 4 images per prompt with student and teacher
- Computes energy E = 1 - cos(z_DiT, z_DINO) for teacher images only
- Saves images and logs to output directory

Usage:
  python visualize_distill_sd3.py --checkpoint_dir checkpoints/distill_e2e_sd3 --log_dir logs/vis_sd3
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from projector import DiT2DINOProjector


def _make_hook(capture_dict: dict, is_final_step: list):
    def hook(module, input, output):
        if is_final_step[0]:
            enc, hid = output
            capture_dict["hidden"] = hid.detach()

    return hook


def generate_and_capture_hidden(
    pipe,
    prompt: str,
    num_images: int,
    height: int,
    width: int,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float = 7.0,
) -> tuple[list[Image.Image], torch.Tensor]:
    """
    Generate images and capture final hidden states for energy computation.
    Returns (pil_images, h_T).
    """
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae

    prompt_embeds, negative_prompt_embeds, pooled_embeds, negative_pooled_embeds = pipe.encode_prompt(
        prompt=[prompt] * num_images,
        prompt_2=[prompt] * num_images,
        prompt_3=[prompt] * num_images,
        negative_prompt=[""] * num_images,
        negative_prompt_2=[""] * num_images,
        negative_prompt_3=[""] * num_images,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)

    latent_channels = transformer.config.in_channels
    latents = pipe.prepare_latents(
        num_images,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator=torch.Generator(device).manual_seed(seed),
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    capture_dict = {}
    is_final = [False]

    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final))

    try:
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            is_final[0] = i == len(timesteps) - 1
            latent_model_input = torch.cat([latents] * 2)
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
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        with torch.no_grad():
            latents_for_decode = (latents / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)
            images = vae.decode(latents_for_decode, return_dict=False)[0]
        pil_images = pipe.image_processor.postprocess(images.float(), output_type="pil")
    finally:
        handle.remove()

    H_T = capture_dict["hidden"]
    h_T = H_T[num_images:].mean(dim=1)  # positive (text-conditioned) half only

    return pil_images, h_T


def compute_energies(
    h_T: torch.Tensor,
    pil_images: list[Image.Image],
    projector: torch.nn.Module,
    dinov2_model: torch.nn.Module,
    dinov2_processor,
    device: torch.device,
) -> list[float]:
    """Compute energy E = 1 - cos(z_DiT, z_DINO) for each image."""
    dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
    pixel_values = dinov2_inputs["pixel_values"].to(device)

    with torch.no_grad():
        dinov2_out = dinov2_model(pixel_values=pixel_values)
        z_DINO = dinov2_out.last_hidden_state[:, 0, :]
        z_DINO = F.normalize(z_DINO, p=2, dim=-1)

    z_DiT = projector(h_T.float())
    cos_sim = (z_DiT * z_DINO).sum(dim=-1)
    energies = (1.0 - cos_sim).cpu().tolist()
    return energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/distill_e2e_sd3_KL_1")
    parser.add_argument("--log_dir", default="logs/vis_sd3")
    parser.add_argument("--teacher_id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--student_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (e.g. 5000). Uses final if not set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir)
    if args.step is not None:
        student_ckpt = ckpt_dir / f"student_step{args.step}.pt"
        projector_ckpt = ckpt_dir / f"projector_step{args.step}.pt"
    else:
        student_ckpt = ckpt_dir / "student_step1000.pt"
        projector_ckpt = ckpt_dir / "projector_step1000.pt"

    if not student_ckpt.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt}")
    if not projector_ckpt.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_ckpt}")

    prompts = [
        "A red cube on top of a blue sphere, with a green cylinder under the sphere",
        "A yellow mug to the left of a laptop, while a spoon is inside the mug",
        "A cat sitting under a wooden chair, with a ball on top of the chair",
        "A bicycle leaning against the right side of a wall, with a backpack hanging on the left side",
        "A book placed between two candles, with a key behind the left candle",
        "A small dog in front of a large mirror, and a lamp behind the dog",
        "A plate on a table, with a fork to the left of the plate and a knife to the right",
        "A vase in the center, with three apples arranged in a triangle around it",
        "A clock above a window, and a plant below the window",
        "A chair facing a desk, with the chair's back toward the viewer",
    ][: args.num_prompts]

    # --- Load teacher (for comparison) ---
    print("Loading teacher (SD3.5 Large)...")
    pipe_teacher = StableDiffusion3Pipeline.from_pretrained(
        args.teacher_id,
        torch_dtype=torch.bfloat16,
    )
    pipe_teacher = pipe_teacher.to(device)

    # --- Load student with distilled weights ---
    print("Loading student (SD3.5 Medium) with distilled weights...")
    pipe_student = StableDiffusion3Pipeline.from_pretrained(
        args.student_id,
        torch_dtype=torch.bfloat16,
    )
    pipe_student.transformer.load_state_dict(torch.load(student_ckpt, map_location=device))
    pipe_student = pipe_student.to(device)

    # Share VAE, text encoders, scheduler
    pipe_student.vae = pipe_teacher.vae
    pipe_student.text_encoder = pipe_teacher.text_encoder
    pipe_student.text_encoder_2 = pipe_teacher.text_encoder_2
    pipe_student.text_encoder_3 = pipe_teacher.text_encoder_3
    pipe_student.tokenizer = pipe_teacher.tokenizer
    pipe_student.tokenizer_2 = pipe_teacher.tokenizer_2
    pipe_student.tokenizer_3 = pipe_teacher.tokenizer_3
    pipe_student.scheduler = pipe_teacher.scheduler

    # --- Projector (trained for teacher) ---
    dit_dim_teacher = (
        pipe_teacher.transformer.config.num_attention_heads
        * pipe_teacher.transformer.config.attention_head_dim
    )
    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id)
    dinov2_model = dinov2_model.to(device).eval()
    dinov2_dim = dinov2_model.config.hidden_size

    projector_teacher = DiT2DINOProjector(dit_dim=dit_dim_teacher, dinov2_dim=dinov2_dim).to(device)
    projector_teacher.load_state_dict(torch.load(projector_ckpt, map_location=device))
    projector_teacher.eval()

    all_results = []

    for p_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        result = {"prompt": prompt, "student": [], "teacher": []}

        # --- Student generation (no energy) ---
        out_stu = pipe_student(
            prompt=prompt,
            num_images_per_prompt=args.num_images,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device).manual_seed(args.seed + p_idx * 100),
        )
        imgs_stu = out_stu.images

        for i, img in enumerate(imgs_stu):
            fname = log_dir / f"prompt{p_idx:02d}_student_img{i:02d}.png"
            img.save(fname)
            result["student"].append({"image": str(fname)})

        # --- Teacher generation ---
        imgs_tea, h_T_tea = generate_and_capture_hidden(
            pipe_teacher,
            prompt,
            args.num_images,
            args.height,
            args.width,
            args.num_inference_steps,
            device,
            args.seed + p_idx * 100 + 1,
        )
        energies_tea = compute_energies(
            h_T_tea, imgs_tea, projector_teacher, dinov2_model, dinov2_processor, device
        )

        for i, (img, e) in enumerate(zip(imgs_tea, energies_tea)):
            fname = log_dir / f"prompt{p_idx:02d}_teacher_img{i:02d}_E{e:.4f}.png"
            img.save(fname)
            result["teacher"].append({"image": str(fname), "energy": e})

        all_results.append(result)

    # --- Save log ---
    log_path = log_dir / "energy_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "checkpoint_dir": str(args.checkpoint_dir),
                "step": args.step,
                "prompts": prompts,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved images and log to {log_dir}")
    print(f"Energy log: {log_path}")
    print("\nNote: Energy is computed only for teacher (large model) images.")


if __name__ == "__main__":
    main()
