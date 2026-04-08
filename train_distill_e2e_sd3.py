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
            capture_dict["hidden"] = hid

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
            latent_model_input = latents 
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


def _scalar_timestep_value(t) -> float:
    if torch.is_tensor(t):
        return float(t.flatten()[0].item())
    return float(t)


def _resolve_device(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")

    cuda_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= cuda_count:
        raise ValueError(f"--gpu {gpu_index} is out of range; found {cuda_count} CUDA device(s).")

    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    return device


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = str(device) if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _parameter_l2_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def _gradient_l2_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def _tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "min": value.min().item(),
        "max": value.max().item(),
        "rms": value.pow(2).mean().sqrt().item(),
        "has_nan": bool(torch.isnan(value).any().item()),
        "has_inf": bool(torch.isinf(value).any().item()),
    }


def _gradient_debug_summary(named_parameters, top_k: int) -> dict:
    records = []
    total_sq = 0.0
    max_abs_grad = 0.0
    params_with_grad = 0
    params_without_grad = 0
    tensors_with_nonfinite_grad = 0
    nonfinite_grad_values = 0

    for name, param in named_parameters:
        if param.grad is None:
            params_without_grad += 1
            continue

        params_with_grad += 1
        grad = param.grad.detach().float()
        finite_mask = torch.isfinite(grad)
        has_nonfinite = not bool(finite_mask.all().item())
        grad_max_abs = grad.abs().max().item()
        max_abs_grad = max(max_abs_grad, grad_max_abs)

        record = {
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "max_abs_grad": grad_max_abs,
            "has_nonfinite_grad": has_nonfinite,
        }
        if has_nonfinite:
            tensors_with_nonfinite_grad += 1
            nonfinite_grad_values += int((~finite_mask).sum().item())
            record["grad_norm"] = None
        else:
            grad_norm = grad.norm().item()
            record["grad_norm"] = grad_norm
            total_sq += grad.pow(2).sum().item()
        records.append(record)

    records.sort(
        key=lambda item: (
            item["has_nonfinite_grad"],
            item["grad_norm"] if item["grad_norm"] is not None else float("inf"),
            item["max_abs_grad"],
        ),
        reverse=True,
    )

    return {
        "total_grad_norm": total_sq ** 0.5,
        "max_abs_grad": max_abs_grad,
        "params_with_grad": params_with_grad,
        "params_without_grad": params_without_grad,
        "tensors_with_nonfinite_grad": tensors_with_nonfinite_grad,
        "nonfinite_grad_values": nonfinite_grad_values,
        "top_parameters": records[:top_k],
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


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
    parser.add_argument("--beta_ent", type=float, default=0.00)
    parser.add_argument("--lr_student", type=float, default=1e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--debug_every",
        type=int,
        default=0,
        help="Write detailed gradient debug records every N steps (0 uses log_every)",
    )
    parser.add_argument("--debug_top_k", type=int, default=5)
    parser.add_argument("--coco_path", default="/data/gpfs/datasets/COCO")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--wandb_project", default="energy-based-dit")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.gpu)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Using device: {device}")

    caption_pool = None
    if args.coco_path and Path(args.coco_path).exists():
        caption_pool = load_coco_captions(args.coco_path)
        print(f"Loaded {len(caption_pool)} COCO captions")

    # --- Load teacher (frozen) ---
    print("Loading teacher (SD3.5 Large)...")
    pipe_teacher = StableDiffusion3Pipeline.from_pretrained(
        args.teacher_id,
        torch_dtype=torch.bfloat16,
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
    student_named_params = list(pipe_student.transformer.named_parameters())
    projector_named_params = list(projector.named_parameters())
    student_params = [param for _, param in student_named_params]
    projector_params = [param for _, param in projector_named_params]

    # --- Freeze all except student DiT and projector ---
    for c in [pipe_student.vae, pipe_student.text_encoder, pipe_student.text_encoder_2, pipe_student.text_encoder_3]:
        c.eval()
        for p in c.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": student_params, "lr": args.lr_student},
            {"params": projector_params, "lr": args.lr_projector},
        ]
    )

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scheduler = pipe_teacher.scheduler
    latent_channels = pipe_teacher.transformer.config.in_channels
    debug_every = args.debug_every if args.debug_every > 0 else args.log_every
    debug_path = Path(args.output_dir) / "debug_train_metrics.jsonl"

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
            generator=_make_generator(device, args.seed + step + 1),
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
        cos_sim = (z_DiT * z_DINO).sum(dim=-1)
        L_align = 1.0 - cos_sim.mean()
        E = 1.0 - cos_sim
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
        mse_per = (diff ** 2).mean(dim=(1, 2, 3))
        L_distill = (q * mse_per).sum()

        eps_flat_T = eps_tea.reshape(args.K, -1)
        eps_flat_S = eps_stu.float().reshape(args.K, -1)
        mu_T = eps_flat_T.mean(0)
        mu_S = eps_flat_S.mean(0)
        var_T = eps_flat_T.var(0).clamp(min=1e-4)
        var_S = eps_flat_S.var(0).clamp(min=1e-4)
        L_KL = 0.5 * (var_S / var_T + (mu_S - mu_T).pow(2) / var_T
                       - 1.0 + (var_T / var_S).log()).mean()

        L_entropy = (q * (q + 1e-8).log()).sum()

        loss = (L_distill + args.lambda_align * L_align
                + args.beta_kl * L_KL + args.beta_ent * L_entropy)

        loss.backward()
        student_grad_norm = _gradient_l2_norm(student_params)
        projector_grad_norm = _gradient_l2_norm(projector_params)
        should_debug = debug_every > 0 and ((step + 1) % debug_every == 0)
        student_grad_debug = None
        projector_grad_debug = None
        if should_debug:
            student_grad_debug = _gradient_debug_summary(student_named_params, args.debug_top_k)
            projector_grad_debug = _gradient_debug_summary(projector_named_params, args.debug_top_k)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student_params + projector_params,
            max_norm=1.0,
        )
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()
        student_grad_norm_post_clip = _gradient_l2_norm(student_params)
        projector_grad_norm_post_clip = _gradient_l2_norm(projector_params)
        grad_norm_post_clip = (
            student_grad_norm_post_clip ** 2 + projector_grad_norm_post_clip ** 2
        ) ** 0.5
        optimizer.step()
        student_param_norm = _parameter_l2_norm(student_params)
        projector_param_norm = _parameter_l2_norm(projector_params)
        grad_clip_ratio = grad_norm_post_clip / grad_norm if grad_norm > 0 else 1.0

        q_entropy = -(q * (q + 1e-8).log()).sum()
        q_effective_k = torch.exp(q_entropy)
        mse_mean = mse_per.mean()
        mse_std = mse_per.std(unbiased=False)
        energy_mean = E.mean()
        energy_std = E.std(unbiased=False)
        cos_mean = cos_sim.mean()
        cos_std = cos_sim.std(unbiased=False)
        h_t_norm_mean = h_T.float().norm(dim=-1).mean()
        metrics = {
            "train/loss": loss.item(),
            "train/L_distill": L_distill.item(),
            "train/L_align": L_align.item(),
            "train/L_KL": L_KL.item(),
            "train/L_entropy": L_entropy.item(),
            "train/weighted_L_align": (args.lambda_align * L_align).item(),
            "train/weighted_L_KL": (args.beta_kl * L_KL).item(),
            "train/weighted_L_entropy": (args.beta_ent * L_entropy).item(),
            "train/cos_mean": cos_mean.item(),
            "train/cos_std": cos_std.item(),
            "train/energy_mean": energy_mean.item(),
            "train/energy_std": energy_std.item(),
            "train/q_entropy": q_entropy.item(),
            "train/q_effective_k": q_effective_k.item(),
            "train/q_max": q.max().item(),
            "train/q_min": q.min().item(),
            "train/mse_per_mean": mse_mean.item(),
            "train/mse_per_std": mse_std.item(),
            "train/h_T_norm_mean": h_t_norm_mean.item(),
            "train/grad_norm": grad_norm,
            "train/grad_norm_post_clip": grad_norm_post_clip,
            "train/student_grad_norm": student_grad_norm,
            "train/projector_grad_norm": projector_grad_norm,
            "train/student_grad_norm_post_clip": student_grad_norm_post_clip,
            "train/projector_grad_norm_post_clip": projector_grad_norm_post_clip,
            "train/grad_clip_ratio": grad_clip_ratio,
            "train/student_param_norm": student_param_norm,
            "train/projector_param_norm": projector_param_norm,
            "train/energy_min": E.min().item(),
            "train/energy_max": E.max().item(),
            "train/cos_min": cos_sim.min().item(),
            "train/cos_max": cos_sim.max().item(),
            "train/latent_rms": latent_model_input.pow(2).mean().sqrt().item(),
            "train/final_latents_rms": final_latents.float().pow(2).mean().sqrt().item(),
            "train/eps_teacher_rms": eps_tea.pow(2).mean().sqrt().item(),
            "train/eps_student_rms": eps_stu.float().pow(2).mean().sqrt().item(),
            "train/diff_rms": diff.pow(2).mean().sqrt().item(),
            "train/timestep_idx": t_idx,
            "train/timestep_value": _scalar_timestep_value(t),
        }

        pbar.set_postfix(
            loss=f"{metrics['train/loss']:.1f}",
            align=f"{metrics['train/L_align']:.3f}",
            cos=f"{metrics['train/cos_mean']:.3f}",
            qmax=f"{metrics['train/q_max']:.3f}",
            grad=f"{metrics['train/grad_norm']:.2f}",
        )

        if not args.no_wandb:
            wandb.log(metrics, step=step + 1)

        if (step + 1) % args.log_every == 0:
            print(
                f"Step {step+1} | loss={metrics['train/loss']:.1f} | "
                f"distill={metrics['train/L_distill']:.1f} | "
                f"align={metrics['train/L_align']:.3f} (w={metrics['train/weighted_L_align']:.3f}) | "
                f"KL={metrics['train/L_KL']:.4f} (w={metrics['train/weighted_L_KL']:.4f}) | "
                f"ent={metrics['train/L_entropy']:.4f} (w={metrics['train/weighted_L_entropy']:.4f}) | "
                f"cos={metrics['train/cos_mean']:.3f}+/-{metrics['train/cos_std']:.3f} | "
                f"E={metrics['train/energy_mean']:.3f}+/-{metrics['train/energy_std']:.3f} "
                f"[{metrics['train/energy_min']:.3f},{metrics['train/energy_max']:.3f}] | "
                f"qmax={metrics['train/q_max']:.3f} qmin={metrics['train/q_min']:.3f} "
                f"qH={metrics['train/q_entropy']:.3f} q_eff={metrics['train/q_effective_k']:.2f} | "
                f"mse={metrics['train/mse_per_mean']:.1f}+/-{metrics['train/mse_per_std']:.1f} | "
                f"grad={metrics['train/grad_norm']:.2f}->{metrics['train/grad_norm_post_clip']:.2f} "
                f"(stu={metrics['train/student_grad_norm']:.2f}->{metrics['train/student_grad_norm_post_clip']:.2f}, "
                f"proj={metrics['train/projector_grad_norm']:.2f}->{metrics['train/projector_grad_norm_post_clip']:.2f}, "
                f"clip={metrics['train/grad_clip_ratio']:.5f}) | "
                f"param={metrics['train/student_param_norm']:.2f}/{metrics['train/projector_param_norm']:.2f} | "
                f"rms(lat={metrics['train/latent_rms']:.3f}, fin={metrics['train/final_latents_rms']:.3f}, "
                f"tea={metrics['train/eps_teacher_rms']:.3f}, stu={metrics['train/eps_student_rms']:.3f}, diff={metrics['train/diff_rms']:.3f}) | "
                f"t_idx={metrics['train/timestep_idx']} t={metrics['train/timestep_value']:.1f}"
            )

        if should_debug:
            debug_record = {
                "step": step + 1,
                "prompt": prompts[0],
                "metrics": metrics,
                "q_values": [float(x) for x in q.detach().cpu().tolist()],
                "energies": [float(x) for x in E.detach().cpu().tolist()],
                "cos_sim": [float(x) for x in cos_sim.detach().cpu().tolist()],
                "mse_per": [float(x) for x in mse_per.detach().cpu().tolist()],
                "tensor_stats": {
                    "h_T": _tensor_stats(h_T),
                    "z_DiT": _tensor_stats(z_DiT),
                    "z_DINO": _tensor_stats(z_DINO),
                    "latent_model_input": _tensor_stats(latent_model_input),
                    "final_latents": _tensor_stats(final_latents),
                    "eps_teacher": _tensor_stats(eps_tea),
                    "eps_student": _tensor_stats(eps_stu),
                    "diff": _tensor_stats(diff),
                },
                "gradient_debug": {
                    "student": student_grad_debug,
                    "projector": projector_grad_debug,
                },
            }
            _append_jsonl(debug_path, debug_record)
            print(
                f"[debug] step={step+1} wrote {debug_path.name} | "
                f"student_top={student_grad_debug['top_parameters'][0]['name'] if student_grad_debug['top_parameters'] else 'none'} | "
                f"projector_top={projector_grad_debug['top_parameters'][0]['name'] if projector_grad_debug['top_parameters'] else 'none'}"
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
