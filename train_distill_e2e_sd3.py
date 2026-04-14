#!/usr/bin/env python3
"""
End-to-end Energy-Based Distillation for SD3.5 DiT.

Uses official Stable Diffusion 3.5 models:
- Teacher: stabilityai/stable-diffusion-3.5-large (8.1B params, frozen, bfloat16)
- Student: stabilityai/stable-diffusion-3.5-medium (2.5B params, trainable, float32)

Both share the same latent space. Projector g_phi is trained end-to-end with L_align.

Training objective (unchanged from original design):
  L = L_distill + lambda_align * L_align + beta_kl * L_KL + beta_ent * L_entropy

  where L_distill uses energy-based Gibbs weights q computed from DINOv2 alignment of
  teacher-generated images, and L_KL uses min-SNR-gamma weighting.

NOTE on alignment architecture: h_T is captured from the *teacher* transformer's last
block (not the student's), so L_align only directly trains the projector g_phi. The
student receives gradient only through L_distill and L_KL. The Gibbs weights q are
shaped by the projector, which softly up-weights distillation targets where the teacher
produced high-quality, semantically coherent images (low energy).

Usage:
  python train_distill_e2e_sd3.py --num_steps 10000 --K 4 --batch_size 1
"""

import argparse
import copy
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from projector import DiT2DINOProjector


# ── Dataset ────────────────────────────────────────────────────────────────────

class COCOCaptionDataset(Dataset):
    """Wraps COCO caption annotations as a proper Dataset over all captions."""

    _ANNOTATION_CANDIDATES = [
        "annotations/captions_train2017.json",
        "annotations/captions_train2014.json",
        "captions_train2017.json",
        "captions_train2014.json",
    ]

    def __init__(self, coco_path: str):
        path = Path(coco_path)
        ann_file = None
        if path.is_file():
            ann_file = path
        elif path.is_dir():
            for cand in self._ANNOTATION_CANDIDATES:
                candidate = path / cand
                if candidate.exists():
                    ann_file = candidate
                    break
        if ann_file is None:
            raise FileNotFoundError(f"No COCO captions JSON found under {coco_path}")
        with open(ann_file) as f:
            ann = json.load(f)
        self.captions = [a["caption"] for a in ann["annotations"]]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


class _ListDataset(Dataset):
    """Trivial Dataset wrapper around a list of strings (fallback when COCO unavailable)."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


_FALLBACK_PROMPTS = [
    "A photo of a cat sitting on a wooden table",
    "A landscape with mountains and a lake at sunset",
    "A bowl of fresh fruit on a kitchen counter",
    "A person walking a dog in the park",
]


# ── EMA ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    """Step the EMA model toward the current model weights."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ── Teacher hidden-state capture ───────────────────────────────────────────────

def _make_hook(capture_dict: dict, is_final_step: list):
    def hook(module, input, output):
        if is_final_step[0]:
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states)
            _enc, hid = output
            capture_dict["hidden"] = hid
    return hook


def run_teacher_denoising_capture_hidden(
    pipe, latents, prompt_embeds, pooled_embeds, device, capture_dict, is_final_step_ref
):
    """Run teacher denoising (no_grad), capture hidden states at the final step.

    Returns:
        h_T:          [B*K, dim]  – mean-pooled last-block hidden states
        pil_images:   list of B*K PIL images
        final_latents:[B*K, C, H, W] float32 clean latents
    """
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae

    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final_step_ref))

    # Keep latents in the teacher's dtype throughout denoising; return float32 at the end.
    latents = latents.to(transformer.dtype)

    try:
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            is_final_step_ref[0] = (i == len(timesteps) - 1)
            bsz = latents.shape[0]
            timestep = t.reshape(1).expand(bsz)

            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        with torch.no_grad():
            latents_decode = (latents / vae.config.scaling_factor) + getattr(
                vae.config, "shift_factor", 0.0
            )
            image = vae.decode(latents_decode, return_dict=False)[0]
        pil_images = pipe.image_processor.postprocess(image, output_type="pil")
    finally:
        handle.remove()

    if "hidden" not in capture_dict:
        raise RuntimeError("Failed to capture SD3 hidden states at final step")

    H_T = capture_dict["hidden"]      # [B*K, seq, dim]
    h_T = H_T.mean(dim=1)             # [B*K, dim]
    return h_T, pil_images, latents.float()  # final_latents back to float32 for student


# ── Misc helpers ───────────────────────────────────────────────────────────────

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
    gen_device = str(device) if device.type == "cuda" else "cpu"
    return torch.Generator(device=gen_device).manual_seed(seed)


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
        key=lambda r: (
            r["has_nonfinite_grad"],
            r["grad_norm"] if r["grad_norm"] is not None else float("inf"),
            r["max_abs_grad"],
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--output_dir", default="checkpoints/distill_e2e_sd3")
    parser.add_argument("--teacher_id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--student_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--coco_path", default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO")
    # training
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="Total number of optimizer update steps.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of distinct prompts per step. Each prompt generates K samples.")
    parser.add_argument("--K", type=int, default=4,
                        help="Number of samples per prompt for energy-based weighting.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_step", type=int, default=0,
                        help="Optimizer step to resume from (loads checkpoint_stepN.pt).")
    # loss weights
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Target Gibbs temperature (reached after warmup_steps).")
    parser.add_argument("--tau_max", type=float, default=5.0,
                        help="Initial tau. Linearly decays to --tau over --warmup_steps. "
                             "High tau = uniform weights (safe while projector is random). "
                             "Low tau = peaked weights (only once projector has learned).")
    parser.add_argument("--lambda_align", type=float, default=0.0,
                        help="Weight for L_align (projector training signal). "
                             "Matches REPA proj_coeff=0.5; 0.1 starves the projector and keeps "
                             "energy estimates noisy throughout training.")
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument("--beta_ent", type=float, default=0.0)
    parser.add_argument("--snr_gamma", type=float, default=5.0,
                        help="Min-SNR-gamma clamp for L_KL weighting.")
    # optimizer
    parser.add_argument("--lr_student", type=float, default=5e-5,
                        help="Student LR. 1e-5 is too slow for the student to track the teacher; "
                             "REPA uses 1e-4 (from scratch). 5e-5 with warmup is a safe middle "
                             "ground for a pretrained 2.5B model.")
    parser.add_argument("--lr_projector", type=float, default=1e-4,
                        help="Projector LR. Higher than student because projector starts random. "
                             "Matches REPA learning_rate=1e-4.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Linear LR warmup steps. 10%% of num_steps is appropriate for a "
                             "2.5B pretrained model; 500 (5%%) was too short.")
    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # diffusion
    parser.add_argument("--num_inference_steps", type=int, default=8,
                        help="Teacher denoising steps. 8 is sufficient for distillation targets "
                             "and uses 3.5× less peak activation memory than the default 28.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    # memory optimisations
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit AdamW. Cuts optimizer-state VRAM from "
                             "~20 GB to ~5 GB for a 2.5B model. Requires: pip install bitsandbytes")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on the student transformer. "
                             "Roughly halves activation memory during backward at ~30%% compute cost.")
    parser.add_argument("--offload_text_encoders", action="store_true",
                        help="Keep text encoders (T5 + CLIP×2, ~15 GB) on CPU; move to GPU "
                             "only during encode_prompt, then move back. Saves ~15 GB VRAM at "
                             "the cost of one CPU↔GPU transfer per step.")
    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N optimizer steps. 500 gives 20 checkpoints "
                             "over 10k steps, enough to catch regressions early.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--debug_every", type=int, default=0,
                        help="Write detailed gradient debug records every N steps (0 = use log_every).")
    parser.add_argument("--debug_top_k", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--wandb_project", default="energy-based-dit")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.gpu)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Using device: {device}")

    B = args.batch_size
    K = args.K
    BK = B * K

    # ── Dataset & DataLoader ───────────────────────────────────────────────────
    if args.coco_path and Path(args.coco_path).exists():
        train_dataset = COCOCaptionDataset(args.coco_path)
        print(f"Loaded {len(train_dataset):,} COCO captions")
    else:
        print("COCO path not found; using built-in fallback prompts.")
        train_dataset = _ListDataset(_FALLBACK_PROMPTS)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        # pin_memory not useful for string data
    )
    print(f"Dataset: {len(train_dataset):,} captions | DataLoader batch_size={B} | steps/epoch≈{len(train_dataloader):,}")

    # Persistent iterator; restarts automatically at epoch boundary
    data_iter = iter(train_dataloader)

    # ── Load teacher (frozen, bfloat16) ───────────────────────────────────────
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

    # ── Load student (trainable, bfloat16) ────────────────────────────────────
    # bfloat16 halves student VRAM vs float32 (5 GB instead of 10 GB).
    # PyTorch AdamW stores fp32 master copies of params internally for numerical
    # stability, so optimizer-state memory is unchanged by this; use --use_8bit_adam
    # to also halve that.
    print("Loading student (SD3.5 Medium)...")
    pipe_student = StableDiffusion3Pipeline.from_pretrained(
        args.student_id,
        torch_dtype=torch.bfloat16,
    )
    pipe_student.transformer = pipe_student.transformer.to(device)
    pipe_student.transformer.train()

    if args.gradient_checkpointing:
        pipe_student.transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for student transformer.")
    # Share frozen components with teacher (saves ~10 GB)
    pipe_student.vae = pipe_teacher.vae
    pipe_student.text_encoder = pipe_teacher.text_encoder
    pipe_student.text_encoder_2 = pipe_teacher.text_encoder_2
    pipe_student.text_encoder_3 = pipe_teacher.text_encoder_3
    pipe_student.tokenizer = pipe_teacher.tokenizer
    pipe_student.tokenizer_2 = pipe_teacher.tokenizer_2
    pipe_student.tokenizer_3 = pipe_teacher.tokenizer_3
    pipe_student.scheduler = pipe_teacher.scheduler

    # Optionally offload text encoders to CPU.  T5-XXL + CLIP×2 ≈ 15 GB; keeping
    # them on CPU and transferring only during encode_prompt frees that VRAM for
    # the student backward pass.
    _text_encoders = [pipe_teacher.text_encoder, pipe_teacher.text_encoder_2, pipe_teacher.text_encoder_3]
    if args.offload_text_encoders:
        for enc in _text_encoders:
            enc.to("cpu")
        torch.cuda.empty_cache()
        print("Text encoders offloaded to CPU (~15 GB freed).")

    # ── DINOv2 (frozen) ────────────────────────────────────────────────────────
    print("Loading DINOv2...")
    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id)
    dinov2_model.eval().to(device)
    for p in dinov2_model.parameters():
        p.requires_grad = False

    # ── Projector (trainable) ──────────────────────────────────────────────────
    dit_dim = (
        pipe_teacher.transformer.config.num_attention_heads
        * pipe_teacher.transformer.config.attention_head_dim
    )
    dinov2_dim = dinov2_model.config.hidden_size
    projector = DiT2DINOProjector(dit_dim=dit_dim, dinov2_dim=dinov2_dim).to(device)

    # Freeze shared frozen components (VAE, text encoders)
    for c in [pipe_student.vae, pipe_student.text_encoder, pipe_student.text_encoder_2, pipe_student.text_encoder_3]:
        c.eval()
        for p in c.parameters():
            p.requires_grad = False

    student_named_params = list(pipe_student.transformer.named_parameters())
    projector_named_params = list(projector.named_parameters())
    student_params = [p for _, p in student_named_params]
    projector_params = [p for _, p in projector_named_params]

    n_student = sum(p.numel() for p in student_params)
    n_proj = sum(p.numel() for p in projector_params)
    print(f"Trainable: student={n_student:,} params | projector={n_proj:,} params")

    # ── EMA of student transformer ─────────────────────────────────────────────
    ema_student = copy.deepcopy(pipe_student.transformer)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False
    update_ema(ema_student, pipe_student.transformer, decay=0.0)  # init from live weights

    # ── Optimizer ──────────────────────────────────────────────────────────────
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("Using 8-bit AdamW (bitsandbytes) — optimizer states in int8 (~5 GB instead of ~20 GB).")
        except ImportError:
            print("WARNING: bitsandbytes not found; falling back to standard AdamW. "
                  "Install with: pip install bitsandbytes")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [
            {"params": student_params, "lr": args.lr_student},
            {"params": projector_params, "lr": args.lr_projector},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Linear LR warmup; holds constant afterwards
    def lr_lambda(current_step: int) -> float:
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── Resume ─────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume_step > 0:
        ckpt_path = Path(args.output_dir) / f"checkpoint_step{args.resume_step}.pt"
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        pipe_student.transformer.load_state_dict(ckpt["model"])
        ema_student.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        global_step = ckpt["steps"]
        print(f"Resumed at optimizer step {global_step}")

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scheduler = pipe_teacher.scheduler
    latent_channels = pipe_teacher.transformer.config.in_channels
    debug_every = args.debug_every if args.debug_every > 0 else args.log_every
    debug_path = Path(args.output_dir) / "debug_train_metrics.jsonl"

    # ── Training loop ──────────────────────────────────────────────────────────
    # Each iteration of this loop is one *optimizer update step* (which may span
    # gradient_accumulation_steps forward passes).
    total_forward_passes = args.num_steps * args.gradient_accumulation_steps
    pbar = tqdm(total=args.num_steps, initial=global_step, desc="Distillation SD3.5")

    forward_pass = 0  # counts individual forward passes (for gradient accumulation)
    optimizer.zero_grad()

    while global_step < args.num_steps:
        pipe_student.transformer.train()
        projector.train()

        # ── Fetch batch of prompts from DataLoader ─────────────────────────────
        try:
            batch_prompts = next(data_iter)
        except StopIteration:
            # Epoch complete; restart the iterator (shuffle happens automatically)
            data_iter = iter(train_dataloader)
            batch_prompts = next(data_iter)

        # DataLoader returns a list/tuple of B strings
        if isinstance(batch_prompts, (list, tuple)):
            batch_prompts = list(batch_prompts)
        else:
            batch_prompts = [str(batch_prompts)]

        # Repeat each of the B prompts K times: [p1]*K + [p2]*K + ... → B*K total
        prompts_BK = [p for p in batch_prompts for _ in range(K)]

        # ── Encode prompts (no grad; shared text encoders, frozen) ────────────
        if args.offload_text_encoders:
            for enc in _text_encoders:
                enc.to(device)
        with torch.no_grad():
            prompt_embeds, _, pooled_embeds, _ = pipe_teacher.encode_prompt(
                prompt=prompts_BK,
                prompt_2=prompts_BK,
                prompt_3=prompts_BK,
                do_classifier_free_guidance=False,
                device=device,
                num_images_per_prompt=1,
            )
        if args.offload_text_encoders:
            for enc in _text_encoders:
                enc.to("cpu")
            torch.cuda.empty_cache()

        # ── Prepare initial latents (float32; unique noise per step) ──────────
        # Use forward_pass as part of the seed to ensure different noise each accumulation step
        latents = pipe_teacher.prepare_latents(
            BK,
            latent_channels,
            args.height,
            args.width,
            torch.float32,  # always float32; teacher casts internally to bfloat16
            device,
            generator=_make_generator(device, args.seed + forward_pass),
        )

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        capture_dict = {}
        is_final = [False]

        # ── Teacher denoising + hidden-state capture (no grad) ────────────────
        h_T, pil_images, final_latents = run_teacher_denoising_capture_hidden(
            pipe_teacher, latents, prompt_embeds, pooled_embeds, device, capture_dict, is_final
        )
        # h_T: [B*K, dim], pil_images: B*K PIL images, final_latents: [B*K, C, H, W]

        # ── DINOv2 features of teacher-generated images (no grad) ─────────────
        dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
        pixel_values = dinov2_inputs["pixel_values"].to(device)
        with torch.no_grad():
            dinov2_out = dinov2_model(pixel_values=pixel_values)
            z_DINO = dinov2_out.last_hidden_state[:, 0, :]  # [B*K, dino_dim]
            z_DINO = F.normalize(z_DINO, p=2, dim=-1)

        # ── Projector: map teacher hidden states → DINOv2 space ───────────────
        z_DiT = F.normalize(projector(h_T.float()), p=2, dim=-1)  # [B*K, dino_dim]
        cos_sim = (z_DiT * z_DINO).sum(dim=-1)  # [B*K]
        E = 1.0 - cos_sim                        # [B*K], energy per sample

        # ── Issue 5: tau warmup — start uniform, sharpen as projector learns ────
        # While the projector is randomly initialised, energies are meaningless.
        # Starting at tau_max≈uniform and decaying to tau prevents random q from
        # poisoning the distillation weights during the first warmup_steps steps.
        tau_frac = min(1.0, global_step / max(1, args.warmup_steps))
        current_tau = args.tau_max + (args.tau - args.tau_max) * tau_frac

        # ── Energy-based Gibbs weights (per prompt group of K) ─────────────────
        # Reshape to [B, K] so softmax is computed within each prompt group.
        E_BK = E.view(B, K)                                   # [B, K]
        q_BK = F.softmax(-E_BK / current_tau, dim=1)         # [B, K], sums to 1 over K

        L_align = (1.0 - cos_sim).mean()                      # scalar; trains projector only

        # ── Issue 2: per-sample independent timesteps ─────────────────────────
        # REPA samples t per sample; a single t per batch means the student only
        # trains at one noise level per step (28× higher variance, uneven coverage).
        timesteps = scheduler.timesteps
        t_indices = torch.randint(0, len(timesteps), (BK,), device=device)  # [B*K]
        t_batch   = timesteps[t_indices]                                      # [B*K] timestep values

        # Forward diffusion: z_t = σ_t * ε + (1 - σ_t) * x_0  (linear flow matching)
        # scale_noise uses an internal step_index scalar and does not support per-sample
        # timesteps, so we compute the noisy latents directly from sigmas.
        noise     = torch.randn_like(final_latents, dtype=torch.float32)
        sigmas_BK = scheduler.sigmas[t_indices].to(dtype=torch.float32)      # [B*K]
        z_t       = sigmas_BK.view(BK, 1, 1, 1) * noise \
                  + (1.0 - sigmas_BK.view(BK, 1, 1, 1)) * final_latents

        # ── Teacher noise prediction (frozen, bfloat16) ───────────────────────
        with torch.no_grad():
            eps_tea = pipe_teacher.transformer(
                hidden_states=z_t.to(pipe_teacher.transformer.dtype),
                timestep=t_batch,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0].float()

        # ── Student noise prediction (trainable, bfloat16) ────────────────────
        stu_dtype = pipe_student.transformer.dtype  # bfloat16
        eps_stu = pipe_student.transformer(
            hidden_states=z_t.to(stu_dtype),
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds.to(stu_dtype),
            pooled_projections=pooled_embeds.to(stu_dtype),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0].float()  # upcasted to float32 for loss computation

        # ── Losses ────────────────────────────────────────────────────────────
        diff      = eps_stu - eps_tea                       # [B*K, C, H, W], both float32
        mse_per   = (diff ** 2).mean(dim=(1, 2, 3))        # [B*K]

        # Normalise by teacher output magnitude so mse_per is a dimensionless
        # relative error (≈1 at init when student ≈ random, →0 at convergence).
        # Without this, L_distill and L_KL are in raw transformer output units
        # (typically 10–100×), while L_align ∈ [0, 2].  The normalisation puts
        # all three losses on the same O(1) scale, so lambda_align and beta_kl
        # become true relative-importance weights rather than scale-dependent knobs.
        # eps_tea was computed under torch.no_grad(), so mse_ref carries no grad.
        mse_ref   = eps_tea.pow(2).mean(dim=(1, 2, 3)).detach() + 1e-6  # [B*K]
        mse_per   = mse_per / mse_ref                      # relative MSE, O(1)

        mse_per_BK = mse_per.view(B, K)                    # [B, K]

        # Issue 6: stop-gradient on q so the projector's only training signal is
        # L_align, not a leaky gradient through the MSE weights. Without detach,
        # dL_distill/d(projector) = mse_i * dq_i/d(projector), which pushes q to
        # up-weight samples where the student already performs well — conflating
        # "image quality" with "student difficulty".
        L_distill = (q_BK.detach() * mse_per_BK).sum(dim=1).mean()

        # Issue 4 + 2: L_KL is now per-sample SNR-weighted uniform MSE.
        # With per-sample t (issue 2), each sample has its own σ_t, so L_KL and
        # L_distill are evaluated at different noise levels per sample — naturally
        # less redundant than the single-t case. The combined per-sample weight is
        #   q_i (energy-weighted) + beta_kl * w_i/K (SNR-weighted uniform floor),
        # where the floor prevents q from collapsing to a single sample.
        alpha_BK = 1.0 - sigmas_BK                                            # [B*K]
        snr_BK   = (alpha_BK / sigmas_BK.clamp(min=1e-6)) ** 2               # [B*K]
        w_BK     = snr_BK.clamp(max=args.snr_gamma) / snr_BK.clamp(min=1e-6) # [B*K]
        L_KL     = (w_BK * mse_per).mean()

        # L_entropy: optional regularization on q (negative entropy → penalizes spread)
        L_entropy = (q_BK * (q_BK + 1e-8).log()).sum(dim=1).mean()

        loss = (
            L_distill
            + args.lambda_align * L_align
            + args.beta_kl * L_KL
            + args.beta_ent * L_entropy
        )

        # ── Backward (gradient accumulation) ──────────────────────────────────
        (loss / args.gradient_accumulation_steps).backward()
        forward_pass += 1

        if forward_pass % args.gradient_accumulation_steps != 0:
            # Not yet time to update; keep accumulating gradients
            continue

        # ── Optimizer update ───────────────────────────────────────────────────
        student_grad_norm = _gradient_l2_norm(student_params)
        projector_grad_norm = _gradient_l2_norm(projector_params)

        should_debug = (global_step + 1) % debug_every == 0
        student_grad_debug = projector_grad_debug = None
        if should_debug:
            student_grad_debug = _gradient_debug_summary(student_named_params, args.debug_top_k)
            projector_grad_debug = _gradient_debug_summary(projector_named_params, args.debug_top_k)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            student_params + projector_params, max_norm=1.0
        )
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        update_ema(ema_student, pipe_student.transformer, decay=args.ema_decay)

        global_step += 1
        pbar.update(1)

        # ── Metrics ───────────────────────────────────────────────────────────
        student_grad_norm_post = _gradient_l2_norm(student_params)
        projector_grad_norm_post = _gradient_l2_norm(projector_params)
        grad_norm_post = (student_grad_norm_post ** 2 + projector_grad_norm_post ** 2) ** 0.5
        student_param_norm = _parameter_l2_norm(student_params)
        projector_param_norm = _parameter_l2_norm(projector_params)
        grad_clip_ratio = grad_norm_post / grad_norm if grad_norm > 0 else 1.0

        q_flat = q_BK.detach()
        q_entropy = -(q_flat * (q_flat + 1e-8).log()).sum(dim=1).mean()
        q_effective_k = torch.exp(q_entropy)
        mse_mean = mse_per.mean()
        mse_std = mse_per.std(unbiased=False)
        mse_ref_mean = mse_ref.mean()
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
            "train/q_max": q_flat.max().item(),
            "train/q_min": q_flat.min().item(),
            "train/mse_per_mean": mse_mean.item(),        # relative MSE, should be ≈1 at init
            "train/mse_per_std": mse_std.item(),
            "train/mse_ref_mean": mse_ref_mean.item(),  # teacher output magnitude (unnorm)
            "train/h_T_norm_mean": h_t_norm_mean.item(),
            "train/tau": current_tau,
            "train/sigma_t_mean": sigmas_BK.mean().item(),
            "train/snr_mean": snr_BK.float().mean().item(),
            "train/w_t_mean": w_BK.float().mean().item(),
            "train/timestep_value_mean": t_batch.float().mean().item(),
            "train/grad_norm": grad_norm,
            "train/grad_norm_post_clip": grad_norm_post,
            "train/student_grad_norm": student_grad_norm,
            "train/projector_grad_norm": projector_grad_norm,
            "train/student_grad_norm_post_clip": student_grad_norm_post,
            "train/projector_grad_norm_post_clip": projector_grad_norm_post,
            "train/grad_clip_ratio": grad_clip_ratio,
            "train/student_param_norm": student_param_norm,
            "train/projector_param_norm": projector_param_norm,
            "train/energy_min": E.min().item(),
            "train/energy_max": E.max().item(),
            "train/cos_min": cos_sim.min().item(),
            "train/cos_max": cos_sim.max().item(),
            "train/latent_rms": z_t.pow(2).mean().sqrt().item(),
            "train/final_latents_rms": final_latents.float().pow(2).mean().sqrt().item(),
            "train/eps_teacher_rms": eps_tea.pow(2).mean().sqrt().item(),
            "train/eps_student_rms": eps_stu.float().pow(2).mean().sqrt().item(),
            "train/diff_rms": diff.pow(2).mean().sqrt().item(),
            "train/lr_student": optimizer.param_groups[0]["lr"],
        }

        pbar.set_postfix(
            loss=f"{metrics['train/loss']:.1f}",
            align=f"{metrics['train/L_align']:.3f}",
            cos=f"{metrics['train/cos_mean']:.3f}",
            qmax=f"{metrics['train/q_max']:.3f}",
            grad=f"{metrics['train/grad_norm']:.2f}",
        )

        if not args.no_wandb:
            wandb.log(metrics, step=global_step)

        if global_step % args.log_every == 0:
            print(
                f"Step {global_step} | loss={metrics['train/loss']:.1f} | "
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
                f"t_mean={metrics['train/timestep_value_mean']:.1f} σ_mean={metrics['train/sigma_t_mean']:.3f} | "
                f"tau={metrics['train/tau']:.3f} | "
                f"lr={metrics['train/lr_student']:.2e}"
            )

        if should_debug:
            debug_record = {
                "step": global_step,
                "prompt": batch_prompts[0],
                "metrics": metrics,
                "q_values": q_flat.cpu().tolist(),
                "energies": E.detach().cpu().tolist(),
                "cos_sim": cos_sim.detach().cpu().tolist(),
                "mse_per": mse_per.detach().cpu().tolist(),
                "tensor_stats": {
                    "h_T": _tensor_stats(h_T),
                    "z_DiT": _tensor_stats(z_DiT),
                    "z_DINO": _tensor_stats(z_DINO),
                    "z_t": _tensor_stats(z_t),
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
            s_top = student_grad_debug["top_parameters"][0]["name"] if student_grad_debug["top_parameters"] else "none"
            p_top = projector_grad_debug["top_parameters"][0]["name"] if projector_grad_debug["top_parameters"] else "none"
            print(f"[debug] step={global_step} wrote {debug_path.name} | student_top={s_top} | projector_top={p_top}")

        if global_step % args.save_every == 0:
            ckpt = {
                "model": pipe_student.transformer.state_dict(),
                "ema": ema_student.state_dict(),
                "opt": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "steps": global_step,
                "args": vars(args),
            }
            ckpt_path = Path(args.output_dir) / f"checkpoint_step{global_step}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # ── Final save ─────────────────────────────────────────────────────────────
    final_ckpt = {
        "model": pipe_student.transformer.state_dict(),
        "ema": ema_student.state_dict(),
        "opt": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "steps": global_step,
        "args": vars(args),
    }
    torch.save(final_ckpt, Path(args.output_dir) / "checkpoint_final.pt")
    # Bare state dicts for easy inference loading
    torch.save(pipe_student.transformer.state_dict(), Path(args.output_dir) / "student_final.pt")
    torch.save(ema_student.state_dict(), Path(args.output_dir) / "student_ema_final.pt")
    torch.save(projector.state_dict(), Path(args.output_dir) / "projector_final.pt")
    print("Done.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
