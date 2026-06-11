#!/usr/bin/env python3
"""
Encoder-Scored Self-Distillation for Text-to-Image Flow Matching (SD3.5-M).

Asymmetric Timestep Distillation (ATD), v5 formulation:

  * One architecture for student θ (trainable; fp32 master weights, bf16
    autocast compute) and teacher ξ (fp32 EMA of θ):
        ξ ← m·ξ + (1−m)·θ      (m ≈ 0.999)
  * Projector maps VAE latents → frozen-encoder feature space; co-trained.
  * Frozen CLIP (--feature_encoder clip, recommended):
      - vision tower precomputes f̄_img per real COCO image (cached once)
        → the projector's L_align regression target;
      - text tower embeds the caption at each step → f_text, the scoring
        target, making s a learned CLIPScore proxy.
    (--feature_encoder dino: legacy DINOv2 CLS as both targets.)
  * No standard diffusion loss. No reward model network.

Per training step (one caption + paired f̄_img):

  Pass 1 (no grad):
    Roll out N teacher CFG-guided trajectories z_T → z_0 over K Euler
    steps. At each scoring index k, cache the teacher's Tweedie estimate
    x̃₀ = z_k − σ_k·v_ξ from the CFG-mixed velocity (guidance-distilled
    target, à la LCM/Guided Distillation).

  Pass 2 (single batched grad forward over the (k, i) grid):
    Student at the NOISIER state z_{k−Δ} (Δ ~ U[δmin, δmax], one per k):
      x̂₀ = z_{k−Δ} − σ_{k−Δ}·v_θ
      s   = cos(projector(x̂₀), f_text)
    Importance weights (DETACHED — pure constants, RWR/soft-RAFT style):
      w_traj = softmax_i(mean_k s / τ_N);  w_step = softmax_k(s / τ_K)
      W = w_traj ⊗ w_step
    Loss:
      L = Σ_{i,k} W·d(x̂₀, sg[x̃₀])          consistency in x₀ space,
                                              pseudo-Huber (iCT) or MSE
        + λ_reward · mean(1 − s)              ReFL-style alignment reward
                                              (grad → student + projector)
        + λ_align · (1 − cos(projector(VAE(real_img)), f̄_img))
                                              projector anchor: keeps the
                                              projector a faithful latent-
                                              space image encoder so the
                                              reward can't be hacked by
                                              projector collapse

  AdamW step (fp32 master weights), then fp32 EMA update of teacher.

Dataset:
  Defaults to COCO (image + caption pairs). f̄_img is precomputed once at
  startup for all unique images and held in RAM. Override with --coco_path;
  for non-COCO datasets, swap COCOPairDataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Four COCO-style prompts for periodic visual sample logging (--log_image_samples);
# four additional T2I-CompBench prompts chosen once at startup from --compbench_prompts_dir (--seed).
SAMPLE_PROMPTS = [
    "A photograph of a tabby cat sitting on a wooden table next to a cup of coffee",
    "A landscape painting of mountains reflecting in a still lake at sunset",
    "A red bicycle leaning against a brick wall covered in ivy",
    "A futuristic city skyline at night with neon signs reflected in wet streets",
]


def load_t2i_compbench_prompts(dataset_dir: Path) -> list[str]:
    """All non-empty lines from *.txt under T2I-CompBench_dataset (deduped, order kept)."""
    if not dataset_dir.is_dir():
        return []
    lines: list[str] = []
    for path in sorted(dataset_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            s = line.strip()
            if s:
                lines.append(s)
    return list(dict.fromkeys(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers   : EMA, hooks, DINO + VAE
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed(default_gpu: int = 0, nccl_timeout_min: int = 30):
    """Returns (rank, world_size, local_rank, device, is_main).

    If launched under torchrun, joins NCCL group and pins to LOCAL_RANK GPU.
    Otherwise runs single-process on cuda:default_gpu (or CPU).

    nccl_timeout_min: timeout for collective ops. Default 30 min, well above
    the 10-min default — needed for slow collective waits like the DINO
    precompute barrier when one rank is doing more work than the others.
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        from datetime import timedelta
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # GPU pinning: handle two LSF/SLURM allocation patterns transparently.
        #   (a) All N GPUs visible to every rank → device_count() == N.
        #       Each rank pins to cuda:LOCAL_RANK.
        #   (b) Per-process CUDA_VISIBLE_DEVICES (each rank sees ONLY its
        #       assigned GPU, exposed as cuda:0) → device_count() == 1.
        #       Each rank pins to cuda:0. Setting cuda:LOCAL_RANK in this
        #       case raises an "invalid argument" CUDA error on the first
        #       NCCL collective. Fail fast otherwise.
        n_visible = torch.cuda.device_count()
        assert n_visible >= 1, (
            f"rank {rank}: torch.cuda.device_count()=0; expected at least 1 visible GPU"
        )
        if n_visible == 1:
            cuda_idx = 0
        elif local_rank < n_visible:
            cuda_idx = local_rank
        else:
            raise RuntimeError(
                f"rank {rank}: LOCAL_RANK={local_rank} but only {n_visible} "
                f"GPU(s) visible. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        torch.cuda.set_device(cuda_idx)
        device = torch.device(f"cuda:{cuda_idx}")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=nccl_timeout_min),
            )
        return rank, world_size, local_rank, device, rank == 0
    # Single-process: CUDA is required. CPU fallback would silently mask
    # missing-GPU bugs and produce nonsensical wall-clock numbers.
    assert torch.cuda.is_available(), "CUDA required (no CPU fallback)"
    device = torch.device(f"cuda:{default_gpu}")
    torch.cuda.set_device(device)
    return 0, 1, default_gpu, device, True


def dist_barrier() -> None:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            # Pass device_ids so the implicit all_reduce that backs barrier
            # is placed on the correct GPU. Without this, PyTorch's NCCL
            # barrier allocates the work tensor on torch.cuda.current_device()
            # which can disagree with what NCCL has bound to (cluster setups
            # with cgroup/MIG isolation surface this as
            # "Cuda failure 1 'invalid argument'" on the first collective).
            dist.barrier(device_ids=[torch.cuda.current_device()])


@torch.no_grad()
def update_ema(teacher, student, decay: float) -> None:
    """ξ ← m·ξ + (1−m)·θ. Also copies buffers (e.g. norm running stats).

    Uses torch._foreach_* to fuse the per-parameter mul_/add_ into a single
    C++ call — order of magnitude faster than the Python-level loop on a
    multi-thousand-parameter SD3.5 transformer.
    """
    t_named = OrderedDict(teacher.named_parameters())
    s_named = OrderedDict(student.named_parameters())
    # Order matters: pair teacher and student tensors by name.
    t_list = [t_named[n] for n in s_named]
    s_list = [sp for sp in s_named.values()]
    torch._foreach_mul_(t_list, decay)
    torch._foreach_add_(t_list, s_list, alpha=1.0 - decay)
    t_buffers = OrderedDict(teacher.named_buffers())
    s_buffers = OrderedDict(student.named_buffers())
    for n, sb in s_buffers.items():
        t_buffers[n].copy_(sb)


@torch.no_grad()
def param_l2_distance(model_a, model_b) -> float:
    """‖θ − ξ‖₂ across all parameters. Diagnostic for EMA divergence."""
    a_params = OrderedDict(model_a.named_parameters())
    b_params = OrderedDict(model_b.named_parameters())
    total_sq = 0.0
    for n, ap in a_params.items():
        bp = b_params[n]
        total_sq += (ap.float() - bp.float()).pow(2).sum().item()
    return total_sq ** 0.5


@torch.no_grad()
def grad_l2_norm(parameters) -> float:
    """Total L2 norm of grads — used to capture POST-clip norm for the clip ratio."""
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
    return total_sq ** 0.5


@torch.no_grad()
def sample_teacher_images(
    teacher, vae, scheduler, prompt_embeds, pooled_embeds,
    height_lat, width_lat, latent_channels, num_steps, device, seed=0,
):
    """Run num_steps Euler steps with the teacher (EMA) and return decoded images
    in [0, 1] on CPU as a [B, 3, H, W] float tensor — ready for wandb.Image."""
    scheduler.set_timesteps(num_steps, device=device)
    sigmas = scheduler.sigmas.to(device, dtype=torch.float32)
    timesteps = scheduler.timesteps.to(device)
    bsz = prompt_embeds.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(
        bsz, latent_channels, height_lat, width_lat,
        device=device, dtype=torch.bfloat16, generator=g,
    )
    for k in range(num_steps):
        t = timesteps[k].expand(bsz)
        # autocast: teacher params are fp32 master weights; compute in bf16.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps = teacher(
                hidden_states=z, timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        sigma, sigma_next = sigmas[k], sigmas[k + 1]
        z = z + (sigma_next - sigma) * eps.float()
        z = z.to(torch.bfloat16)
    img = vae_decode(vae, z)              # [-1, 1]
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)  # [0, 1]
    return img.cpu()


def load_compbench_probe_prompts(
    t2i_compbench_root: str | Path,
    n_per_category: int = 30,
    categories: tuple[str, ...] = ("color", "shape", "texture", "spatial", "non-spatial", "complex"),
) -> list[dict]:
    """Load the first `n_per_category` prompts from each T2I-CompBench category
    train file. Returns a flat list of dicts with keys {category, idx, prompt}.
    Default 6 categories × 2 = 12 prompts spanning the compositionality axis.
    """
    root = Path(t2i_compbench_root)
    items: list[dict] = []
    for cat in categories:
        f = root / f"{cat}_train.txt"
        if not f.exists():
            print(f"  warn: missing {f}")
            continue
        with open(f) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        for j in range(min(n_per_category, len(lines))):
            items.append({"category": cat, "idx": j, "prompt": lines[j]})
    return items


@torch.no_grad()
def probe_compbench(
    pipe, teacher, projector_module, capture_xi: dict,
    prompts: list[dict], N: int, K: int,
    height: int, width: int, latent_channels: int,
    scheduler, device,
    f_bar_ref: torch.Tensor | None,
    text_encoders: list, offload_text_encoders: bool,
) -> list[dict]:
    """For each probe prompt, run N teacher trajectories at K Euler steps and
    measure last-block hidden state z_DiT properties + an energy proxy:

        norm(z_DiT)     L2 norm of the last-block hidden state per traj
        entropy(z_DiT)  Shannon entropy of softmax(z_DiT.flatten()) per traj
        energy          cos(projector(x̂_K), f̄_ref) per traj
                         (f̄_ref = mean of training f̄ cache)

    Returns one record per prompt with mean/std over N trajectories.
    """
    H_lat = height // pipe.vae_scale_factor
    W_lat = width // pipe.vae_scale_factor
    scheduler.set_timesteps(K, device=device)
    sigmas = scheduler.sigmas.to(device, dtype=torch.float32)
    timesteps = scheduler.timesteps.to(device)

    f_bar_n_ref = (
        F.normalize(f_bar_ref.float(), p=2, dim=-1) if f_bar_ref is not None else None
    )

    if offload_text_encoders:
        for enc in text_encoders:
            enc.to(device)

    results: list[dict] = []
    for item in prompts:
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=[item["prompt"]],
            prompt_2=[item["prompt"]],
            prompt_3=[item["prompt"]],
            do_classifier_free_guidance=False,
            device=device, num_images_per_prompt=1,
        )
        prompt_embeds_N = prompt_embeds.repeat(N, 1, 1)
        pooled_N = pooled.repeat(N, 1)

        # K-step Euler rollout, capturing the LAST iteration's h_xi
        z = torch.randn(
            N, latent_channels, H_lat, W_lat,
            device=device, dtype=torch.bfloat16,
        )
        last_h_xi = None
        for k in range(K):
            t = timesteps[k].expand(N)
            eps = teacher(
                hidden_states=z, timestep=t,
                encoder_hidden_states=prompt_embeds_N,
                pooled_projections=pooled_N,
                return_dict=False,
            )[0]
            last_h_xi = capture_xi["h"].detach().clone()  # [N, seq, dim]
            sigma = sigmas[k]
            sigma_next = sigmas[k + 1]
            z = z + (sigma_next - sigma) * eps.float()
            z = z.to(torch.bfloat16)

        # ── Hidden state metrics on z_DiT (last transformer block output) ──
        flat = last_h_xi.float().flatten(1)                 # [N, seq*dim]
        h_norms = flat.norm(dim=-1)                          # [N]
        log_p = F.log_softmax(flat, dim=-1)
        h_entropies = -(log_p.exp() * log_p).sum(dim=-1)    # [N]

        # ── Energy proxy via projector ──
        if projector_module is not None and f_bar_n_ref is not None:
            f_hat = projector_module(z)                       # [N, dino_dim]
            f_hat_n = F.normalize(f_hat.float(), p=2, dim=-1)
            energies = (f_hat_n * f_bar_n_ref.unsqueeze(0)).sum(dim=-1)  # [N]
        else:
            energies = torch.zeros(N, device=device)

        results.append({
            **item,
            "h_norm_mean": float(h_norms.mean().item()),
            "h_norm_std": float(h_norms.std(unbiased=False).item()),
            "h_entropy_mean": float(h_entropies.mean().item()),
            "h_entropy_std": float(h_entropies.std(unbiased=False).item()),
            "energy_mean": float(energies.mean().item()),
            "energy_std": float(energies.std(unbiased=False).item()),
        })

    if offload_text_encoders:
        for enc in text_encoders:
            enc.to("cpu")
        torch.cuda.empty_cache()

    return results


@torch.no_grad()
def save_training_samples(
    output_dir: Path,
    global_step: int,
    rank: int,
    item: dict,
    z_at_K: torch.Tensor,
    vae,
    metadata: dict,
    z_tweedie: torch.Tensor | None = None,
    tweedie_sigma: float | None = None,
    tweedie_step_idx: int | None = None,
) -> Path:
    """Save N teacher trajectory images (post K-step rollout) + optional
    Tweedie x̂₀ estimates, plus a JSON describing what's in them.

    Layout:
        <output_dir>/train_samples/step_NNNNNNNN_rN/
            ├── prompt.txt
            ├── metadata.json
            ├── traj_0_eulerK.png    # end of K-step Euler rollout (z_at_list[K][i])
            ├── traj_1_eulerK.png    #   blurry at low K — integration error
            ├── …
            ├── traj_0_tweedie.png   # x̂₀ ≈ z_t − σ·ε_θ at the cleanest score idx
            ├── traj_1_tweedie.png   #   sharper because no Euler integration
            └── …                    # only present if z_tweedie is provided
    """
    samples_dir = output_dir / "train_samples" / f"step_{global_step:08d}_r{rank}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.utils import save_image

    # End of K-step Euler rollout
    img_euler = vae_decode(vae, z_at_K)
    img_euler = ((img_euler + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
    for i in range(img_euler.shape[0]):
        save_image(img_euler[i], samples_dir / f"traj_{i}_eulerK.png")

    # Optional: Tweedie x̂₀ at the cleanest scoring step.
    # Sharper than the end of a coarse K-step Euler rollout because we skip
    # the discretized integration: x̂₀ = z_t − σ_t · ε_θ in one shot from a
    # low-σ input where the model's ε prediction is most accurate.
    if z_tweedie is not None:
        img_tw = vae_decode(vae, z_tweedie)
        img_tw = ((img_tw + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        for i in range(img_tw.shape[0]):
            save_image(img_tw[i], samples_dir / f"traj_{i}_tweedie.png")

    with open(samples_dir / "prompt.txt", "w") as f:
        f.write(item.get("caption", "") + "\n")

    full_meta = {
        "step": int(global_step),
        "rank": int(rank),
        "prompt": item.get("caption"),
        "image_id": int(item["image_id"]) if "image_id" in item else None,
        "real_image_path": item.get("image_path"),
        "num_trajectories": int(img_euler.shape[0]),
        "tweedie": (
            {"sigma": tweedie_sigma, "step_idx": tweedie_step_idx}
            if z_tweedie is not None else None
        ),
        **metadata,
    }
    with open(samples_dir / "metadata.json", "w") as f:
        json.dump(full_meta, f, indent=2)
    return samples_dir


def make_hook(capture: dict):
    """SD3 transformer block forward returns (encoder_hidden_states, hidden_states).
    We capture the hidden_states (image-side stream)."""
    def hook(module, input, output):
        # Strict: SD3's JointTransformerBlock must return (enc_hidden, img_hidden).
        # If this ever fires, the upstream model API has changed and we want to know.
        assert isinstance(output, tuple) and len(output) == 2, (
            f"hook output not a 2-tuple: type={type(output).__name__}"
        )
        capture["h"] = output[1]
    return hook


def dino_preprocess(x_minus1_to_1: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [-1, 1] → DINOv2 input ([0,1] → ImageNet-norm → 224²)."""
    x = (x_minus1_to_1 + 1.0) / 2.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


@torch.no_grad()
def dino_embed(dino_model, x_minus1_to_1: torch.Tensor) -> torch.Tensor:
    """DINOv2 CLS token, L2-normalized. Input: [B, 3, H, W] in [-1, 1]."""
    pixel_values = dino_preprocess(x_minus1_to_1.to(torch.float32))
    out = dino_model(pixel_values=pixel_values)
    cls = out.last_hidden_state[:, 0, :]
    return F.normalize(cls.float(), dim=-1)


@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """SD3.5 VAE decode → image in [-1, 1]."""
    # Direct attribute access: SD3.5 VAE has both scaling_factor and shift_factor.
    # If a different VAE config lacks shift_factor, fail loudly rather than zero-default.
    z = (latents.to(vae.dtype) / vae.config.scaling_factor) + vae.config.shift_factor
    img = vae.decode(z, return_dict=False)[0]
    return img.clamp(-1.0, 1.0).to(torch.float32)


import torch.nn as nn


class LatentToDINOProjector(nn.Module):
    """Maps SD3.5 VAE-space latents directly to DINOv2 feature space — no VAE
    decode, no DINO forward at training time.

    Architecture is REPA-analogous (REPA/models/sit.py:build_mlp):
        Linear(in, projector_dim) → SiLU
      → Linear(projector_dim, projector_dim) → SiLU
      → Linear(projector_dim, dino_dim)
    with projector_dim=2048 (REPA default; see REPA/models/mmdit.py:540).

    Implementation: 1×1 convs over [B, C, H, W] are mathematically identical
    to per-token Linears applied to flattened spatial tokens, so this matches
    REPA's per-token projection exactly while running faster on GPU.

    Input norm: parameter-free GroupNorm(num_groups=1) is equivalent to
    LayerNorm over channels, applied per spatial position. REPA doesn't need
    this because their projector input (DiT hidden states) is already
    LayerNormed by the model. Our input is the raw predicted clean latent
    x̂_0 = z_stu − σ·ε_θ whose scale varies with the timestep, so we
    explicitly normalize for training stability.

    Aggregation: global spatial mean of per-token DINO-projected features →
    single global vector to compare against the precomputed DINOv2 CLS f̄.
    (REPA aligns per-patch against DINO patch tokens; we have only CLS as
    target so we mean-pool. Different supervision target, same arch.)

    Input:  [B, latent_channels=16, H_lat, W_lat]   (16×64×64 at 512²)
    Output: [B, dino_dim=768]                       (UNNORMALIZED — caller does L2)

    Receives gradient through the loss path:
        cos(projector(x̂_0), f̄)  →  s  →  W  →  L = Σ W·d
    so the projector is co-trained with the student under a single objective.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        dino_dim: int = 768,
        projector_dim: int = 2048,
    ):
        super().__init__()
        # GroupNorm(num_groups=1) == LayerNorm over channels at each spatial
        # position. affine=False keeps it parameter-free, so it cannot be
        # undone by a learned scale/shift later in the network.
        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=latent_channels,
            eps=1e-6, affine=False,
        )
        # 3-layer MLP via 1×1 convs — mirrors REPA's build_mlp shape.
        self.token_proj = nn.Sequential(
            nn.Conv2d(latent_channels, projector_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(projector_dim, projector_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(projector_dim, dino_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.norm(x)
        feats = self.token_proj(x)         # [B, dino_dim, H, W]
        return feats.mean(dim=(2, 3))      # [B, dino_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: COCO image+caption pairs
# ─────────────────────────────────────────────────────────────────────────────

class COCOPairDataset(Dataset):
    """One example per (image, caption) pair (COCO has ~5 captions per image).

    If `image_size` is set (a square edge length in pixels), __getitem__ also
    loads + resizes + [-1, 1]-normalizes the image and returns it under
    the key "image_tensor" of shape [3, image_size, image_size]. Used for
    the projector alignment loss (VAE-encoded real images → f̄). Worker
    threads do the I/O and decode in parallel with GPU compute.
    """

    def __init__(
        self,
        coco_root: str,
        split: str = "val2017",
        max_examples: int = 0,
        image_size: int | None = None,
    ):
        coco_root = Path(coco_root)
        ann_path = coco_root / "annotations" / f"captions_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing COCO captions: {ann_path}")
        img_dir = coco_root / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing COCO images: {img_dir}")
        with open(ann_path) as f:
            data = json.load(f)
        id2path = {img["id"]: str(img_dir / img["file_name"]) for img in data["images"]}
        items = []
        for a in data["annotations"]:
            p = id2path.get(a["image_id"])
            if p and Path(p).exists():
                items.append({
                    "image_id": int(a["image_id"]),
                    "image_path": p,
                    "caption": a["caption"].strip(),
                })
        if max_examples and max_examples < len(items):
            items = items[:max_examples]
        self.items = items
        self.image_size = image_size
        self._transform = None
        if image_size is not None:
            from torchvision import transforms
            self._transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),                                # [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),            # [-1, 1]
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = dict(self.items[i])
        if self._transform is not None:
            with Image.open(item["image_path"]) as im:
                img = im.convert("RGB")
            item["image_tensor"] = self._transform(img)               # [3, H, W]
        return item


@torch.no_grad()
def extract_global_embedding(encoder_out) -> torch.Tensor:
    """Global image embedding from a vision-encoder forward, L2-normalized.

    CLIPVisionModelWithProjection → image_embeds (the joint text-image space
    that CLIP was contrastively trained in — the right target for
    prompt-image alignment). DINOv2 → CLS token of last_hidden_state.
    """
    if hasattr(encoder_out, "image_embeds"):
        emb = encoder_out.image_embeds
    else:
        emb = encoder_out.last_hidden_state[:, 0, :]
    return F.normalize(emb.float(), dim=-1)


@torch.no_grad()
def precompute_f_bar(
    dataset: COCOPairDataset, encoder_model, encoder_proc, device, batch_size: int = 32,
    rank: int = 0, world_size: int = 1, show_pbar: bool = True,
) -> dict[int, torch.Tensor]:
    """Compute the global feature embedding f̄ for each unique image_id
    (DINOv2 CLS or CLIP image_embeds, depending on the loaded encoder).

    With world_size > 1, each rank computes its interleaved shard
    (image_ids[rank::world_size] of the deterministic-sorted unique list).
    Returns {image_id: [feature_dim]} for that rank's shard only — the caller
    is expected to merge shards from all ranks via shared disk.
    """
    unique: dict[int, str] = {}
    for it in dataset.items:
        unique.setdefault(it["image_id"], it["image_path"])
    # Deterministic ordering across ranks: sort by image_id
    all_keys = sorted(unique.keys())
    my_keys = all_keys[rank::world_size]
    cache: dict[int, torch.Tensor] = {}
    desc = f"f̄ precompute[r{rank}/{world_size}]" if world_size > 1 else "f̄ precompute"
    for start in tqdm(
        range(0, len(my_keys), batch_size), desc=desc, disable=not show_pbar,
    ):
        batch_ids = my_keys[start:start + batch_size]
        imgs = [Image.open(unique[i]).convert("RGB") for i in batch_ids]
        proc = encoder_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        emb = extract_global_embedding(encoder_model(pixel_values=proc)).cpu()
        for j, image_id in enumerate(batch_ids):
            cache[image_id] = emb[j]
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--output_dir", default="checkpoints/self_distill_sd3m")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--feature_encoder", choices=["dino", "clip"], default="dino",
                        help="Vision encoder for the f̄ targets the projector is "
                             "scored against. 'dino' = DINOv2 CLS token (semantic "
                             "/ structural features). 'clip' = CLIP vision tower "
                             "image_embeds — the joint text-image space, i.e. "
                             "features explicitly trained for text alignment.")
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--clip_id", default="openai/clip-vit-large-patch14",
                        help="HF id of the CLIP model (vision tower + projection) "
                             "used when --feature_encoder clip.")
    parser.add_argument("--coco_path", default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO")
    parser.add_argument("--coco_split", default="val2017",
                        help="train2017 (118k) or val2017 (5k) — val2017 is fine for early dev.")
    parser.add_argument("--max_dataset_examples", type=int, default=0,
                        help="0 = use full split; otherwise truncate.")
    parser.add_argument("--dino_cache_path", default=None,
                        help="Where to load/save the f̄ cache. Default is keyed "
                             "by (feature_encoder, encoder id, coco_split, "
                             "max_dataset_examples) and shared across runs with "
                             "the same data config so the precompute happens "
                             "only once. dino keeps the legacy "
                             "cache/dino_f_bar/... path so existing caches are "
                             "reused; clip goes to cache/f_bar/clip__....pt.")
    # training
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Student LR. Low for fine-tuning a pretrained 2.5B model.")
    parser.add_argument("--projector_lr", type=float, default=1e-4,
                        help="Projector LR — much higher than student because the "
                             "projector starts from random init and needs to move a "
                             "lot. Matches REPA's `learning_rate=1e-4` for their "
                             "projector / DiT-from-scratch setup.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Student-only grad clip. Projector clipped separately.")
    parser.add_argument("--projector_grad_clip", type=float, default=5.0,
                        help="Projector-only grad clip. Higher than --grad_clip "
                             "so the projector's gradient isn't squeezed by the "
                             "much-larger student inside a combined clip budget.")
    # method hyperparameters (per spec)
    parser.add_argument("--N_traj", type=int, default=2,
                        help="Number of teacher trajectories per training step (spec default: 3).")
    parser.add_argument("--K_steps", type=int, default=6,
                        help="Number of teacher denoising rollout steps per trajectory.")
    parser.add_argument("--scoring_window", default="0.3,0.7",
                        help="Score in step_idx ∈ [lo·K, hi·K]. Mid-noise band.")
    parser.add_argument("--delta_min", type=int, default=1)
    parser.add_argument("--delta_max", type=int, default=2)
    # NB: --beta dropped. The DINO scoring path is now a single cosine
    # cos(projector(x̂_0), f̄); s_cons (teacher-student consistency under DINO)
    # was removed along with the VAE-decode + DINO-forward path.
    parser.add_argument("--projector_hidden_dim", type=int, default=2048,
                        help="Projector hidden dim (REPA's `projector_dim`, "
                             "default 2048; see REPA/models/mmdit.py:540).")
    parser.add_argument("--lambda_align", type=float, default=1.0,
                        help="Weight of the projector anchor loss "
                             "L_align = 1 − cos(projector(VAE_encode(real_image)), f̄_img). "
                             "0 disables. Keeps the projector a faithful "
                             "latent-space image encoder; without it the "
                             "L_reward path can be 'hacked' by a projector "
                             "that collapses toward the text-embedding "
                             "direction instead of reading the latent.")
    parser.add_argument("--tau_N", type=float, default=1.0)
    parser.add_argument("--tau_K", type=float, default=1.0)
    parser.add_argument("--score_norm", choices=["zscore", "none"], default="zscore",
                        help="Normalization of s before the Boltzmann softmaxes. "
                             "zscore: standardize within each softmax axis "
                             "(mean 0, std 1), making tau a scale-free "
                             "selectivity knob. Raw CLIP cosines spread only "
                             "a few hundredths across samples of one prompt, "
                             "so without this softmax(s/tau) is ~uniform and "
                             "the importance weighting is inert (W_entropy "
                             "pinned at log(N*K_w)).")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--distance", choices=["pseudo_huber", "mse"],
                        default="pseudo_huber",
                        help="Metric for d(x̂₀, sg[x̃₀]) in latent x₀ space. "
                             "pseudo_huber (iCT/LCM practice: robust to the "
                             "outlier pairs an aggressive Δ produces) or mse.")
    parser.add_argument("--huber_c", type=float, default=0.0,
                        help="Pseudo-Huber constant c in sqrt(‖Δ‖² + c²) − c. "
                             "0 = auto: 0.00054·sqrt(C·H·W) per iCT's "
                             "dimension-scaled heuristic.")
    parser.add_argument("--lambda_reward", type=float, default=1.0,
                        help="Weight of the explicit alignment reward "
                             "L_reward = mean(1 − s) over all (i,k) pairs. "
                             "This is the ONLY path by which the alignment "
                             "score reaches the student (ReFL-style, through "
                             "the projector proxy); W is detached. 0 disables.")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                        help="Classifier-Free Guidance scale for pass-1a teacher "
                             "rollout. 0 or 1 → no CFG (single conditional forward "
                             "per step, faster). >1 → CFG-guided rollout, doubles "
                             "the teacher forward count per Euler step but produces "
                             "much higher quality trajectories. SD3.5 paper default: "
                             "7.0. Recommended: lower --K_steps (e.g. 10) + CFG=7 "
                             "is faster AND better than K=28 no-CFG.")
    parser.add_argument(
        "--latent_matching_baseline",
        action="store_true",
        help="Ablate the encoder scoring: skip f̄ precompute, projector, "
             "L_reward and L_align. Loss is mean over (trajectory, timestep) "
             "pairs of d(x̂₀, x̃₀) with uniform weights (no softmax).",
    )
    # generation
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    # memory
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Halves student activation memory at ~30%% compute cost.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="bnb.AdamW8bit — cuts optimizer state ~4×.")
    parser.add_argument("--offload_text_encoders", action="store_true",
                        help="Move text encoders to CPU between encode_prompt calls (~15 GB).")
    # io
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_train_images_every", type=int, default=0,
                        help="Every N steps, decode + save the N teacher "
                             "trajectories' final images (z_at_list[K]) for the "
                             "rank's current prompt to "
                             "<output_dir>/train_samples/step_NNNNNNNN_r0/. "
                             "Saves alongside metadata.json (prompt, image_id, "
                             "per-traj s and W, etc.). 0 disables. Rank 0 only.")
    parser.add_argument("--probe_every", type=int, default=0,
                        help="Every N steps, run a CompBench probe: take 2 "
                             "prompts each from {color, shape, texture, spatial, "
                             "non-spatial, complex} = 12 prompts. For each, do "
                             "N teacher trajectories at K_steps, log per-prompt "
                             "and per-category h_xi norm + entropy and energy "
                             "(cos to mean f̄). For correlating compositionality "
                             "with these signals over training. 0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="dino-self-distill")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--compbench_prompts_dir",
        default="T2I-CompBench/T2I-CompBench_dataset",
        help="Directory of T2I-CompBench *.txt prompt files. With --log_image_samples, "
             "4 prompts are drawn once at startup (RNG: Random(--seed)).",
    )
    parser.add_argument("--log_image_samples", action="store_true",
                        help="Every save_every steps, sample the 4 fixed COCO-style prompts plus "
                             "4 fixed CompBench prompts (picked once per run) with the teacher "
                             "(EMA) and log to wandb.Image. Adds ~15-20 s per save interval.")
    parser.add_argument("--sample_inference_steps", type=int, default=28,
                        help="# steps for teacher image sampling (used only when "
                             "--log_image_samples is set).")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    rank, world_size, local_rank, device, is_main = setup_distributed(args.gpu)

    # Per-rank diagnostic print BEFORE any collective, so if dist_barrier
    # fails we still see the env each rank had.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"[r{rank}/{world_size}] device={device} "
        f"local_rank={local_rank} "
        f"current_device={torch.cuda.current_device()} "
        f"device_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={cvd}",
        flush=True,
    )

    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    dist_barrier()  # ensure output_dir exists on all ranks before they write to it

    # Rank-aware seed so different ranks sample different prompts & deltas.
    seed_offset = rank * 1_000_003
    torch.manual_seed(args.seed + seed_offset)
    random.seed(args.seed + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + seed_offset)
    if is_main:
        print(f"Device: {device} | rank {rank}/{world_size}")

    # ── Load SD3.5 pipeline (text encoders + scheduler + VAE + transformer) ─
    print("Loading SD3.5-Medium pipeline...")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)

    # Force frozen components to bfloat16. `torch_dtype=` on from_pretrained
    # does not always reach every submodule (e.g. CLIP's text_projection has
    # been observed to load in fp16 while the body loads in bf16, which then
    # trips a c10::Half != c10::BFloat16 mat-mul mismatch inside
    # encode_prompt). The transformer is excluded: it becomes the fp32-master
    # student below.
    for component in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2,
                      pipe.text_encoder_3):
        component.to(dtype=torch.bfloat16)

    for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    # ── Student (trainable; fp32 master weights, bf16 autocast compute) ────
    # Pure-bf16 training silently drops both small AdamW updates and the
    # (1−m)·(θ−ξ) EMA increments (bf16 ulp ≈ 4e-3 relative): standard
    # practice (diffusers LCM script, k-diffusion, EDM) is fp32 weights +
    # fp32 EMA with autocast for compute. All transformer forwards below run
    # under torch.autocast(bf16).
    student = pipe.transformer
    student.to(dtype=torch.float32)
    student.train()
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
        if is_main:
            print("Gradient checkpointing enabled on student.")

    # NOTE: the v5 loss d(x̂₀, sg[x̃₀]) supervises the full network output
    # (velocity head included), so norm_out/proj_out are inside the gradient
    # path and stay trainable — unlike the hidden-state-matching variants.

    # ── Teacher = deep copy of unwrapped student, frozen, EMA-tracked ──────
    # Important: copy BEFORE DDP wrap so teacher mirrors raw module shape.
    # fp32, like the student, so EMA increments don't round to zero.
    if is_main:
        print("Building teacher (deep copy of student; fp32 EMA-tracked)...")
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── DDP wrap (only when launched under torchrun with WORLD_SIZE > 1) ───
    student_module = student          # raw module — used for hooks, EMA, params
    student_ddp = student              # forward call — DDP if multi-rank, else raw
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=False: surface any genuine unused-parameter
        # bug at backward time. If gradient checkpointing legitimately leaves
        # some params unused on a given step, we'll see it explicitly.
        # device.index, NOT local_rank: when CUDA_VISIBLE_DEVICES is per-process
        # (single visible GPU), the right device id is 0 even if local_rank > 0.
        student_ddp = DDP(
            student, device_ids=[device.index], find_unused_parameters=False,
        )
        student_module = student_ddp.module
        if is_main:
            print(f"Wrapped student in DDP (world_size={world_size}, find_unused_parameters=False).")

    # ── Feature encoder (frozen) — only used at PRECOMPUTE TIME for f̄. At
    # training time we never call it; the projector replaces the VAE-decode +
    # encoder-forward path entirely. We still need encoder+processor here to
    # populate the f̄ cache if it doesn't exist on disk.
    encoder_model = None
    encoder_proc = None
    clip_text_model = None
    clip_tokenizer = None
    feature_dim = 768
    if not args.latent_matching_baseline:
        if args.feature_encoder == "clip":
            print(f"Loading CLIP vision tower {args.clip_id} "
                  "(only for f̄ precompute; not used at train time)...")
            from transformers import (
                CLIPVisionModelWithProjection, CLIPImageProcessor,
                CLIPTextModelWithProjection, CLIPTokenizer,
            )
            encoder_proc = CLIPImageProcessor.from_pretrained(args.clip_id)
            encoder_model = (
                CLIPVisionModelWithProjection.from_pretrained(args.clip_id)
                .to(device).eval()
            )
            # image_embeds dim (joint text-image space), NOT the tower hidden size
            feature_dim = encoder_model.config.projection_dim
            # Text tower IS used at train time: f_text = text_embeds(caption)
            # is the scoring target (s = learned CLIPScore proxy). Small
            # (~120M params), stays resident on GPU.
            clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_id)
            clip_text_model = (
                CLIPTextModelWithProjection.from_pretrained(args.clip_id)
                .to(device).eval()
            )
            for p in clip_text_model.parameters():
                p.requires_grad = False
        else:
            print("Loading DINOv2 (only for f̄ precompute; not used at train time)...")
            from transformers import Dinov2Model, BitImageProcessor
            encoder_proc = BitImageProcessor.from_pretrained(args.dinov2_id)
            encoder_model = Dinov2Model.from_pretrained(args.dinov2_id).to(device).eval()
            feature_dim = encoder_model.config.hidden_size
        for p in encoder_model.parameters():
            p.requires_grad = False
    elif is_main:
        print("Latent-matching baseline: skipping feature encoder load.")

    # ── Projector: VAE latent space → encoder feature space ───────────────
    # Receives gradient through the loss path: cos(projector(x̂_0), f̄) → s
    # → W → L = Σ W·d.  Co-trained with the student under a single objective.
    projector = None             # forward call (DDP wrapper if multi-rank)
    projector_module = None      # raw module (params + state_dict)
    if not args.latent_matching_baseline:
        latent_channels = student_module.config.in_channels
        # fp32 params (like the student); forward runs under bf16 autocast.
        projector_module = LatentToDINOProjector(
            latent_channels=latent_channels,
            dino_dim=feature_dim,
            projector_dim=args.projector_hidden_dim,
        ).to(device=device, dtype=torch.float32)
        projector_module.train()
        projector = projector_module
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            projector_ddp = DDP(
                projector_module, device_ids=[device.index],
                find_unused_parameters=False,
            )
            projector = projector_ddp
            projector_module = projector_ddp.module
        if is_main:
            n_proj = sum(p.numel() for p in projector_module.parameters())
            print(f"Projector (REPA-style 3-layer MLP): "
                  f"latent[{latent_channels}] → {args.projector_hidden_dim} → "
                  f"{args.projector_hidden_dim} → "
                  f"{args.feature_encoder.upper()}[{feature_dim}]; "
                  f"{n_proj:,} params.")

    # ── Dataset + DINO cache (parallel shard precompute, merge via disk) ───
    # Each rank computes 1/world_size of the unique images. After all ranks
    # finish (shard files written), every rank loads all shards. The canonical
    # merged cache is then written once by rank 0 so subsequent runs hit a
    # fast-path without recomputing.
    if is_main:
        print(f"Loading COCO ({args.coco_split}) from {args.coco_path}...")
    # Load image tensors per-sample only if we'll use them for L_align
    align_enabled = (
        not args.latent_matching_baseline and args.lambda_align > 0
    )
    dataset_image_size = args.height if align_enabled else None
    dataset = COCOPairDataset(
        args.coco_path, split=args.coco_split,
        max_examples=args.max_dataset_examples,
        image_size=dataset_image_size,
    )
    if is_main:
        print(f"Dataset: {len(dataset):,} (image, caption) pairs"
              + (f" (loading images at {dataset_image_size}² for L_align)" if align_enabled else ""))

    # Cache lives in a SHARED location keyed by (encoder, coco_split,
    # max_examples) — NOT in args.output_dir, so it survives across runs.
    if args.dino_cache_path:
        cache_path = Path(args.dino_cache_path)
    elif args.feature_encoder == "clip":
        clip_safe = args.clip_id.replace("/", "_")
        cache_path = (
            Path("cache") / "f_bar"
            / f"clip__{clip_safe}__{args.coco_split}__max{args.max_dataset_examples}.pt"
        )
    else:
        # Legacy path layout — existing DINO caches on disk stay valid.
        dino_safe = args.dinov2_id.replace("/", "_")
        cache_path = (
            Path("cache") / "dino_f_bar"
            / f"{dino_safe}__{args.coco_split}__max{args.max_dataset_examples}.pt"
        )
    if is_main and not args.latent_matching_baseline:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"f̄ cache path: {cache_path} (exists={cache_path.exists()})")
    if not args.latent_matching_baseline:
        dist_barrier()  # all ranks wait until rank 0 has created the parent dir

    if args.latent_matching_baseline:
        f_bar_cache = {}
    elif cache_path.exists():
        # Fast path: canonical cache already exists (resume / second run).
        if is_main:
            print(f"Loading existing f̄ cache from {cache_path}")
        raw = torch.load(cache_path, map_location="cpu", weights_only=False)
        f_bar_cache = {int(k): v for k, v in raw.items()}
        # Guard against pointing a clip run at a dino cache (or vice versa)
        any_dim = next(iter(f_bar_cache.values())).shape[-1]
        assert any_dim == feature_dim, (
            f"f̄ cache at {cache_path} has dim {any_dim} but "
            f"--feature_encoder {args.feature_encoder} expects {feature_dim}. "
            f"Wrong cache for this encoder config."
        )
    else:
        # Parallel shard precompute. Each rank shows its own progress bar.
        my_shard = precompute_f_bar(
            dataset, encoder_model, encoder_proc, device,
            rank=rank, world_size=world_size, show_pbar=is_main,
        )
        shard_path = Path(args.output_dir) / f"_dino_shard_rank{rank}_of{world_size}.pt"
        torch.save({k: v.cpu() for k, v in my_shard.items()}, shard_path)
        del my_shard
        # Wait for every rank's shard to be on disk
        dist_barrier()
        # All ranks load all shards and merge
        f_bar_cache = {}
        for r in range(world_size):
            sp = Path(args.output_dir) / f"_dino_shard_rank{r}_of{world_size}.pt"
            shard = torch.load(sp, map_location="cpu", weights_only=False)
            f_bar_cache.update({int(k): v for k, v in shard.items()})
        # Rank 0 writes the canonical merged cache, then cleans shard files.
        if is_main:
            torch.save(f_bar_cache, cache_path)
            print(f"Saved merged f̄ cache → {cache_path}")
            for r in range(world_size):
                sp = Path(args.output_dir) / f"_dino_shard_rank{r}_of{world_size}.pt"
                try:
                    sp.unlink()
                except FileNotFoundError:
                    pass
        # Make sure rank 0's cleanup completes before we proceed (filesystems
        # on lustre can be eventual-consistency-y for unlinks)
        dist_barrier()

    if args.latent_matching_baseline:
        dist_barrier()

    if is_main:
        if args.latent_matching_baseline:
            print("Latent-matching baseline: no f̄ cache (caption-only data).")
        else:
            print(f"f̄ cache ({args.feature_encoder}): "
                  f"{len(f_bar_cache)} unique image embeddings, dim {feature_dim}")

    # Rank-aware sampler so different ranks see disjoint prompt streams
    sampler = None
    if world_size > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=(sampler is None), sampler=sampler,
        num_workers=2, drop_last=True, collate_fn=lambda batch: batch[0],
    )
    data_iter = iter(dataloader)
    epoch_counter = 0

    # ── Optionally offload text encoders ───────────────────────────────────
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    if args.offload_text_encoders:
        for enc in text_encoders:
            enc.to("cpu")
        torch.cuda.empty_cache()
        print("Text encoders offloaded to CPU (~15 GB freed).")

    # ── Optimizer: student (requires_grad=True only) + projector ──────────
    student_opt_params = [p for p in student_module.parameters() if p.requires_grad]
    projector_opt_params = (
        list(projector_module.parameters()) if projector_module is not None else []
    )
    if is_main:
        n_student_total = sum(p.numel() for p in student_module.parameters())
        n_student_train = sum(p.numel() for p in student_opt_params)
        n_proj_train = sum(p.numel() for p in projector_opt_params)
        print(f"Trainable: student {n_student_train:,}/{n_student_total:,} "
              f"({100.0 * n_student_train / n_student_total:.2f} %); "
              f"projector {n_proj_train:,}.")
        print(f"Optimizer LRs: student={args.lr:.2e}, projector={args.projector_lr:.2e}")
        print(f"Grad clip: student={args.grad_clip}, projector={args.projector_grad_clip}")

    # Two param groups so student (fine-tune) and projector (from random init)
    # can have very different learning rates — projector typically needs ~20×
    # higher LR than the pretrained student.
    param_groups = [{"params": student_opt_params, "lr": args.lr, "name": "student"}]
    if projector_opt_params:
        param_groups.append(
            {"params": projector_opt_params, "lr": args.projector_lr, "name": "projector"}
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8,
        )
        if is_main:
            print("Using 8-bit AdamW.")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8,
        )

    def lr_lambda(s):
        return min(1.0, float(s) / max(1, args.num_warmup_steps))

    # LambdaLR multiplies each param group's `lr` by lr_lambda(step), so the
    # warmup schedule applies proportionally to both student and projector LRs.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Scoring window ─────────────────────────────────────────────────────
    win_lo, win_hi = (float(x) for x in args.scoring_window.split(","))
    K = args.K_steps
    score_idxs = list(range(
        max(1, round(win_lo * K)),
        min(K - 1, round(win_hi * K)) + 1,
    ))
    # No empty-window fallback: if K and scoring_window combine to give nothing,
    # that is a misconfiguration — fail loudly.
    assert score_idxs, (
        f"Empty scoring window: K={K}, scoring_window={args.scoring_window}. "
        f"Need at least one index in [round(win_lo*K), round(win_hi*K)]."
    )
    if is_main:
        print(f"K={K} | score_idxs={score_idxs} | Δ ∈ [{args.delta_min}, {args.delta_max}]")

    compbench_pool: list[str] = []
    compbench_four: list[str] = []
    if is_main and args.log_image_samples:
        compbench_root = Path(args.compbench_prompts_dir)
        if not compbench_root.is_absolute():
            compbench_root = Path(__file__).resolve().parent / compbench_root
        compbench_pool = load_t2i_compbench_prompts(compbench_root)
        rng_cb = random.Random(args.seed)
        if len(compbench_pool) >= 4:
            compbench_four = rng_cb.sample(compbench_pool, 4)
        elif compbench_pool:
            compbench_four = rng_cb.choices(compbench_pool, k=4)
            print(
                f"Warning: only {len(compbench_pool)} unique CompBench prompts; "
                "using random choice with replacement for the 4-slot panel."
            )
        else:
            print(
                f"Warning: no T2I-CompBench prompts found under {compbench_root}; "
                "W&B samples will use only the 4 COCO-style prompts."
            )
        if compbench_four:
            print("CompBench sample prompts (fixed for this run):")
            for j, p in enumerate(compbench_four):
                print(f"  [{j + 1}] {p}")

    # ── Latent shape ───────────────────────────────────────────────────────
    latent_channels = student_module.config.in_channels
    H_lat = args.height // pipe.vae_scale_factor
    W_lat = args.width // pipe.vae_scale_factor

    # Pseudo-Huber constant: iCT's dimension-scaled heuristic c = 0.00054·√D
    # over the flattened latent dim. Quadratic (MSE-like) for ‖Δ‖ ≪ c,
    # linear (L1-like) beyond — robust to outlier (i, k) pairs.
    huber_c = (
        args.huber_c if args.huber_c > 0
        else 0.00054 * (latent_channels * H_lat * W_lat) ** 0.5
    )
    if is_main and args.distance == "pseudo_huber":
        print(f"Pseudo-Huber c = {huber_c:.4f} "
              f"(latent dim {latent_channels * H_lat * W_lat})")

    # ── CompBench probe prompts (12 prompts: 2 × {color,shape,texture,spatial,
    #     non-spatial,complex}) and reference f̄ direction ──────────────────
    probe_items: list[dict] = []
    f_bar_probe_ref: torch.Tensor | None = None
    if args.probe_every > 0 and is_main:
        probe_items = load_compbench_probe_prompts(args.compbench_prompts_dir)
        if probe_items:
            print(f"Probe: {len(probe_items)} prompts from CompBench "
                  f"({len(set(it['category'] for it in probe_items))} categories)")
            for it in probe_items:
                print(f"  [{it['category']:>11s} #{it['idx']}] {it['prompt']}")
        # Energy reference: mean of the f̄ cache (a "global DINO direction"
        # for the training distribution). Each probe energy is cos vs this.
        if not args.latent_matching_baseline and len(f_bar_cache) > 0:
            f_bar_probe_ref = (
                torch.stack(list(f_bar_cache.values())).mean(dim=0).to(device).float()
            )

    # ── Pre-encode negative (empty) prompt embeds ONCE for CFG-guided
    #     pass-1a rollout. Static across the run, so we cache once.
    use_cfg = args.guidance_scale > 1.0
    neg_prompt_embeds: torch.Tensor | None = None
    neg_pooled_embeds: torch.Tensor | None = None
    if use_cfg:
        if args.offload_text_encoders:
            for enc in text_encoders:
                enc.to(device)
        with torch.no_grad():
            neg_prompt_embeds, _, neg_pooled_embeds, _ = pipe.encode_prompt(
                prompt=[""], prompt_2=[""], prompt_3=[""],
                do_classifier_free_guidance=False,
                device=device, num_images_per_prompt=1,
            )
        # Detach + on-device, will broadcast to N at use time
        neg_prompt_embeds = neg_prompt_embeds.detach()
        neg_pooled_embeds = neg_pooled_embeds.detach()
        if args.offload_text_encoders:
            for enc in text_encoders:
                enc.to("cpu")
            torch.cuda.empty_cache()
        if is_main:
            print(f"CFG-guided pass-1a rollout enabled (guidance_scale={args.guidance_scale}). "
                  f"Negative prompt embeds cached.")

    # ── W&B (rank 0 only) ──────────────────────────────────────────────────
    if not args.no_wandb and is_main:
        import wandb
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args),
        )

    # ── Training loop ──────────────────────────────────────────────────────
    scheduler = pipe.scheduler
    pbar = tqdm(total=args.num_steps, desc="self-distill", disable=not is_main)
    global_step = 0
    N = args.N_traj
    K_w = len(score_idxs)

    while global_step < args.num_steps:
        # Sample one (image, caption) example
        try:
            item = next(data_iter)
        except StopIteration:
            # New epoch — re-seed the DistributedSampler if used
            if sampler is not None:
                epoch_counter += 1
                sampler.set_epoch(epoch_counter)
            data_iter = iter(dataloader)
            item = next(data_iter)
        caption = item["caption"]
        if not args.latent_matching_baseline:
            # f̄_img: paired real image's encoder embedding — L_align target.
            f_bar = f_bar_cache[item["image_id"]].to(device).float()
            # f_score: what s = cos(projector(x̂₀), f_score) is computed
            # against. CLIP mode: the CAPTION's text embedding (joint-space),
            # so s is a learned CLIPScore proxy — the actual prompt-alignment
            # quantity. Scoring against the paired image's embedding instead
            # would be a single-image reconstruction signal (captions are
            # many-to-many). DINO mode keeps the legacy image-embedding score.
            if clip_text_model is not None:
                with torch.no_grad():
                    tok = clip_tokenizer(
                        [caption], padding="max_length", truncation=True,
                        max_length=clip_tokenizer.model_max_length,
                        return_tensors="pt",
                    ).to(device)
                    f_score = F.normalize(
                        clip_text_model(**tok).text_embeds.float(), dim=-1,
                    ).squeeze(0)                                       # [dim]
            else:
                f_score = f_bar

        # ── Encode prompt (frozen text encoders) ───────────────────────────
        if args.offload_text_encoders:
            for enc in text_encoders:
                enc.to(device)
        with torch.no_grad():
            prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
                prompt=[caption], prompt_2=[caption], prompt_3=[caption],
                do_classifier_free_guidance=False, device=device,
                num_images_per_prompt=1,
            )
        if args.offload_text_encoders:
            for enc in text_encoders:
                enc.to("cpu")
            torch.cuda.empty_cache()

        # Repeat conditioning across N trajectories for the rollout
        prompt_embeds_N = prompt_embeds.repeat(N, 1, 1)
        pooled_embeds_N = pooled_embeds.repeat(N, 1)

        # ── Set up scheduler timesteps for K-step rollout ───────────────────
        scheduler.set_timesteps(K, device=device)
        sigmas = scheduler.sigmas.to(device, dtype=torch.float32)  # [K+1]
        timesteps = scheduler.timesteps.to(device)                 # [K]

        # ── Pass 1a: roll out N teacher trajectories (no grad) ──────────────
        # z_at_list[k] has shape [N, C, H, W] in bfloat16 — state at step_idx k.
        # At each scoring index, cache the teacher's Tweedie target
        # x̃₀ = z_k − σ_k·v_ξ from the CFG-MIXED velocity. Distilling the
        # guided prediction (not the bare conditional one) follows
        # LCM / Guided Distillation, and means pass 2 needs zero extra
        # teacher forwards.
        score_idxs_set = set(score_idxs)
        tweedie_idx = max(score_idxs)
        save_images_this_step = (
            is_main and args.save_train_images_every > 0
            and global_step % args.save_train_images_every == 0
        )

        # x̃₀ targets cached fp32 as [N, C, H, W] per score_idx
        x0_tea_at: dict[int, torch.Tensor] = {}

        # CFG: prepare batched [uncond | cond] embeddings for the rollout
        if use_cfg:
            cfg_prompt_embeds = torch.cat([
                neg_prompt_embeds.repeat(N, 1, 1), prompt_embeds_N,
            ], dim=0)
            cfg_pooled = torch.cat([
                neg_pooled_embeds.repeat(N, 1), pooled_embeds_N,
            ], dim=0)

        z_at_list = []
        with torch.no_grad():
            z = torch.randn(
                N, latent_channels, H_lat, W_lat,
                device=device, dtype=torch.bfloat16,
            )
            # No .clone(): z is freshly allocated and the next iteration's
            # z = z + ...; z.to(bf16) chain produces a new tensor, so old
            # references in z_at_list remain valid views on distinct buffers.
            z_at_list.append(z)
            for k in range(K):
                # autocast: teacher params are fp32; compute in bf16.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if use_cfg:
                        # Batched [2N, ...] forward: uncond + cond at once
                        z_in = torch.cat([z, z], dim=0)        # [2N, ...]
                        t_in = timesteps[k].expand(2 * N)
                        eps_full = teacher(
                            hidden_states=z_in, timestep=t_in,
                            encoder_hidden_states=cfg_prompt_embeds,
                            pooled_projections=cfg_pooled,
                            return_dict=False,
                        )[0]                                    # [2N, ...]
                        eps_uncond, eps_cond = eps_full.chunk(2, dim=0)
                        eps = eps_uncond + args.guidance_scale * (eps_cond - eps_uncond)
                    else:
                        eps = teacher(
                            hidden_states=z, timestep=timesteps[k].expand(N),
                            encoder_hidden_states=prompt_embeds_N,
                            pooled_projections=pooled_embeds_N,
                            return_dict=False,
                        )[0]

                if k in score_idxs_set:
                    # Teacher Tweedie target (CFG-mixed velocity), fp32
                    x0_tea_at[k] = (
                        z.float() - sigmas[k] * eps.float()
                    ).detach()

                sigma = sigmas[k]
                sigma_next = sigmas[k + 1]
                z = z + (sigma_next - sigma) * eps.float()
                z = z.to(torch.bfloat16)
                z_at_list.append(z)

        # ── Pass 2 with PROJECTOR scoring (FULLY BATCHED) ────────────────
        # Per spec: "for each step k along the trajectory, sample a gap Δ_k"
        # → ONE Δ per k_teacher (not per (i, k_teacher) pair). All N
        # trajectories share the same k_student at a given k_teacher, AND
        # all (k_pos, i) pairs share the same prompt → concatenate the
        # whole [K_w, N] grid into a SINGLE [K_w*N, C, H, W] student
        # forward.
        # Gradient routing (v5):
        #   W is DETACHED — pure importance weights, RWR/soft-RAFT style.
        #   The alignment signal reaches student + projector ONLY through
        #   L_reward = mean(1 − s), s = cos(projector(x̂₀), f_score):
        #       L_reward → s → projector params
        #                    ↘ x̂₀ → v_θ → student
        #   L_align anchors the projector to faithful image encoding.
        optimizer.zero_grad(set_to_none=True)

        need_post = is_main and (global_step + 1) % args.log_every == 0

        # One delta per k_teacher
        deltas_per_kt: dict[int, int] = {
            kt: random.randint(args.delta_min, args.delta_max) for kt in score_idxs
        }
        for kt in score_idxs:
            assert kt - deltas_per_kt[kt] >= 0, (
                f"k_student={kt - deltas_per_kt[kt]} < 0 (k_teacher={kt}, "
                f"delta={deltas_per_kt[kt]}). Tighten --delta_max or move "
                f"--scoring_window's lower bound up."
            )

        teacher_idx_per_k: list[int] = list(score_idxs)
        student_idx_per_k: list[int] = [
            kt - deltas_per_kt[kt] for kt in score_idxs
        ]
        delta_per_k_list: list[int] = [
            deltas_per_kt[kt] for kt in score_idxs
        ]

        # ── Single batched student forward over (k_pos, i) — batch = K_w*N ──
        # Layout: outer k_pos, inner N. Reshape outputs as .view(K_w, N, ...).
        z_stu_batched = torch.cat(
            [z_at_list[ks] for ks in student_idx_per_k], dim=0,
        )                                                       # [K_w*N, C, H, W]
        t_stu_batched = torch.cat(
            [timesteps[ks].expand(N) for ks in student_idx_per_k], dim=0,
        )                                                       # [K_w*N]
        prompt_embeds_NK = prompt_embeds.repeat(N * K_w, 1, 1)
        pooled_embeds_NK = pooled_embeds.repeat(N * K_w, 1)

        # autocast: student/projector params are fp32; compute in bf16.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = student_ddp(
                hidden_states=z_stu_batched, timestep=t_stu_batched,
                encoder_hidden_states=prompt_embeds_NK,
                pooled_projections=pooled_embeds_NK,
                return_dict=False,
            )
        eps_stu_batched = out[0]                                # [K_w*N, C, H, W], has grad

        # ── Student Tweedie x̂₀ (fp32, keeps grad to v_θ) — used by BOTH the
        # consistency loss d(x̂₀, sg[x̃₀]) and the projector scoring path.
        sigma_per_kpos = torch.stack(
            [sigmas[ks] for ks in student_idx_per_k]
        )                                                       # [K_w], fp32
        sigma_stu_b = sigma_per_kpos.repeat_interleave(N).view(-1, 1, 1, 1)
        x_hat = (
            z_stu_batched.float() - sigma_stu_b * eps_stu_batched.float()
        )                                                       # [K_w*N, C, H, W]

        proj_out_norm_list: list[float] = []
        if not args.latent_matching_baseline:
            # ── Projector scoring (batched across all K_w*N pairs) ──
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f_hat = projector(x_hat)                        # [K_w*N, feat_dim]
            if need_post:
                proj_out_norm_list = (
                    f_hat.detach().float().norm(dim=-1).cpu().tolist()
                )
            f_hat_n = F.normalize(f_hat.float(), p=2, dim=-1)
            s_flat = (f_hat_n * f_score.unsqueeze(0)).sum(dim=-1)   # [K_w*N]
            s_tensor = s_flat.view(K_w, N).t()                       # [N, K_w], has grad

        # ── Weights W ─────────────────────────────────────────────────────
        # DETACHED: W is a pure importance weight (RWR/soft-RAFT style).
        # Backprop through softmax weights would reward routing mass onto
        # the easiest (lowest-d) pairs — i.e. push s DOWN exactly where
        # matching is hard. The alignment gradient flows only via L_reward.
        if args.latent_matching_baseline:
            w_u = 1.0 / float(N * K_w)
            W = torch.full((N, K_w), w_u, device=device, dtype=torch.float32)
            w_traj = torch.full((N,), 1.0 / N, device=device, dtype=torch.float32)
            w_step = torch.full((N, K_w), 1.0 / K_w, device=device, dtype=torch.float32)
            s_tensor = torch.zeros(N, K_w, device=device)
        else:
            s_detached = s_tensor.detach()
            S_traj = s_detached.mean(dim=1)                              # [N]
            s_step = s_detached                                          # [N, K_w]
            if args.score_norm == "zscore":
                # Standardize within each softmax axis so tau acts on a
                # unit-variance input regardless of the projector's output
                # scale. Degenerate all-equal scores → z = 0 → uniform W.
                S_traj = (
                    (S_traj - S_traj.mean())
                    / S_traj.std(unbiased=False).clamp_min(1e-6)
                )
                s_step = (
                    (s_step - s_step.mean(dim=1, keepdim=True))
                    / s_step.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                )
            w_traj = F.softmax(S_traj / args.tau_N, dim=0)               # [N]
            w_step = F.softmax(s_step / args.tau_K, dim=1)               # [N, K_w]
            W = w_traj.unsqueeze(1) * w_step                              # [N, K_w], no grad

        # ── Optional: save N teacher trajectory final images to disk ─────
        # Decode z_at_list[K] (the cleanest state, end of each rollout) for
        # the rank-0 prompt this step, plus per-traj scores and the prompt.
        # Useful for inspecting what the importance-weighting is operating on.
        # ── Optional: CompBench probe (rank 0 only). Runs the EMA teacher
        # against a fixed set of 12 compositional prompts (2 each from color,
        # shape, texture, spatial, non-spatial, complex) and logs:
        #   probe/<cat>/<idx>/{h_norm, h_entropy, energy}        (per-prompt)
        #   probe_cat/<cat>/{h_norm_mean, h_entropy_mean, …}     (per-category)
        # so we can scrub these vs compositionality over training.
        # CompBench probe disabled (12 prompts × K=10 = 120 sequential teacher
        # forwards every probe_every steps was non-trivial). Re-enable after
        # batching probe_compbench across prompts (one rollout at batch
        # n_prompts*N instead of looping).
        # if (
        #     is_main and args.probe_every > 0
        #     and global_step % args.probe_every == 0
        #     and probe_items
        # ):
        #     probe_results = probe_compbench(
        #         pipe=pipe,
        #         teacher=teacher,
        #         projector_module=projector_module,
        #         capture_xi=capture_xi,
        #         prompts=probe_items,
        #         N=N, K=K,
        #         height=args.height, width=args.width,
        #         latent_channels=latent_channels,
        #         scheduler=scheduler, device=device,
        #         f_bar_ref=f_bar_probe_ref,
        #         text_encoders=text_encoders,
        #         offload_text_encoders=args.offload_text_encoders,
        #     )
        #     # Per-prompt logging
        #     if not args.no_wandb:
        #         import wandb
        #         per_prompt_log = {}
        #         for r in probe_results:
        #             tag = f"{r['category']}/{r['idx']}"
        #             per_prompt_log[f"probe/{tag}/h_norm"] = r["h_norm_mean"]
        #             per_prompt_log[f"probe/{tag}/h_entropy"] = r["h_entropy_mean"]
        #             per_prompt_log[f"probe/{tag}/energy"] = r["energy_mean"]
        #             per_prompt_log[f"probe/{tag}/h_norm_std"] = r["h_norm_std"]
        #             per_prompt_log[f"probe/{tag}/h_entropy_std"] = r["h_entropy_std"]
        #             per_prompt_log[f"probe/{tag}/energy_std"] = r["energy_std"]
        #         # Per-category aggregates (mean over the 2 prompts in the cat)
        #         from collections import defaultdict
        #         by_cat: dict[str, list[dict]] = defaultdict(list)
        #         for r in probe_results:
        #             by_cat[r["category"]].append(r)
        #         for cat, rs in by_cat.items():
        #             per_prompt_log[f"probe_cat/{cat}/h_norm_mean"] = (
        #                 sum(r["h_norm_mean"] for r in rs) / len(rs)
        #             )
        #             per_prompt_log[f"probe_cat/{cat}/h_entropy_mean"] = (
        #                 sum(r["h_entropy_mean"] for r in rs) / len(rs)
        #             )
        #             per_prompt_log[f"probe_cat/{cat}/energy_mean"] = (
        #                 sum(r["energy_mean"] for r in rs) / len(rs)
        #             )
        #         wandb.log(per_prompt_log, step=global_step)
        #     # Persist on disk too (rank 0)
        #     probe_path = Path(args.output_dir) / "probe" / f"step_{global_step:08d}.json"
        #     probe_path.parent.mkdir(parents=True, exist_ok=True)
        #     with open(probe_path, "w") as f:
        #         json.dump({
        #             "step": int(global_step),
        #             "results": probe_results,
        #         }, f, indent=2)
        #     print(f"\n[step {global_step}] CompBench probe → {probe_path}")

        # Includes step 0 — captures the BASELINE state of the teacher (= EMA
        # init = student init = base SD3.5-M) and the random-init projector's
        # scoring before any optimizer updates.
        if (
            is_main and args.save_train_images_every > 0
            and global_step % args.save_train_images_every == 0
        ):
            # Pair-flat form (outer i, inner k_pos) for backward-compat with
            # metadata writers. Built only on save steps.
            student_idx_cache = [
                student_idx_per_k[k_pos] for _ in range(N) for k_pos in range(K_w)
            ]
            teacher_idx_cache = [
                teacher_idx_per_k[k_pos] for _ in range(N) for k_pos in range(K_w)
            ]
            delta_cache = [
                delta_per_k_list[k_pos] for _ in range(N) for k_pos in range(K_w)
            ]
            sample_meta = {
                "tau_N": float(args.tau_N),
                "tau_K": float(args.tau_K),
                "delta_per_pair": list(delta_cache),
                "teacher_idx_per_pair": list(teacher_idx_cache),
                "student_idx_per_pair": list(student_idx_cache),
                "S_traj": s_tensor.detach().mean(dim=1).cpu().tolist()
                          if not args.latent_matching_baseline else None,
                "s_per_pair": s_tensor.detach().cpu().tolist()
                              if not args.latent_matching_baseline else None,
                "w_traj": w_traj.detach().cpu().tolist(),
                "w_step": w_step.detach().cpu().tolist(),
                "W": W.detach().cpu().tolist(),
            }
            # x0_tea_at[tweedie_idx] is the teacher Tweedie x̃₀ already cached
            # during pass 1a (CFG-mixed) — no extra teacher forward needed.
            z_tweedie_stacked = x0_tea_at.get(tweedie_idx)

            saved_dir = save_training_samples(
                output_dir=Path(args.output_dir),
                global_step=global_step,
                rank=rank,
                item=item,
                z_at_K=z_at_list[K],     # [N, C, H, W] final teacher state (post-Euler-K)
                vae=pipe.vae,
                metadata=sample_meta,
                z_tweedie=z_tweedie_stacked,
                tweedie_sigma=float(sigmas[tweedie_idx].item()) if z_tweedie_stacked is not None else None,
                tweedie_step_idx=int(tweedie_idx) if z_tweedie_stacked is not None else None,
            )
            print(f"\n[step {global_step}] saved N={N} teacher trajectories "
                  f"(eulerK + tweedie@σ={float(sigmas[tweedie_idx]):.3f}) → {saved_dir}")

        # ── Consistency loss: Σ_{i,k_pos} W[i,k_pos] · d(x̂₀, sg[x̃₀]) ──
        # Student's Tweedie prediction at the NOISIER index k−Δ must match
        # the teacher's at k — consistency in x₀ space (CM/LCM-style), not
        # hidden-state space: the matched quantity has a fixed meaning
        # regardless of timestep conditioning, and there's no constant-
        # feature collapse mode.
        x0_stu_KN = x_hat.view(K_w, N, *x_hat.shape[1:])        # [K_w, N, C, H, W], grad
        x0_tea_stack = torch.stack(
            [x0_tea_at[kt] for kt in score_idxs], dim=0,
        )                                                       # [K_w, N, C, H, W], fp32

        diff = x0_stu_KN - x0_tea_stack                         # [K_w, N, C, H, W]
        if args.distance == "pseudo_huber":
            # iCT: d = sqrt(‖Δ‖² + c²) − c over the flattened latent
            sq = diff.pow(2).sum(dim=(2, 3, 4))                 # [K_w, N]
            d_kw_n = torch.sqrt(sq + huber_c * huber_c) - huber_c
        else:  # mse
            d_kw_n = diff.pow(2).mean(dim=(2, 3, 4))            # [K_w, N]

        d_per_pair = d_kw_n.t()                                 # [N, K_w]
        total_loss = (W * d_per_pair).sum()
        d_per_log = (
            d_per_pair.detach().reshape(-1).cpu().tolist() if need_post else []
        )

        # ── Alignment reward (ReFL-style, via the projector proxy) ────────
        # L_reward = mean(1 − s): the ONLY path by which the alignment score
        # reaches the student (through projector(x̂₀)) and the W-path scoring
        # quality reaches the projector. Uniform over all (i, k) pairs.
        reward_loss_value = 0.0
        if not args.latent_matching_baseline and args.lambda_reward > 0:
            L_reward = (1.0 - s_tensor).mean()
            total_loss = total_loss + args.lambda_reward * L_reward
            if need_post:
                reward_loss_value = float(L_reward.detach().item())

        # ── Projector anchor loss ──────────────────────────────────────────
        # L_align = 1 − cos(projector(VAE_encode(real_image)), f̄_img)
        # Keeps the projector a faithful latent-space image encoder, so
        # L_reward can't be satisfied by a projector that collapses toward
        # the text-embedding direction. The student is unaffected (z_real
        # never sees the student).
        align_loss_value = 0.0
        align_cos_value = 0.0
        if align_enabled and "image_tensor" in item:
            # No-grad VAE encode of the rank's real image
            img_t = item["image_tensor"].unsqueeze(0).to(device, dtype=pipe.vae.dtype)
            with torch.no_grad():
                posterior = pipe.vae.encode(img_t).latent_dist
                z_real = (
                    (posterior.mean - pipe.vae.config.shift_factor)
                    * pipe.vae.config.scaling_factor
                )
            z_real = z_real.float().detach()
            # WITH grad through projector
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f_real = projector(z_real)                              # [1, feat_dim]
            f_real_n = F.normalize(f_real.float(), p=2, dim=-1)
            align_cos = (f_real_n * f_bar.unsqueeze(0)).sum(dim=-1).squeeze()
            L_align = 1.0 - align_cos
            total_loss = total_loss + args.lambda_align * L_align
            if need_post:
                align_loss_value = float(L_align.detach().item())
                align_cos_value = float(align_cos.detach().item())

        total_loss.backward()
        loss_value = float(total_loss.detach().item()) if need_post else 0.0

        # ── Optimizer step + EMA ───────────────────────────────────────────
        # Clip student and projector SEPARATELY at their own thresholds.
        # Combined clipping had the projector's small grad squeezed by the
        # much-larger student grad inside a shared budget.
        # clip_grad_norm_ returns the PRE-clip total norm of the params it
        # was passed → free per-component pre-clip diagnostic.
        # need_post is hoisted to the top of the step (after optimizer.zero_grad)
        # so all the .item()/.cpu() syncs in pass 1b can be guarded too.

        if args.grad_clip > 0:
            s_norm_t = torch.nn.utils.clip_grad_norm_(
                student_opt_params, max_norm=args.grad_clip,
            )
            student_gn_pre = float(s_norm_t.item() if torch.is_tensor(s_norm_t) else s_norm_t)
        else:
            student_gn_pre = grad_l2_norm(student_module.parameters()) if need_post else 0.0

        if projector_opt_params and args.projector_grad_clip > 0:
            p_norm_t = torch.nn.utils.clip_grad_norm_(
                projector_opt_params, max_norm=args.projector_grad_clip,
            )
            projector_gn_pre = float(p_norm_t.item() if torch.is_tensor(p_norm_t) else p_norm_t)
        elif projector_opt_params:
            projector_gn_pre = grad_l2_norm(projector_module.parameters()) if need_post else 0.0
        else:
            projector_gn_pre = 0.0

        grad_norm_pre = (student_gn_pre ** 2 + projector_gn_pre ** 2) ** 0.5

        # Post-clip grad-norm is mathematically min(pre, max_norm); the explicit
        # grad_l2_norm walk over thousands of parameters with .item() per param
        # was the dominant cost of the logging path. Disabled for speed; the
        # log fields below default to 0.0 and clip_ratio to 1.0.
        # if need_post:
        #     student_gn_post = grad_l2_norm(student_module.parameters())
        #     projector_gn_post = (
        #         grad_l2_norm(projector_module.parameters())
        #         if projector_module is not None else 0.0
        #     )
        #     grad_norm_post = (student_gn_post ** 2 + projector_gn_post ** 2) ** 0.5
        # else:
        student_gn_post = 0.0
        projector_gn_post = 0.0
        grad_norm_post = 0.0

        optimizer.step()
        lr_scheduler.step()
        update_ema(teacher, student_module, decay=args.ema_decay)

        global_step += 1
        pbar.update(1)

        # ── Logging (rank 0 only) ──────────────────────────────────────────
        if is_main and global_step % args.log_every == 0:
            w_traj_entropy = -(w_traj * (w_traj + 1e-8).log()).sum().item()
            w_step_entropy = -(w_step * (w_step + 1e-8).log()).sum(dim=1).mean().item()

            # Entropy of the energy/score distribution across all (i, k) pairs.
            # Temperature-free: uses softmax(s) directly (τ=1) so the metric
            # depends only on the *relative spread* of the s values, not on
            # tau_K / tau_N. Uniform s → entropy ≈ log(N·K_w); concentrated s
            # → entropy → 0. Falling = projector becoming informative.
            with torch.no_grad():
                s_flat = s_tensor.detach().reshape(-1).float()
                log_p_e = F.log_softmax(s_flat, dim=0)
                p_e = log_p_e.exp()
                energy_entropy = float(-(p_e * log_p_e).sum().item())

            # param_l2_distance walks every parameter with .item() per param —
            # disabled for speed. Re-enable for EMA-divergence diagnostics.
            # param_div = param_l2_distance(student_module, teacher)
            sigma_t_log = float(sigmas[teacher_idx_per_k[0]].item())
            sigma_s_log = float(sigmas[student_idx_per_k[0]].item())
            clip_ratio = (grad_norm_post / grad_norm_pre) if grad_norm_pre > 0 else 1.0
            d_mean = sum(d_per_log) / len(d_per_log)

            base_metrics = {
                "train/loss": loss_value,
                "train/d_mean": d_mean,
                "train/d_min": min(d_per_log),
                "train/d_max": max(d_per_log),
                "train/W_max": W.max().item(),
                "train/W_min": W.min().item(),
                "train/W_entropy": -(W * (W + 1e-8).log()).sum().item(),
                "train/w_traj_entropy": w_traj_entropy,
                "train/w_step_entropy": w_step_entropy,
                "train/energy_entropy": energy_entropy,
                # Combined and per-component grad norms.
                "train/grad_norm_pre_clip": grad_norm_pre,
                "train/grad_norm_post_clip": grad_norm_post,
                "train/clip_ratio": clip_ratio,
                "train/student_grad_norm_pre_clip": student_gn_pre,
                "train/student_grad_norm_post_clip": student_gn_post,
                "train/lr_student": lr_scheduler.get_last_lr()[0],
                "train/lr_projector": (
                    lr_scheduler.get_last_lr()[1] if len(lr_scheduler.get_last_lr()) > 1
                    else 0.0
                ),
                # "train/param_l2_divergence": param_div,
                "train/avg_delta": float(sum(delta_per_k_list) / len(delta_per_k_list)),
                "train/sigma_teacher": sigma_t_log,
                "train/sigma_student": sigma_s_log,
            }
            if args.latent_matching_baseline:
                metrics = {
                    **base_metrics,
                    "train/latent_matching_baseline": 1.0,
                    "train/uniform_W": 1.0 / float(N * K_w),
                }
                postfix = {
                    "loss": f"{loss_value:.3g}",
                    "d": f"{d_mean:.3g}",
                    "g": f"{grad_norm_pre:.2g}",
                    # "Δθξ": f"{param_div:.2g}",
                }
            else:
                # Cosine cos(projector(x̂_0), f̄): the headline projector signal.
                s_mean_v = s_tensor.mean().item()
                s_std_v = s_tensor.std(unbiased=False).item()
                s_min_v = s_tensor.min().item()
                s_max_v = s_tensor.max().item()
                # Projector L2-norm of params (capacity used) and of outputs
                # (collapse / explosion check, BEFORE final L2 normalize).
                proj_p_norm = sum(
                    p.detach().float().pow(2).sum().item()
                    for p in projector_module.parameters()
                ) ** 0.5
                proj_out_norm_mean = (
                    sum(proj_out_norm_list) / len(proj_out_norm_list)
                    if proj_out_norm_list else 0.0
                )
                metrics = {
                    **base_metrics,
                    # Cosine vs f_score (CLIP text embed in clip mode) — the
                    # headline alignment signal (weights + L_reward)
                    "train/s_mean": s_mean_v,
                    "train/s_std": s_std_v,
                    "train/s_min": s_min_v,
                    "train/s_max": s_max_v,
                    # Explicit alignment reward (only score→student path)
                    "train/L_reward": reward_loss_value,
                    "train/lambda_reward": args.lambda_reward,
                    # Projector anchor loss (direct projector supervision)
                    "train/L_align": align_loss_value,
                    "train/align_cos": align_cos_value,
                    "train/lambda_align": args.lambda_align,
                    # Projector health
                    "train/projector_param_l2_norm": proj_p_norm,
                    "train/projector_output_l2_mean": proj_out_norm_mean,
                    # Projector grad norm — separately from student
                    "train/projector_grad_norm_pre_clip": projector_gn_pre,
                    "train/projector_grad_norm_post_clip": projector_gn_post,
                }
                postfix = {
                    "loss": f"{loss_value:.3g}",
                    "d": f"{d_mean:.3g}",
                    "s": f"{s_mean_v:.3g}",
                    "Lr": f"{reward_loss_value:.3g}",
                    "Lα": f"{align_loss_value:.3g}",
                    "g_s": f"{student_gn_pre:.2g}",
                    "g_p": f"{projector_gn_pre:.2g}",
                }
            if not args.no_wandb:
                import wandb
                wandb.log(metrics, step=global_step)
            pbar.set_postfix(postfix)

        # ── Checkpoint (rank 0 only; barrier so other ranks wait) ──────────
        if global_step % args.save_every == 0:
            if is_main:
                ckpt_path = Path(args.output_dir) / f"checkpoint_step{global_step}.pt"
                ckpt = {
                    "model": student_module.state_dict(),
                    "ema_model": teacher.state_dict(),
                    "opt": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": global_step,
                }
                if projector_module is not None:
                    ckpt["projector"] = projector_module.state_dict()
                torch.save(ckpt, ckpt_path)
                print(f"\nSaved {ckpt_path}")

                # Optional: sample SAMPLE_PROMPTS with the teacher (EMA) and log
                # to wandb. Costs ~15-20 s. Rank 0 only — non-main ranks just wait
                # at the dist_barrier below. No try/except: if sampling fails,
                # we want the full traceback, not a swallowed warning.
                if args.log_image_samples and not args.no_wandb:
                    import wandb
                    log_prompts = list(SAMPLE_PROMPTS) + compbench_four
                    log_captions = (
                        [f"[COCO] {p}" for p in SAMPLE_PROMPTS]
                        + [f"[CompBench] {p}" for p in compbench_four]
                    )
                    sample_t0 = torch.cuda.Event(enable_timing=True)
                    sample_t1 = torch.cuda.Event(enable_timing=True)
                    sample_t0.record()
                    with torch.no_grad():
                        if args.offload_text_encoders:
                            for enc in text_encoders:
                                enc.to(device)
                        sp_emb, _, sp_pool, _ = pipe.encode_prompt(
                            prompt=log_prompts,
                            prompt_2=log_prompts,
                            prompt_3=log_prompts,
                            do_classifier_free_guidance=False,
                            device=device,
                            num_images_per_prompt=1,
                        )
                        if args.offload_text_encoders:
                            for enc in text_encoders:
                                enc.to("cpu")
                            torch.cuda.empty_cache()
                        imgs = sample_teacher_images(
                            teacher, pipe.vae, scheduler,
                            sp_emb, sp_pool,
                            H_lat, W_lat, latent_channels,
                            num_steps=args.sample_inference_steps,
                            device=device, seed=args.seed,
                        )
                    sample_t1.record()
                    torch.cuda.synchronize()
                    sample_secs = sample_t0.elapsed_time(sample_t1) / 1000.0
                    wandb.log({
                        "samples/teacher_ema": [
                            wandb.Image(
                                imgs[i].permute(1, 2, 0).numpy(),
                                caption=log_captions[i],
                            )
                            for i in range(imgs.shape[0])
                        ],
                        "samples/elapsed_seconds": sample_secs,
                    }, step=global_step)
            dist_barrier()

    # ── Final (rank 0 only) ────────────────────────────────────────────────
    if is_main:
        final_path = Path(args.output_dir) / "checkpoint_final.pt"
        final_ckpt = {
            "model": student_module.state_dict(),
            "ema_model": teacher.state_dict(),
            "step": global_step,
        }
        if projector_module is not None:
            final_ckpt["projector"] = projector_module.state_dict()
        torch.save(final_ckpt, final_path)
        print(f"Final checkpoint: {final_path}")

    if not args.no_wandb and is_main:
        import wandb
        wandb.finish()

    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
