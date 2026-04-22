#!/usr/bin/env python3
"""
Feature-Matching Knowledge Distillation for Sana: 1.6B -> 600M.

Teacher: Efficient-Large-Model/Sana_1600M_1024px_diffusers  (20 blocks, inner_dim=2240)
Student: Efficient-Large-Model/Sana_600M_1024px_diffusers   (28 blocks, inner_dim=1152)

Both models share the same DC-AE VAE (32 latent channels, 32x spatial
compression), the same Gemma-2-2B text encoder, and the same flow-matching
formulation.

Mathematics
-----------
Sana operates on 32-channel DC-AE latents with patch_size=1.  Given prompt c,
clean latent x_0, noise eps ~ N(0,I), and interpolant fraction sigma_t:

    z_t = (1 - sigma_t) * x_0 + sigma_t * eps

Each transformer predicts a velocity (or noise) field through L blocks that
produce hidden states h^l of shape [B, seq, D] where seq = H'*W' tokens.

Last-layer feature matching:
  A single linear projector P: R^{D_s} -> R^{D_t} (1152 -> 2240) aligns the
  student's final block features to the teacher's.

    L_feat = || normalize(P(h_s^{-1})) - normalize(h_t^{-1}) ||^2     (MSE)
        or  tau^2 * KL( softmax(h_t^{-1}/tau) || softmax(P(h_s^{-1})/tau) )  (KL)

  MSE on L2-normalised features equals 2*(1 - cosine_similarity).

Projector scheduling:
  The projector starts random; feeding random gradients into the student early
  on is harmful.  Two mechanisms:

  1. L_feat weight ramps linearly from 0 to beta_feat over warmup_feat_steps.
  2. Projector LR (--lr_projector, default 5e-4) >> student LR (--lr, 5e-6).

Timestep sampling (--timestep_sampling):
  uniform:      index t ~ U{0, ..., T-1}
  logit_normal: sigma ~ sigmoid(N(mu, s^2))  (SD3-style, flow-matching friendly)

Multi-GPU:
  torchrun --nproc_per_node=N train_distill_sd3.py [args]

Single GPU:
  python train_distill_sd3.py [args]
"""

import argparse
import copy
import datetime
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers import SanaPipeline
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# -- Dataset -------------------------------------------------------------------

class COCOCaptionDataset(Dataset):
    """Wraps COCO caption annotations as a Dataset over all captions."""

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


# -- EMA ----------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# -- DDP setup ----------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(
            "nccl", timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, world_size, local_rank, True
    else:
        if not torch.cuda.is_available():
            return torch.device("cpu"), 0, 1, 0, False
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device, 0, 1, 0, False


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


# -- Feature capture (single last block, tensor output) ------------------------

class LastBlockCapture:
    """Persistent forward hook that captures the output of a single transformer block.

    Sana transformer blocks return a single hidden-state tensor (unlike SD3's
    JointTransformerBlock which returns a tuple), so the hook is simple.
    """

    def __init__(self):
        self.feature = None
        self._handle = None

    def register(self, block: nn.Module):
        def _hook(module, input, output):
            # SanaTransformerBlock returns a single Tensor [B, seq, D].
            self.feature = output[0] if isinstance(output, tuple) else output
        self._handle = block.register_forward_hook(_hook)

    def clear(self):
        self.feature = None

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# -- Loss primitives -----------------------------------------------------------

def softmax_kl(student_logits, teacher_logits, tau):
    """KL(softmax(teacher/tau) || softmax(student/tau)) summed over last dim."""
    log_p_s = F.log_softmax(student_logits / tau, dim=-1)
    p_t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)


def feature_matching_loss(h_student, h_teacher, projector,
                          loss_type="mse", tau=1.0):
    """Align projected student final-block features to teacher final-block features.

      mse: MSE on L2-normalised features, equal to 2*(1 - cos_sim) token-wise.
      kl:  tau^2 * KL( softmax(h_t/tau) || softmax(P(h_s)/tau) ) over feature dim.
    """
    h_t = h_teacher.float()                     # [B, seq, D_t]
    h_s_proj = projector(h_student.float())     # [B, seq, D_t]

    h_t_norm = F.normalize(h_t, dim=-1)
    h_s_norm = F.normalize(h_s_proj, dim=-1)

    if loss_type == "mse":
        loss = (h_s_norm - h_t_norm).pow(2).mean()
    elif loss_type == "kl":
        loss = (tau ** 2) * softmax_kl(h_s_proj, h_t, tau).mean()
    else:
        raise ValueError(f"Unknown feature loss_type: {loss_type}")

    cos_mean = (h_s_norm * h_t_norm).sum(dim=-1).mean().item()
    return loss, cos_mean


# -- Timestep sampling ---------------------------------------------------------

def sample_timesteps(batch_size, num_steps, device, method="uniform",
                     mu=0.0, sigma=1.0):
    if method == "uniform":
        return torch.randint(0, num_steps, (batch_size,), device=device)
    elif method == "logit_normal":
        u = torch.randn(batch_size, device=device) * sigma + mu
        t_continuous = torch.sigmoid(u)
        return (t_continuous * num_steps).long().clamp(0, num_steps - 1)
    else:
        raise ValueError(f"Unknown timestep sampling: {method}")


# -- Logging helpers -----------------------------------------------------------

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
    v = tensor.detach().float()
    return {
        "mean": v.mean().item(), "std": v.std(unbiased=False).item(),
        "min": v.min().item(), "max": v.max().item(),
        "rms": v.pow(2).mean().sqrt().item(),
        "has_nan": bool(torch.isnan(v).any().item()),
        "has_inf": bool(torch.isinf(v).any().item()),
    }


def _gradient_debug_summary(named_parameters, top_k: int) -> dict:
    records = []
    total_sq = 0.0
    params_with_grad = 0
    params_without_grad = 0

    for name, param in named_parameters:
        if param.grad is None:
            params_without_grad += 1
            continue
        params_with_grad += 1
        grad = param.grad.detach().float()
        grad_norm = grad.norm().item()
        total_sq += grad.pow(2).sum().item()
        records.append({
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "max_abs_grad": grad.abs().max().item(),
            "grad_norm": grad_norm,
        })
    records.sort(key=lambda r: r["grad_norm"], reverse=True)
    return {
        "total_grad_norm": total_sq ** 0.5,
        "params_with_grad": params_with_grad,
        "params_without_grad": params_without_grad,
        "top_parameters": records[:top_k],
    }


# -- Weight deviation tracking -------------------------------------------------

def _classify_param(name: str) -> str:
    """Group Sana student transformer parameters by architectural role."""
    if "patch_embed" in name:
        return "patch_embed"
    if "time_embed" in name or "adaln_single" in name or "time_proj" in name:
        return "time_embed"
    if "caption_projection" in name or "caption_norm" in name:
        return "caption_proj"
    if "proj_out" in name or "norm_out" in name or "scale_shift_table" in name:
        return "out_proj"
    if "transformer_blocks" in name:
        if ".attn1" in name:
            return "block_self_attn"
        if ".attn2" in name:
            return "block_cross_attn"
        if ".ff" in name:
            return "block_ffn"
        if ".norm" in name:
            return "block_norm"
        return "block_other"
    return "other"


@torch.no_grad()
def snapshot_weights_cpu(named_params, dtype=torch.bfloat16):
    return {name: p.detach().to("cpu", dtype=dtype).clone()
            for name, p in named_params}


@torch.no_grad()
def weight_deviation_metrics(named_params, initial_cpu: dict) -> dict:
    group_sq_diff = {}
    group_sq_init = {}
    for name, p in named_params:
        if name not in initial_cpu:
            continue
        init = initial_cpu[name].to(p.device, dtype=torch.float32, non_blocking=True)
        diff = (p.detach().float() - init).pow(2).sum().item()
        norm = init.pow(2).sum().item()
        group = _classify_param(name)
        group_sq_diff[group] = group_sq_diff.get(group, 0.0) + diff
        group_sq_init[group] = group_sq_init.get(group, 0.0) + norm

    total_diff = sum(group_sq_diff.values())
    total_init = sum(group_sq_init.values())

    metrics = {
        "deviation/total_abs": total_diff ** 0.5,
        "deviation/total_rel": (total_diff / max(total_init, 1e-12)) ** 0.5,
    }
    for g, d in group_sq_diff.items():
        i = group_sq_init.get(g, 1e-12)
        metrics[f"deviation/{g}_abs"] = d ** 0.5
        metrics[f"deviation/{g}_rel"] = (d / max(i, 1e-12)) ** 0.5
        metrics[f"deviation/{g}_share"] = d / max(total_diff, 1e-12)
    return metrics


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = str(device) if device.type == "cuda" else "cpu"
    return torch.Generator(device=gen_device).manual_seed(seed)


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Last-layer feature-matching distillation: Sana-1.6B -> Sana-600M"
    )

    # paths
    parser.add_argument("--output_dir", default="checkpoints/distill_sana")
    parser.add_argument("--teacher_id",
                        default="Efficient-Large-Model/Sana_1600M_1024px_diffusers")
    parser.add_argument("--student_id",
                        default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--coco_path",
                        default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO")

    # training
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU batch size. Sana-1.6B + Sana-600M at 1024px "
                             "is memory-heavy; start small.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_step", type=int, default=0)

    # feature matching
    parser.add_argument("--beta_feat", type=float, default=1.0,
                        help="Feature matching loss weight (after warmup).")
    parser.add_argument("--feat_loss_type", choices=["mse", "kl"], default="mse",
                        help="mse: MSE on L2-normalised features (cosine-equivalent). "
                             "kl: KL(softmax(h_t/tau) || softmax(P(h_s)/tau)) over the "
                             "teacher feature dim at each token.")
    parser.add_argument("--feat_kl_tau", type=float, default=1.0,
                        help="Temperature for feature KL mode. tau>=1.0 recommended "
                             "since feature dim (2240) produces peaky softmaxes.")
    parser.add_argument("--warmup_feat_steps", type=int, default=0,
                        help="Linear warmup steps for feature loss weight. "
                             "0 = use --warmup_steps.")

    # timestep sampling
    parser.add_argument("--timestep_sampling", choices=["uniform", "logit_normal"],
                        default="logit_normal")
    parser.add_argument("--logit_normal_mu", type=float, default=0.0)
    parser.add_argument("--logit_normal_sigma", type=float, default=1.0)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Student learning rate.")
    parser.add_argument("--lr_projector", type=float, default=5e-4,
                        help="Projector learning rate.  Higher than student because "
                             "the projector starts random and is small (~2-3M params).")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # diffusion / resolution
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Used to set up the scheduler's timestep grid; training "
                             "samples t uniformly from this grid.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma tokenizer max seq length (Sana default: 300).")

    # memory
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offload_text_encoder", action="store_true",
                        help="Move Gemma text encoder to CPU between forward passes.")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--debug_every", type=int, default=0)
    parser.add_argument("--debug_top_k", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", default="sana-distillation")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    warmup_feat = args.warmup_feat_steps if args.warmup_feat_steps > 0 else args.warmup_steps

    # -- Distributed setup -----------------------------------------------------
    device, rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = (rank == 0)

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    if is_main:
        print(f"Device: {device} | World size: {world_size} | Distributed: {is_distributed}")

    B = args.batch_size
    effective_batch = B * world_size * args.gradient_accumulation_steps

    # -- Dataset ---------------------------------------------------------------
    if not args.coco_path or not Path(args.coco_path).exists():
        raise FileNotFoundError(
            f"--coco_path '{args.coco_path}' does not exist. "
            "A valid COCO captions dataset is required."
        )
    train_dataset = COCOCaptionDataset(args.coco_path)
    if is_main:
        print(f"Loaded {len(train_dataset):,} COCO captions")

    sampler = (DistributedSampler(train_dataset, num_replicas=world_size,
                                  rank=rank, shuffle=True)
               if is_distributed else None)
    train_dataloader = DataLoader(
        train_dataset, batch_size=B, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, drop_last=True,
    )
    if is_main:
        print(f"Effective batch size: {effective_batch} "
              f"({B} x {world_size} GPUs x {args.gradient_accumulation_steps} accum)")

    data_iter = iter(train_dataloader)
    epoch = 0

    # -- Load teacher (frozen) -------------------------------------------------
    # Sana's official recipe: transformer in fp16 (the 1.6B 1024px variant has
    # no BF16 release and produces garbage when naively cast to bf16), but the
    # Gemma-2 text encoder and DC-AE VAE must stay in bf16 -- Gemma-2's
    # attn_logit_softcapping silently overflows in fp16 and yields NaN embeds.
    # Teacher transformer is frozen, so fp16's narrow range is fine (no grads).
    if is_main:
        print(f"Loading teacher ({args.teacher_id})...")
    pipe_teacher = SanaPipeline.from_pretrained(
        args.teacher_id, torch_dtype=torch.float16,
    ).to(device)
    pipe_teacher.text_encoder.to(torch.bfloat16)
    pipe_teacher.vae.to(torch.bfloat16)
    pipe_teacher.transformer.eval()
    for p in pipe_teacher.transformer.parameters():
        p.requires_grad = False

    # -- Load student (trainable) ----------------------------------------------
    if is_main:
        print(f"Loading student ({args.student_id})...")
    pipe_student = SanaPipeline.from_pretrained(
        args.student_id, torch_dtype=torch.bfloat16,
    )
    student_transformer = pipe_student.transformer.to(device)
    student_transformer.train()

    if args.gradient_checkpointing:
        student_transformer.enable_gradient_checkpointing()
        if is_main:
            print("Gradient checkpointing enabled.")

    # Share frozen components (same VAE, same text encoder, same tokenizer, same scheduler)
    pipe_student.vae = pipe_teacher.vae
    pipe_student.text_encoder = pipe_teacher.text_encoder
    pipe_student.tokenizer = pipe_teacher.tokenizer
    pipe_student.scheduler = pipe_teacher.scheduler

    pipe_teacher.vae.eval()
    pipe_teacher.text_encoder.eval()
    for p in pipe_teacher.vae.parameters():
        p.requires_grad = False
    for p in pipe_teacher.text_encoder.parameters():
        p.requires_grad = False

    if args.offload_text_encoder:
        pipe_teacher.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        if is_main:
            print("Gemma text encoder offloaded to CPU.")

    # -- Architecture info -----------------------------------------------------
    teacher_config = pipe_teacher.transformer.config
    student_config = student_transformer.config
    teacher_dim = teacher_config.num_attention_heads * teacher_config.attention_head_dim
    student_dim = student_config.num_attention_heads * student_config.attention_head_dim
    n_teacher_blocks = len(pipe_teacher.transformer.transformer_blocks)
    n_student_blocks = len(student_transformer.transformer_blocks)

    if is_main:
        print(f"Teacher: {n_teacher_blocks} blocks, dim={teacher_dim}")
        print(f"Student: {n_student_blocks} blocks, dim={student_dim}")

    # -- Projector (student_dim -> teacher_dim, last layer only) ---------------
    projector = nn.Linear(student_dim, teacher_dim).to(device)

    if is_main:
        n_proj_params = sum(p.numel() for p in projector.parameters())
        print(f"Feature matching: single last-block pair "
              f"(teacher block {n_teacher_blocks-1} <-> student block {n_student_blocks-1})")
        print(f"Projector: Linear({student_dim} -> {teacher_dim}), "
              f"{n_proj_params:,} params ({n_proj_params/1e6:.1f}M)")

    # -- Register hooks on last block only -------------------------------------
    teacher_capture = LastBlockCapture()
    student_capture = LastBlockCapture()
    teacher_capture.register(pipe_teacher.transformer.transformer_blocks[-1])
    student_capture.register(student_transformer.transformer_blocks[-1])

    # -- Parameter bookkeeping -------------------------------------------------
    # DDP can't handle gradients flowing through hook-captured intermediate
    # activations; run the student unwrapped and manually all-reduce.
    student_module = student_transformer
    student_named_params = list(student_module.named_parameters())
    student_params = [p for _, p in student_named_params]
    projector_named_params = [(f"projector.{n}", p) for n, p in projector.named_parameters()]
    projector_params = [p for _, p in projector_named_params]

    if is_main:
        n_student = sum(p.numel() for p in student_params)
        print(f"Trainable: student={n_student:,} params")

    # -- Snapshot initial student weights (rank 0, CPU) ------------------------
    initial_student_cpu = None
    if is_main:
        print("Snapshotting initial student weights on CPU for deviation tracking...")
        initial_student_cpu = snapshot_weights_cpu(student_named_params)

    # -- EMA -------------------------------------------------------------------
    ema_student = copy.deepcopy(student_module)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False
    update_ema(ema_student, student_module, decay=0.0)

    # -- Optimizer -------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            if is_main:
                print("Using 8-bit AdamW.")
        except ImportError:
            if is_main:
                print("WARNING: bitsandbytes not found; using standard AdamW.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    param_groups = [
        {"params": student_params, "lr": args.lr},
        {"params": projector_params, "lr": args.lr_projector},
    ]

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -- Resume ----------------------------------------------------------------
    global_step = 0
    if args.resume_step > 0:
        ckpt_path = Path(args.output_dir) / f"checkpoint_step{args.resume_step}.pt"
        if is_main:
            print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        student_module.load_state_dict(ckpt["model"])
        ema_student.load_state_dict(ckpt["ema"])
        if "projector" in ckpt:
            projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["opt"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        global_step = ckpt["steps"]
        if is_main:
            print(f"Resumed at step {global_step}")

    # -- W&B -------------------------------------------------------------------
    if is_main and not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scheduler = pipe_teacher.scheduler

    # Sana uses a DC-AE VAE with 32-channel latents and 32x spatial compression.
    latent_channels = pipe_teacher.transformer.config.in_channels
    vae_compression = 2 ** (len(pipe_teacher.vae.config.encoder_block_out_channels) - 1) \
        if hasattr(pipe_teacher.vae.config, "encoder_block_out_channels") else 32
    latent_h = args.height // vae_compression
    latent_w = args.width // vae_compression

    debug_every = args.debug_every if args.debug_every > 0 else args.log_every
    debug_path = Path(args.output_dir) / "debug_train_metrics.jsonl"

    scheduler.set_timesteps(args.num_inference_steps, device=device)
    has_sigmas = hasattr(scheduler, "sigmas") and scheduler.sigmas is not None

    if is_main:
        print(f"Latent: channels={latent_channels}, shape={latent_h}x{latent_w} "
              f"(VAE compression {vae_compression}x)")
        print(f"Scheduler: {type(scheduler).__name__}, "
              f"flow-matching sigmas={'yes' if has_sigmas else 'no'}")

    # -- Training loop ---------------------------------------------------------
    pbar = tqdm(total=args.num_steps, initial=global_step,
                desc="Distill Sana", disable=not is_main)

    forward_pass = 0
    optimizer.zero_grad()

    while global_step < args.num_steps:
        student_module.train()
        projector.train()

        # -- Fetch prompts -----------------------------------------------------
        try:
            batch_prompts = next(data_iter)
        except StopIteration:
            epoch += 1
            if is_distributed:
                sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch_prompts = next(data_iter)

        if isinstance(batch_prompts, (list, tuple)):
            batch_prompts = list(batch_prompts)
        else:
            batch_prompts = [str(batch_prompts)]

        # -- Encode prompts (frozen Gemma) -------------------------------------
        if args.offload_text_encoder:
            pipe_teacher.text_encoder.to(device)

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = pipe_teacher.encode_prompt(
                prompt=batch_prompts,
                do_classifier_free_guidance=False,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=None,
            )

        if args.offload_text_encoder:
            pipe_teacher.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # -- Sample noisy latents (flow-matching interpolant) ------------------
        gen = _make_generator(device, args.seed + forward_pass + rank * 100000)
        x_0 = torch.randn(
            B, latent_channels, latent_h, latent_w,
            device=device, dtype=torch.float32, generator=gen,
        )
        noise = torch.randn_like(x_0)

        t_indices = sample_timesteps(
            B, len(scheduler.timesteps), device,
            method=args.timestep_sampling,
            mu=args.logit_normal_mu, sigma=args.logit_normal_sigma,
        )
        t_batch = scheduler.timesteps[t_indices]

        if has_sigmas:
            # sigmas often stay on CPU while timesteps are moved to `device` by set_timesteps.
            sigmas = scheduler.sigmas[t_indices.cpu()].to(
                device=device, dtype=torch.float32)
            s = sigmas.view(B, 1, 1, 1)
            z_t = s * noise + (1.0 - s) * x_0
        else:
            # Fallback: scheduler.add_noise (epsilon-style schedulers).
            z_t = scheduler.add_noise(x_0, noise, t_indices)
            sigmas = torch.zeros(B, device=device)  # placeholder for logging

        # -- Clear captures before forward passes -----------------------------
        teacher_capture.clear()
        student_capture.clear()

        # -- Teacher forward (frozen, hook captures last-block features) -------
        # Gemma-2 runs in bf16, teacher transformer in fp16 -- cast explicitly.
        tch_dtype = pipe_teacher.transformer.dtype
        with torch.no_grad():
            pipe_teacher.transformer(
                hidden_states=z_t.to(tch_dtype),
                encoder_hidden_states=prompt_embeds.to(tch_dtype),
                encoder_attention_mask=prompt_attention_mask,
                timestep=t_batch,
                return_dict=False,
            )

        # -- Student forward (trainable, hook captures last-block features) ----
        stu_dtype = student_module.dtype
        student_module(
            hidden_states=z_t.to(stu_dtype),
            encoder_hidden_states=prompt_embeds.to(stu_dtype),
            encoder_attention_mask=prompt_attention_mask,
            timestep=t_batch,
            return_dict=False,
        )

        # -- Feature matching loss (last layer only) ---------------------------
        feat_alpha = min(1.0, global_step / max(1, warmup_feat))
        current_feat_weight = args.beta_feat * feat_alpha

        L_feat, cos_mean = feature_matching_loss(
            student_capture.feature, teacher_capture.feature,
            projector,
            loss_type=args.feat_loss_type,
            tau=args.feat_kl_tau,
        )

        loss = current_feat_weight * L_feat

        # -- Backward ----------------------------------------------------------
        (loss / args.gradient_accumulation_steps).backward()
        forward_pass += 1

        # Free captured features
        teacher_capture.clear()
        student_capture.clear()

        if forward_pass % args.gradient_accumulation_steps != 0:
            continue

        # -- Sync all gradients across ranks (manual DDP replacement) ----------
        # Flatten grads into a single buffer so we issue ONE all_reduce per step
        # instead of one per parameter (hundreds of tiny NCCL calls otherwise).
        if is_distributed:
            grads = [p.grad for p in student_params + projector_params
                     if p.grad is not None]
            if grads:
                flat = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world_size)
                for g, u in zip(grads,
                                torch._utils._unflatten_dense_tensors(flat, grads)):
                    g.copy_(u)

        # -- Gradient clipping & optimizer step --------------------------------
        all_params = student_params + projector_params
        student_grad_norm = _gradient_l2_norm(student_params)
        projector_grad_norm = _gradient_l2_norm(projector_params)

        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.max_grad_norm)
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()

        student_grad_norm_post = _gradient_l2_norm(student_params)
        projector_grad_norm_post = _gradient_l2_norm(projector_params)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        update_ema(ema_student, student_module, decay=args.ema_decay)

        global_step += 1
        pbar.update(1)

        # -- Checkpoint (all ranks synchronise around save) --------------------
        if global_step % args.save_every == 0:
            if is_main:
                ckpt = {
                    "model": student_module.state_dict(),
                    "ema": ema_student.state_dict(),
                    "projector": projector.state_dict(),
                    "opt": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "steps": global_step,
                    "args": vars(args),
                }
                ckpt_path = Path(args.output_dir) / f"checkpoint_step{global_step}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
            if is_distributed:
                dist.barrier()

        # All ranks sync here so rank 0 can safely do non-collective work below.
        if is_distributed:
            dist.barrier()

        # -- Logging (rank 0 only) ---------------------------------------------
        if not is_main:
            continue

        student_param_norm = _parameter_l2_norm(student_params)
        projector_param_norm = _parameter_l2_norm(projector_params)

        metrics = {
            "train/loss": loss.item(),
            "train/L_feat": L_feat.item(),
            "train/weighted_L_feat": (current_feat_weight * L_feat).item(),
            "train/feat_alpha": feat_alpha,
            "train/feat_weight": current_feat_weight,
            "train/feat_cos_mean": cos_mean,
            "train/grad_norm_pre_clip": grad_norm,
            "train/student_grad_norm": student_grad_norm,
            "train/student_grad_norm_post": student_grad_norm_post,
            "train/projector_grad_norm": projector_grad_norm,
            "train/projector_grad_norm_post": projector_grad_norm_post,
            "train/student_param_norm": student_param_norm,
            "train/projector_param_norm": projector_param_norm,
            "train/lr_student": optimizer.param_groups[0]["lr"],
            "train/lr_projector": optimizer.param_groups[1]["lr"],
            "train/sigma_mean": sigmas.float().mean().item(),
            "train/timestep_mean": t_batch.float().mean().item(),
            "train/epoch": epoch,
        }

        if global_step % args.log_every == 0 and initial_student_cpu is not None:
            metrics.update(weight_deviation_metrics(
                student_named_params, initial_student_cpu))

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            feat=f"{L_feat.item():.4f}",
            cos=f"{cos_mean:.3f}",
            alpha=f"{feat_alpha:.2f}",
            grad=f"{grad_norm:.2f}",
        )

        if not args.no_wandb:
            wandb.log(metrics, step=global_step)

        if global_step % args.log_every == 0:
            dev_str = ""
            if "deviation/total_rel" in metrics:
                top_group = max(
                    (k[len("deviation/"):-len("_share")] for k in metrics
                     if k.endswith("_share")),
                    key=lambda g: metrics[f"deviation/{g}_share"],
                )
                dev_str = (
                    f" | dev: abs={metrics['deviation/total_abs']:.2f} "
                    f"rel={metrics['deviation/total_rel']*100:.3f}% "
                    f"top={top_group}({metrics[f'deviation/{top_group}_share']*100:.1f}%)"
                )

            print(
                f"Step {global_step}/{args.num_steps} | "
                f"loss={loss.item():.4f} | "
                f"L_feat={L_feat.item():.4f} (w={current_feat_weight:.3f}, a={feat_alpha:.2f}) | "
                f"cos={cos_mean:.3f} | "
                f"grad stu={student_grad_norm:.2f}->{student_grad_norm_post:.2f} "
                f"proj={projector_grad_norm:.2f}->{projector_grad_norm_post:.2f} | "
                f"param stu={student_param_norm:.2f} proj={projector_param_norm:.2f} | "
                f"sigma={sigmas.float().mean().item():.3f} | "
                f"lr_s={optimizer.param_groups[0]['lr']:.2e}"
                + dev_str
            )

        # -- Debug JSONL -------------------------------------------------------
        should_debug = (global_step % debug_every == 0)
        if should_debug:
            stu_grad_debug = _gradient_debug_summary(student_named_params, args.debug_top_k)
            proj_grad_debug = _gradient_debug_summary(projector_named_params, args.debug_top_k)
            debug_record = {
                "step": global_step,
                "prompt": batch_prompts[0],
                "metrics": metrics,
                "tensor_stats": {
                    "z_t": _tensor_stats(z_t),
                },
                "gradient_debug": {
                    "student": stu_grad_debug,
                    "projector": proj_grad_debug,
                },
            }
            _append_jsonl(debug_path, debug_record)

    # -- Final save (rank 0) ---------------------------------------------------
    if is_main:
        final_ckpt = {
            "model": student_module.state_dict(),
            "ema": ema_student.state_dict(),
            "projector": projector.state_dict(),
            "opt": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "steps": global_step,
            "args": vars(args),
        }
        torch.save(final_ckpt, Path(args.output_dir) / "checkpoint_final.pt")
        torch.save(student_module.state_dict(),
                   Path(args.output_dir) / "student_final.pt")
        torch.save(ema_student.state_dict(),
                   Path(args.output_dir) / "student_ema_final.pt")
        print("Done.")
        if not args.no_wandb:
            wandb.finish()

    if is_distributed:
        dist.barrier()

    # -- Cleanup ---------------------------------------------------------------
    teacher_capture.remove()
    student_capture.remove()
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
