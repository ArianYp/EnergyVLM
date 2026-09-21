#!/usr/bin/env python3
"""Consistency distillation of SD3.5 on selected teacher trajectories.

Each update takes one caption from the candidate cache, picks ONE of its N cached candidates with
the chosen selector, re-rolls that candidate's K-step guided teacher trajectory from its seed, and
trains the student to predict the teacher's clean-latent estimate from a noisier state:

    v_k    = (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k)        teacher velocity on segment k
    x0_k   = z_k - sigma_k * v_k                              teacher's clean-latent estimate
    x0_hat = z_{k-d} - sigma_{k-d} * v_theta(z_{k-d}, c)      student, d steps noisier, no guidance
    loss   = mean_k  pseudo_huber(x0_hat - sg[x0_k])

over the supervised window of k. The student runs a single conditional forward, so guidance is
absorbed and it is sampled with cfg 1.

Selectors (the only difference between arms; everything else is identical):
    random            the cached uniform draw `random_idx` (fixed per caption; naive distillation)
    dino_patch        argmax of `dino_patch_cos`: DINOv2 mean-patch cosine to the caption's photograph
    latent            argmax of `latent_cos`: the decode-free latent scorer (train/latent_scorer.py), a
                      projector from the terminal latent into DINO space scored against the photograph
    bench             argmax of `bench_score`: the official T2I-CompBench++ evaluator of the prompt's own
                      category on each decoded candidate (data/build_bench_selection.py; the GORS / CTCal
                      data protocol on benchmark TRAIN prompts, no photograph, docs/bench/)
    boltzmann         every candidate, its loss weighted by softmax(dino_patch_cos / T); exact soft
                      selection, N rollouts and N student passes per caption (--temp)
    boltzmann_sample  one candidate drawn from softmax(dino_patch_cos / T) on every visit; the
                      one-sample estimator of `boltzmann` at single-candidate cost (--temp, --sel_seed)
    boltzmann_frozen  one candidate drawn from softmax(dino_patch_cos / T) ONCE per caption by a
                      per-caption generator (--map_seed) and kept for the whole run: the soft
                      selection distribution without per-visit resampling
    boltzmann_mc      --mc_draws iid draws from softmax(dino_patch_cos / T) per visit, losses
                      weighted by count / draws, one clipped step: the same expected gradient as
                      boltzmann_sample with the candidate-sampling variance divided by the draws
    uniform_visit     one candidate drawn uniformly on every visit (the control that separates
                      target persistence from label quality; `random` draws once per caption)

Reward arms (--reward_mode, single-candidate selectors only): -lambda * r is added to the loss,
    r = mean over the --reward_states least-noisy supervised inputs of
        cos( phi(x0_hat), u_pat(photo) )              u_pat(photo) from data/build_reward_refs.py
    proj   phi = the latent projector P (train/latent_scorer.py), frozen; or refreshed every
           --reward_refresh_every updates on decoded recent predictions plus replay (report Sec. 11.6)
    rgb    phi = DINOv2 o VAE-decode, the RGB scorer itself, differentiable end to end; the exact
           reward the projector approximates (report Sec. 11.7). About 1.2x the step time.
The paper's arm (paper/, "ours") is --selector dino_patch --reward_mode proj --reward_lambda 80
--reward_refresh_every 100 --reward_refresh_steps 16: the refresh runs on rank 0 (it needs rank 0's
buffer of recent predictions) and the refreshed projector is broadcast to every other rank right
after, so the arm trains on any number of GPUs.
Every --reward_monitor_every updates the 32 most recent predictions are decoded and scored by the
OFFLINE scorer (8-bit image, HF processor) and logged next to the reward: the hacking monitor.
RNG note: loading the scorer (and constructing the projector) advances the global torch generator.
The trainer saves the generator state before loading the reward machinery and restores it after, so
a reward arm visits the captions in the same order as a reward-free run with the same --seed, and a
reward mode with --reward_lambda 0 (scorer loaded, no reward term) is bit-identical to no reward.
The first reward runs of the report (Sections 11.6 and the first exact-reward run of 11.7) predate
this and saw a different caption order; --reward_legacy_rng reproduces them.

--accum M accumulates M captions per optimizer update (batch M per GPU at the same per-update
cost); --num_steps counts optimizer updates, so the data budget is num_steps * accum * world.
--lr_schedule: after the linear warm-up the learning rate is constant (every run of the report and
of the paper's main tables) or decays with a cosine to 0 at --num_steps (the converged schedule of
the paper's Section 3.6); the multiplier is a function of the optimizer-update count.

Multi-GPU under torchrun: a DistributedSampler shows every caption once per global epoch, so
num_steps = epochs * n_captions / (world_size * accum).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.distributed import barrier, setup_distributed, teardown  # noqa: E402
from common.sampling import candidate_noise, encode_prompt, rollout, vae_decode  # noqa: E402

SELECTORS = ("random", "dino_patch", "latent", "bench", "boltzmann", "boltzmann_sample", "boltzmann_frozen",
             "boltzmann_mc", "uniform_visit")
# score field each selector ranks by; the boltzmann selectors take theirs from --score_field
FIELD = {"dino_patch": "dino_patch_cos", "latent": "latent_cos", "bench": "bench_score"}


class Records(Dataset):
    def __init__(self, records):
        self.items = records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def weights(rec: dict, selector: str, temp: float, field: str = "dino_patch_cos") -> np.ndarray:
    """Weight of each cached candidate: a one-hot for the deterministic selectors, the per-visit
    draw distribution for the sampled ones, the loss weights for `boltzmann`."""
    field = FIELD.get(selector, field)
    n = int(rec["N"]) if "N" in rec else len(rec[field])
    if selector == "random":
        w = np.zeros(n); w[int(rec["random_idx"])] = 1.0
        return w
    if selector == "uniform_visit":
        return np.full(n, 1.0 / n)
    s = np.asarray(rec[field], dtype=float)
    if selector in ("dino_patch", "latent", "bench"):
        w = np.zeros(n); w[int(s.argmax())] = 1.0
        return w
    # Boltzmann weights on the RAW score scale (no z-scoring, no floor), computed stably.
    # T -> 0 recovers dino_patch, T -> inf the uniform weighting.
    z = (s - s.max()) / max(temp, 1e-12)
    q = np.exp(z)
    return q / q.sum()


def select(rec: dict, selector: str, temp: float, rng: np.random.Generator, map_seed: int, field: str = "dino_patch_cos") -> int:
    """The candidate this visit trains on. Deterministic selectors need no draw; the sampled ones
    draw from the weights with the selection generator (private to selection, so the draws do not
    touch the data order or the noise-level stream); boltzmann_frozen draws from a generator
    seeded by (map_seed, caption idx), so its draw is the same on every visit and every rank."""
    w = weights(rec, selector, temp, field)
    if int((w > 0).sum()) == 1:
        return int(w.argmax())
    if selector == "boltzmann_frozen":
        return int(np.random.default_rng([int(map_seed), int(rec["idx"])]).choice(len(w), p=w))
    return int(rng.choice(len(w), p=w))


SCORE_FIELDS = ("dino_patch_cos", "latent_cos", "dino_cos", "clip_cos", "endpoint_vqa", "bench_score")


@torch.no_grad()
def log_samples(model, pipe, prompts, steps, cfg, height, device, gstep, tag, neg_emb, neg_pool):
    """Sample a fixed prompt set from `model` and log an image grid and a table to wandb.

    Only private generators are used (seeded by each prompt's idx, the evaluation generator's
    convention), so the RNG streams that decide the data order and the noise-level draws are not
    advanced: a run with sampling on is bit-identical to one with it off.
    """
    import wandb
    was_training = model.training
    model.eval()
    lat_c = model.config.in_channels
    h_lat = height // pipe.vae_scale_factor
    grid, rows = [], []
    for p in prompts:
        emb, pooled = encode_prompt(pipe, p["prompt"], device)
        z0 = candidate_noise(0, int(p["idx"]), (1, lat_c, h_lat, h_lat), device)
        z = rollout(model, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, steps, cfg, device)
        img = vae_decode(pipe.vae, z)
        u8 = ((img + 1) / 2).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        grid.append(wandb.Image(u8, caption=f"[{p.get('bench', '')} {p['idx']}] {p['prompt'][:70]}"))
        rows.append([gstep, int(p["idx"]), p.get("bench", ""), p.get("category", ""), p["prompt"], wandb.Image(u8)])
    if was_training:
        model.train()
    wandb.log({f"{tag}/grid": grid,
               f"{tag}/table": wandb.Table(columns=["step", "idx", "bench", "category", "prompt", "image"], data=rows)},
              step=gstep)


class PatchScorer:
    """The offline RGB DINO scorer of data/build_candidates.py, in-process and without gradients:
    decode, 8-bit image, HF processor (resize 256, centre crop 224, ImageNet statistics), DINOv2
    patch mean with the CLS token dropped, L2-normalised. The independent monitor of the reward arms."""

    def __init__(self, dino_id, device):
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        self.Image, self.device = Image, device
        self.dino = AutoModel.from_pretrained(dino_id).to(device).eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        self.proc = AutoImageProcessor.from_pretrained(dino_id)

    @torch.no_grad()
    def embed(self, images):
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        h = self.dino(pixel_values=px).last_hidden_state
        return F.normalize(h[:, 1:].float().mean(1), dim=-1)

    @torch.no_grad()
    def embed_latents(self, vae, latents):
        u8 = ((vae_decode(vae, latents) + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
        return self.embed([self.Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in u8])


class RGBReward:
    """DIFFERENTIABLE DINOv2 patch-mean cosine between the decoded latent and the reference
    photograph: the exact reward (--reward_mode rgb). Same preprocessing as the offline scorer
    (resize to 256 bicubic, centre crop 224, ImageNet statistics) but on tensors, so gradients flow
    through the frozen VAE decoder and the frozen DINOv2 into the latent. No 8-bit / PIL round trip;
    PatchScorer stays the independent monitor."""

    def __init__(self, vae, dino, device):
        self.vae, self.dino = vae, dino
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(self, lat):
        z = (lat.to(self.vae.dtype) / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        img = self.vae.decode(z, return_dict=False)[0].float()                    # [-1, 1], grad through the decoder
        x = ((img + 1) / 2).clamp(0, 1)
        x = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False, antialias=True)
        x = x[:, :, 16:240, 16:240]
        x = (x - self.mean) / self.std
        h = self.dino(pixel_values=x).last_hidden_state                          # fp32 DINO, grad through its input
        return F.normalize(h[:, 1:].float().mean(1), dim=-1)

    def score(self, lat, e_ref):
        return (self.embed(lat) * e_ref).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True, choices=SELECTORS)
    ap.add_argument("--cache_dir", required=True, help="data/build_candidates.py output")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--num_steps", type=int, default=6000, help="optimizer updates")
    ap.add_argument("--num_warmup_steps", type=int, default=300)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"],
                    help="after the linear warm-up: constant (the report and the paper's main tables) or cosine decay "
                         "to 0 at --num_steps (the paper's converged schedule); counted in optimizer updates")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1,
                    help="captions per optimizer update per GPU; the loss is divided by accum and "
                         "the clip is applied once per update (DDP all-reduces every micro-step: "
                         "exact, at extra communication)")
    ap.add_argument("--temp", type=float, default=0.04, help="T of the boltzmann selectors")
    ap.add_argument("--sel_seed", type=int, default=None,
                    help="seed of the selection draws of the sampled selectors (default: --seed)")
    ap.add_argument("--map_seed", type=int, default=1000003,
                    help="seed of the per-caption draw of boltzmann_frozen (mixed with the caption idx)")
    ap.add_argument("--mc_draws", type=int, default=4, help="draws per visit of boltzmann_mc")
    ap.add_argument("--score_field", default="dino_patch_cos",
                    help="cache field the boltzmann selectors rank by (dino_patch_cos, or latent_cos from the latent scorer)")
    # reward arms: -lambda * cos(phi(x0_hat), DINO(photo)) on the student's least-noisy clean estimates
    ap.add_argument("--reward_mode", default="none", choices=["none", "proj", "rgb"],
                    help="proj: reward through the frozen (or refreshed) latent projector; rgb: the exact reward, decode + DINOv2 with gradients")
    ap.add_argument("--reward_proj", default="checkpoints/latent_scorer/projector.pt", help="projector checkpoint (train/latent_scorer.py)")
    ap.add_argument("--reward_lambda", type=float, default=0.0, help="80 (proj) and 15.5 (rgb) put the reward gradient at 20%% of the consistency gradient")
    ap.add_argument("--reward_states", type=int, default=2, help="reward on the predictions from the K least-noisy student inputs")
    ap.add_argument("--reward_ref", default="cache/reward/ref_emb.pt", help="caption idx -> DINO patch embedding of the reference photo (data/build_reward_refs.py)")
    ap.add_argument("--reward_refresh_every", type=int, default=0, help="0 = frozen projector; else refresh it every N updates on decoded recent predictions")
    ap.add_argument("--reward_refresh_steps", type=int, default=4)
    ap.add_argument("--reward_refresh_lr", type=float, default=1e-4)
    ap.add_argument("--reward_replay", default="cache/reward/replay.pt", help="teacher-candidate latents replayed during refresh and used to verify the rgb path")
    ap.add_argument("--reward_monitor_every", type=int, default=100, help="decode + offline-score recent predictions next to the reward (hacking monitor)")
    ap.add_argument("--reward_grad_probe", type=int, default=0, help="print ||grad CD|| and ||grad reward|| separately for the first N updates (for setting lambda)")
    ap.add_argument("--reward_grad_probe_every", type=int, default=0, help="log ||grad CD||, ||grad reward||, their ratio and cosine every N updates")
    ap.add_argument("--reward_legacy_rng", action="store_true", help="do NOT restore the global RNG after loading the reward machinery (reproduces the first reward runs)")
    # --- prompt-aware ranking on the projector space (train/rank_utils.py, docs/rank/): a head g on the
    # --- 768-d DINO space trained so the student's generation of a caption outranks its generations of
    # --- structured negatives (data/build_negatives.py) against the caption's photograph
    ap.add_argument("--rank_negatives", default=None, help="data/build_negatives.py output; enables the ranking machinery")
    ap.add_argument("--rank_input", default="rollout", choices=["rollout", "xhat"],
                    help="rollout: the student's --rank_steps-step samples of the prompts from one noise (rounds 1-2); xhat: the one-step clean "
                         "estimates at the reward states, positive vs negatives from the SAME teacher state (round 3)")
    ap.add_argument("--rank_mode", default="head", choices=["head", "backprop"],
                    help="head: the ranking loss trains only g (Variant B); backprop: it also reaches the student (Variant A, weight --rank_lambda / --rank_share)")
    ap.add_argument("--rank_lambda", type=float, default=0.0)
    ap.add_argument("--rank_kappa", type=float, default=0.1)
    ap.add_argument("--rank_m", type=int, default=3, help="max negatives per caption")
    ap.add_argument("--rank_steps", type=int, default=4, help="student rollout steps at guidance 1 (rollout mode)")
    ap.add_argument("--rank_every", type=int, default=1, help="ranking term every N captions")
    ap.add_argument("--rank_head_lr", type=float, default=1e-4)
    ap.add_argument("--rank_head_width", type=int, default=1024)
    ap.add_argument("--rank_shaped_reward", action="store_true", help="anchor reward as <g(P(x0_hat)), g(u_ref)> instead of <P(x0_hat), u_ref>")
    ap.add_argument("--rank_monitor_every", type=int, default=100, help="decode recent rollouts and log the true-DINO pairwise accuracy (raw and through g)")
    ap.add_argument("--rank_head_freeze_step", type=int, default=0,
                    help="0 = head trains throughout and the transfer terms act from update 0; N > 0 = staged recipe: plain reward + head "
                         "training for N updates, then the head is frozen and the transfer terms switch on with weights calibrated once")
    ap.add_argument("--rank_shaped_match_norm", action="store_true", help="at the switch, scale the shaped reward so its gradient norm equals the plain reward's")
    ap.add_argument("--rank_share", type=float, default=0.0, help="> 0: at the switch, set lambda_rank so the ranking gradient is this share of the consistency gradient")
    ap.add_argument("--align_share", type=float, default=0.0, help="> 0: at the switch, set lambda_align likewise")
    ap.add_argument("--rank_negatives_shuffle", action="store_true", help="NULL CONTROL: every caption gets another caption's negatives (fixed rotation)")
    ap.add_argument("--rank_theta_side", default="both", choices=["both", "positive"],
                    help="backprop mode: whether the ranking gradient reaches the student through the negatives' rollouts too, or through the positive only")
    ap.add_argument("--align_lambda", type=float, default=0.0, help="REPA-style alignment of one DiT block's image tokens (rollout mode) to sg[g(P(z_K))]")
    ap.add_argument("--align_layer", type=int, default=8, help="1-based DiT block the alignment hook reads")
    ap.add_argument("--align_lr", type=float, default=1e-4)
    ap.add_argument("--align_step", type=int, default=-1, help="which rollout forward the alignment hook reads: -1 = last (sigma ~0.009), 1 = the sigma-0.86 step")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--K", type=int, default=8, help="teacher steps")
    ap.add_argument("--cfg", type=float, default=7.0, help="teacher guidance")
    ap.add_argument("--window", default="0.4,0.9", help="supervised fraction of the trajectory")
    ap.add_argument("--delta_min", type=int, default=1)
    ap.add_argument("--delta_max", type=int, default=3)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default=None, help="log to Weights & Biases if set")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--sample_every", type=int, default=0,
                    help="every N steps, sample --sample_prompts with the student and log an image "
                         "grid and a table to wandb (0 = off)")
    ap.add_argument("--sample_prompts", default=None, help="json list of {idx, prompt[, bench, category]}")
    ap.add_argument("--sample_steps", type=int, default=4)
    ap.add_argument("--sample_cfg", type=float, default=1.0)
    args = ap.parse_args()
    if args.sel_seed is None:
        args.sel_seed = args.seed

    rank, world, local_rank, device, is_main = setup_distributed(0)
    torch.manual_seed(args.seed + rank * 1009)
    random.seed(args.seed + rank * 1009)
    # Private generator for the noise-level draws: wandb.init() consumes a draw from the global
    # `random` stream, which would otherwise make logging change which states are supervised.
    delta_rng = random.Random(args.seed + rank * 1009)
    # Private generator for the selection draws of the sampled selectors.
    sel_rng = np.random.default_rng(args.sel_seed + 7919 * rank)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))
    barrier()

    recs = []
    for f in sorted(Path(args.cache_dir).glob("selection_rank*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                recs.append(json.loads(ln))
    recs.sort(key=lambda r: r["idx"])
    # The cache's scores refer to candidates rolled on the cache's own teacher grid; re-rolling them
    # here with a different --K would silently pair every score with a different image.
    try:
        _cache_K = json.loads((Path(args.cache_dir) / "cache_meta.json").read_text()).get("K")
    except Exception:
        _cache_K = None
    if _cache_K is not None:
        assert int(_cache_K) == int(args.K), f"cache {args.cache_dir} was built with K={_cache_K}, trainer --K {args.K}"
    elif is_main:
        # caches built before cache_meta.json recorded K cannot be checked: a wrong --K would silently
        # pair every cached score with a different image, so say so rather than pass quietly
        print(f"[cache] WARNING: {args.cache_dir}/cache_meta.json does not record K; cannot verify that the "
              f"cache was built on the --K {args.K} teacher grid. Rebuild it with scripts/build_candidates.lsf "
              f"(which writes K) if you are unsure.", flush=True)
    if not recs:
        raise SystemExit(f"no records under {args.cache_dir}")
    if is_main:
        print(f"[r{rank}] selector={args.selector} temp={args.temp} accum={args.accum} map_seed={args.map_seed} "
              f"mc_draws={args.mc_draws} | {len(recs)} captions", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    student = pipe.transformer
    student.to(dtype=torch.float32).train()
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    teacher = copy.deepcopy(student).to(dtype=torch.bfloat16).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_module = student_ddp = student
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        student_ddp = DDP(student, device_ids=[device.index], find_unused_parameters=False)
        student_module = student_ddp.module

    opt = torch.optim.AdamW([p for p in student_module.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    def _lr_mult(s):
        w = max(1, args.num_warmup_steps)
        if s < w:
            return s / w
        if args.lr_schedule == "cosine":
            T = max(1, args.num_steps - w)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - w) / T)))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult)
    # a fixed quarter of the parameter tensors for the ranking / alignment gradient probes and the
    # switch calibration (two full extra gradient sets do not fit next to the graph on an 80 GB GPU)
    probe_params = [p for i, p in enumerate(student_module.parameters()) if p.requires_grad and i % 4 == 0]

    def _gnorm(t):
        gs = torch.autograd.grad(t, probe_params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))

    lat_c = student_module.config.in_channels
    h_lat = args.height // pipe.vae_scale_factor
    huber_c = 0.00054 * (lat_c * h_lat * h_lat) ** 0.5
    K = args.K
    lo, hi = (float(x) for x in args.window.split(","))
    score_idxs = list(range(max(1, round(lo * K)), min(K - 1, round(hi * K)) + 1))
    n_w = len(score_idxs)
    exact_soft = args.selector in ("boltzmann", "boltzmann_mc")     # all candidates rolled out, weighted losses

    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)

    sampler = DistributedSampler(Records(recs), num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(Records(recs), batch_size=1, sampler=sampler, shuffle=(sampler is None),
                        drop_last=True, collate_fn=lambda b: b[0])
    data_iter = iter(loader)
    epoch = 0

    if is_main and args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, save_code=True,
                         config={**vars(args), "n_captions": len(recs), "world_size": world,
                                 "torch": torch.__version__, "cuda": torch.version.cuda})
        # snapshot the code: git commit, working-tree diff, and every source file as a code artifact
        try:
            import subprocess
            git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
            diff = git("diff", "HEAD", "--", "*.py", "*.lsf", "*.sh")
            run.config.update({"git_commit": git("rev-parse", "HEAD").strip(), "git_dirty": bool(diff.strip())},
                              allow_val_change=True)
            patch = Path(args.output_dir) / "git_diff.patch"
            patch.write_text(diff)
            wandb.save(str(patch), base_path=str(Path(args.output_dir)), policy="now")
            # explicit source dirs rather than log_code(root=ROOT): log_code walks everything under
            # root (out/, checkpoints/, third_party/ ...) before filtering
            art = wandb.Artifact(f"source-{run.id}", type="source")     # 'code' is reserved by wandb
            for pat in ("common/*.py", "data/*.py", "train/*.py", "eval/*.py", "eval/compat/*.py",
                        "scripts/*", "README.md", "requirements.txt"):
                for f in sorted(ROOT.glob(pat)):
                    if f.is_file():
                        art.add_file(str(f), name=str(f.relative_to(ROOT)))
            art.add_file(str(patch), name="git_diff.patch")
            run.log_artifact(art)
        except Exception as e:
            print(f"[wandb] code snapshot skipped: {e}", flush=True)
    if is_main:
        # caption -> candidate map of the deterministic selectors; for the sampled ones the hash
        # is of the weights each caption is drawn from (the draw itself changes per visit)
        h = hashlib.sha256()
        for r in recs:
            w = weights(r, args.selector, args.temp, args.score_field)
            if int((w > 0).sum()) == 1 or args.selector == "boltzmann_frozen":
                key = select(r, args.selector, args.temp, np.random.default_rng(0), args.map_seed, args.score_field)
            else:
                key = np.round(w, 6).tolist()
            h.update(f"{int(r['idx'])}:{key}\n".encode())
        print(f"[selection] window={args.window} K={K} supervised_k={score_idxs} "
              f"caption->candidate sha256={h.hexdigest()[:16]}", flush=True)

    reward_load = args.reward_mode != "none"                 # machinery loaded (also at lambda 0: the zero-weight control)
    reward_on = reward_load and args.reward_lambda != 0.0
    loss_cd_v = None
    proj = rgb_reward = scorer = None
    _rng_cpu = torch.random.get_rng_state(); _rng_cuda = torch.cuda.get_rng_state_all()
    if reward_load:
        assert not exact_soft, "the reward arms are single-candidate arms"
        # The refresh optimizes `proj` on rank 0 only (it needs rank 0's buffer of recent
        # predictions); every other rank's copy is re-synchronised by a broadcast right after, so
        # the refreshed arm trains on any number of GPUs.
        assert args.reward_mode == "proj" or args.reward_refresh_every == 0, "refresh applies to the projector only"
        ref_emb = torch.load(args.reward_ref, map_location="cpu", weights_only=False)
        if args.reward_mode == "rgb" or args.reward_refresh_every > 0 or args.reward_monitor_every > 0:
            scorer = PatchScorer(args.dino_id, device)
        if args.reward_mode == "proj":
            from train.latent_scorer import LatentProjector
            proj = LatentProjector().to(device)
            proj.load_state_dict(torch.load(args.reward_proj, map_location=device, weights_only=False)["model"]); proj.eval()
            for _p in proj.parameters():
                _p.requires_grad_(False)
        else:
            rgb_reward = RGBReward(pipe.vae, scorer.dino, device)
            pipe.vae.enable_gradient_checkpointing()                     # decoder activations at 512^2
            # verify the differentiable path against the offline (8-bit, PIL) scorer on teacher latents
            _rp = torch.load(args.reward_replay, map_location="cpu", weights_only=False)
            with torch.no_grad():
                _zc = _rp["z"][:4].to(device).float().flatten(0, 1); _er = _rp["e_ref"][:4].to(device).float().repeat_interleave(4, 0)
                _d = rgb_reward.score(_zc, _er)
                _pil = torch.cat([(scorer.embed_latents(pipe.vae, _zc[i:i + 1]) * _er[i:i + 1]).sum(-1) for i in range(len(_zc))])
                _cached = _rp["cos"][:4].to(device).flatten()
            if is_main:
                print(f"[reward-rgb] differentiable vs PIL scorer on 16 teacher latents: max|diff| {(_d - _pil).abs().max():.4f} "
                      f"mean|diff| {(_d - _pil).abs().mean():.5f} corr {float(np.corrcoef(_d.cpu(), _pil.cpu())[0, 1]):.4f}; "
                      f"PIL vs cache max|diff| {(_pil - _cached).abs().max():.4f}", flush=True)
            del _rp
        replay = torch.load(args.reward_replay, map_location="cpu", weights_only=False) if args.reward_refresh_every > 0 else None
        proj_opt = torch.optim.AdamW(proj.parameters(), lr=args.reward_refresh_lr, weight_decay=0.01) if args.reward_refresh_every > 0 else None
        rbuf = []
        reward_stats = {"r": 0.0, "n": 0, "rgb": float("nan"), "proj": float("nan"), "corr": float("nan"), "refresh_loss": float("nan"),
                        "g_cd": float("nan"), "g_r": float("nan"), "g_cos": float("nan")}
        if is_main:
            print(f"[reward] mode={args.reward_mode} projector {args.reward_proj if proj is not None else None} lambda={args.reward_lambda} "
                  f"states={args.reward_states} refresh_every={args.reward_refresh_every} monitor_every={args.reward_monitor_every} refs={len(ref_emb)}", flush=True)
        if not args.reward_legacy_rng:
            torch.random.set_rng_state(_rng_cpu); torch.cuda.set_rng_state_all(_rng_cuda)
            if is_main:
                print("[reward] global RNG state restored after loading the reward machinery: caption order matches the reward-free run", flush=True)
    # PROMPT-AWARE RANKING (docs/rank/): head g on the projector's DINO space, trained so the student's
    # generation of the caption outranks its generations of structured negatives against the photo
    rank_head = align = None
    if args.rank_negatives is not None:
        assert reward_load and proj is not None, "the ranking arms sit on the projector reward arm (--reward_mode proj)"
        assert scorer is not None or args.rank_monitor_every == 0, "the RGB monitor of the ranking needs the scorer (reward refresh or monitor on)"
        assert args.rank_input == "rollout" or (args.align_lambda == 0 and args.align_share == 0), "the alignment term needs rollouts"
        assert args.rank_input == "rollout" or reward_on, "--rank_input xhat ranks the reward states: the reward must be on"
        from train.rank_utils import (RankHead, head_apply, rollout_schedule, student_rollout, rank_rows, rows_loss, rows_acc,
                                      rows_margin, AlignHook, all_reduce_grads, broadcast_params, rollout_noise)
        _rng_cpu2 = torch.random.get_rng_state(); _rng_cuda2 = torch.cuda.get_rng_state_all()
        rank_negs = {int(k): [n["prompt"] for n in v["negatives"]][:args.rank_m]
                     for k, v in json.load(open(args.rank_negatives))["negatives"].items()}
        if args.rank_negatives_shuffle:
            _ks = sorted(rank_negs); rank_negs = {k: rank_negs[_ks[(i + 1) % len(_ks)]] for i, k in enumerate(_ks)}
        rank_frozen = False
        rank_calibrated = args.rank_head_freeze_step == 0
        rank_scale = {"shaped": 1.0, "rank": args.rank_lambda, "align": args.align_lambda}
        _cal_n_cd = float("nan")
        torch.manual_seed(args.seed + 424242)                     # identical head init on every rank
        rank_head = RankHead(768, args.rank_head_width).to(device)
        broadcast_params(rank_head, world)
        rank_head_opt = torch.optim.AdamW(rank_head.parameters(), lr=args.rank_head_lr, weight_decay=0.01)
        rank_sig, rank_ts = rollout_schedule(pipe.scheduler, args.rank_steps, K, device)
        if args.align_lambda > 0 or args.align_share > 0:
            torch.manual_seed(args.seed + 434343)
            align = AlignHook(student_module, args.align_layer,
                              in_dim=student_module.config.num_attention_heads * student_module.config.attention_head_dim, device=device)
            broadcast_params(align.proj, world)
            align_opt = torch.optim.AdamW(align.proj.parameters(), lr=args.align_lr, weight_decay=0.01)
        torch.random.set_rng_state(_rng_cpu2); torch.cuda.set_rng_state_all(_rng_cuda2)
        rank_stats = {"loss": 0.0, "n": 0, "acc": 0.0, "acc_raw": 0.0, "margin": 0.0, "margin_raw": 0.0, "r_pos": 0.0, "nneg": 0.0,
                      "n_cap": 0, "align": 0.0, "n_align": 0, "acc_rgb": float("nan"), "acc_rgb_shaped": float("nan"),
                      "g_rank": float("nan"), "g_align": float("nan"), "t": 0.0}
        rank_buf = []
        if is_main:
            print(f"[rank] negatives for {len(rank_negs)} captions (max {args.rank_m}){' SHUFFLED (null control)' if args.rank_negatives_shuffle else ''} | "
                  f"input={args.rank_input} mode={args.rank_mode} lambda={args.rank_lambda} share={args.rank_share} theta_side={args.rank_theta_side} "
                  f"kappa={args.rank_kappa} shaped_reward={args.rank_shaped_reward} match_norm={args.rank_shaped_match_norm} "
                  f"freeze_step={args.rank_head_freeze_step} align_lambda={args.align_lambda} align_share={args.align_share} "
                  f"layer={args.align_layer} align_step={args.align_step} | rollout sigmas {[round(float(s), 3) for s in rank_sig]}", flush=True)

    sample_prompts = None
    if is_main and args.wandb_project and args.sample_prompts and args.sample_every > 0:
        sample_prompts = json.loads(Path(args.sample_prompts).read_text())
    gain: dict[str, list] = {}         # score field -> [sum(selected - mean over candidates), count]
    # selection-distribution diagnostics: effective sample size and normalised entropy of the
    # weights (1 / 0 for the deterministic selectors), and how often a revisit changes the target
    sel_stats = {"n": 0, "ess": 0.0, "entropy": 0.0, "revisits": 0, "churn": 0}
    last_sel: dict[int, int] = {}
    pbar = tqdm(total=args.num_steps, disable=not is_main, desc=args.selector)
    gstep = 0
    micro = 0                          # captions consumed on this rank (for --accum)
    t_last = time.time()
    # logging window: the logged loss, per-state losses and consistency loss are means over every
    # caption since the last log (with --accum > 1 the last caption alone is not the update's loss)
    _wl_sum, _wl_n, _wk_sum, _wcd_sum, _wcd_n = 0.0, 0, None, 0.0, 0
    _wandb_fail = 0
    while gstep < args.num_steps:
        try:
            rec = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            rec = next(data_iter)
        w = weights(rec, args.selector, args.temp, args.score_field)
        if args.selector == "boltzmann":
            sel = int(w.argmax())                       # diagnostics only; every candidate is trained on
        else:
            sel = select(rec, args.selector, args.temp, sel_rng, args.map_seed, args.score_field)
        if args.selector == "boltzmann_mc":
            # M iid draws from the weights this visit; the weights become count / M (the scalar
            # draw above is consumed first, matching the experimental trainer's draw order)
            counts = np.bincount(sel_rng.choice(len(w), size=args.mc_draws, p=w), minlength=len(w))
            w = counts / float(args.mc_draws)
        if is_main:
            for f in SCORE_FIELDS:                     # what this arm's choice buys under each scorer
                if f in rec:
                    s = np.asarray(rec[f], dtype=float)
                    if np.all(np.isfinite(s)):
                        g = gain.setdefault(f, [0.0, 0])
                        g[0] += float(s[sel] - s.mean()) if not exact_soft else float((w * s).sum() - s.mean())
                        g[1] += 1
            p = w[w > 0]
            ent = float(-(p * np.log(p)).sum())
            sel_stats["n"] += 1; sel_stats["ess"] += float(np.exp(ent)); sel_stats["entropy"] += ent / np.log(len(w))
            prev = last_sel.get(int(rec["idx"]))
            if prev is not None:
                sel_stats["revisits"] += 1; sel_stats["churn"] += int(prev != sel)
            last_sel[int(rec["idx"])] = sel

        # re-roll the teacher trajectory (of every candidate for the exact soft selector) from seed
        with torch.no_grad():
            emb, pooled = encode_prompt(pipe, rec["prompt"], device)
            if exact_soft:
                z0 = torch.cat([candidate_noise(rec["seed_base"], j, (1, lat_c, h_lat, h_lat), device)
                                for j in range(len(w))], 0)
            else:
                z0 = candidate_noise(rec["seed_base"], sel, (1, lat_c, h_lat, h_lat), device)
            states, sigmas = rollout(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool,
                                     K, args.cfg, device, keep_states=True)

        if micro % args.accum == 0:
            opt.zero_grad(set_to_none=True)
        deltas = {k: delta_rng.randint(args.delta_min, args.delta_max) for k in score_idxs}
        stu_idx = [k - deltas[k] for k in score_idxs]
        t_in = torch.cat([pipe.scheduler.timesteps.to(device)[s].reshape(1) for s in stu_idx], 0)
        sig_stu = torch.stack([sigmas[s] for s in stu_idx]).view(-1, 1, 1, 1)

        def candidate_loss(z):
            """pseudo-Huber x0 consistency loss of one trajectory z[0..K]: (per-state, mean)."""
            z_in = torch.cat([z[s] for s in stu_idx], 0).to(torch.bfloat16)
            with torch.autocast("cuda", torch.bfloat16):
                v_stu = student_ddp(hidden_states=z_in, timestep=t_in,
                                    encoder_hidden_states=emb.repeat(n_w, 1, 1),
                                    pooled_projections=pooled.repeat(n_w, 1), return_dict=False)[0]
            x_hat = torch.cat([z[s] for s in stu_idx], 0) - sig_stu * v_stu.float()
            x_tea = []
            for k in score_idxs:
                v_k = (z[k + 1] - z[k]) / (sigmas[k + 1] - sigmas[k])
                x_tea.append(z[k] - sigmas[k] * v_k)
            x_tea = torch.cat(x_tea, 0).detach()
            sq = (x_hat - x_tea).pow(2).sum(dim=(1, 2, 3))
            d = torch.sqrt(sq + huber_c * huber_c) - huber_c      # one loss per supervised state
            return d, d.mean(), x_hat

        if exact_soft:
            # every candidate's loss, weighted, back-propagated one at a time (peak memory equals
            # the single-candidate selectors'); the gradient is sum_j w_j dL_j / accum
            loss_v, per_k = 0.0, torch.zeros(n_w, device=device)
            for j in range(len(w)):
                wj = float(w[j])
                if wj <= 0.0:
                    continue
                d, lj, _ = candidate_loss([s[j:j + 1].float() for s in states])
                (wj * lj / args.accum).backward()
                loss_v += wj * float(lj.detach()); per_k += wj * d.detach()
            loss = torch.tensor(loss_v, device=device)          # logging only
        else:
            per_k, loss, x_hat = candidate_loss([s.float() for s in states])
            if reward_on:
                # REWARD TERM on the clean estimates from the --reward_states least-noisy student
                # inputs (the others are too blurry to score), through the frozen map phi
                _order = sorted(range(n_w), key=lambda i: float(sig_stu[i]))[:args.reward_states]
                _xr = x_hat[_order]
                _eref = ref_emb[int(rec["idx"])].to(device).float()
                if args.reward_mode == "rgb":
                    _r = rgb_reward.score(_xr, _eref[None].expand(len(_xr), -1))   # exact: decode + DINO, with gradients
                else:
                    with torch.autocast("cuda", torch.bfloat16):
                        _ehat = proj(_xr).float()
                    _r_raw = (_ehat * _eref[None]).sum(-1)
                    _use_shaped = rank_head is not None and args.rank_shaped_reward and (args.rank_head_freeze_step == 0 or rank_frozen)
                    if _use_shaped:
                        # the reward in the ranking-shaped space, both sides through g with its weights
                        # detached (the reward trains the student, never the head)
                        _r_sh = (head_apply(rank_head, _ehat, False) * head_apply(rank_head, _eref[None], False)).sum(-1)
                        if not rank_calibrated and is_main:
                            _cal_n_cd = _gnorm(loss)
                            if args.rank_shaped_match_norm:
                                _n_raw = _gnorm(-args.reward_lambda * _r_raw.mean()); _n_sh = _gnorm(-args.reward_lambda * _r_sh.mean())
                                rank_scale["shaped"] = _n_raw / max(_n_sh, 1e-9)
                        _r = rank_scale["shaped"] * _r_sh
                    else:
                        if rank_head is not None and not rank_calibrated and is_main and math.isnan(_cal_n_cd):
                            _cal_n_cd = _gnorm(loss)
                        _r = _r_raw
                _lr = -args.reward_lambda * _r.mean()
                if is_main and (gstep < args.reward_grad_probe or (args.reward_grad_probe_every and gstep % args.reward_grad_probe_every == 0)):
                    _ps = [p for p in student_module.parameters() if p.requires_grad]
                    _g1 = torch.autograd.grad(loss, _ps, retain_graph=True, allow_unused=True)
                    _g2 = torch.autograd.grad(_lr, _ps, retain_graph=True, allow_unused=True)
                    _n1 = float(torch.sqrt(sum((g.float() ** 2).sum() for g in _g1 if g is not None)))
                    _n2 = float(torch.sqrt(sum((g.float() ** 2).sum() for g in _g2 if g is not None)))
                    _dot = float(sum((a.float() * b.float()).sum() for a, b in zip(_g1, _g2) if a is not None and b is not None))
                    reward_stats.update({"g_cd": _n1, "g_r": _n2, "g_cos": _dot / max(_n1 * _n2, 1e-9)})
                    print(f"[reward-probe] step {gstep} ||grad CD||={_n1:.2f} ||grad reward||={_n2:.4f} ratio={_n2 / max(_n1, 1e-9):.4f} "
                          f"cos(CD,reward)={_dot / max(_n1 * _n2, 1e-9):.4f} r={float(_r.mean()):.4f} cd={float(loss.detach()):.4f}", flush=True)
                    del _g1, _g2
                loss_cd_v = float(loss.detach())
                loss = loss + _lr
                reward_stats["r"] += float(_r.mean()); reward_stats["n"] += 1
                rbuf.append((_xr.detach()[:1].clone(), int(rec["idx"])))
                if len(rbuf) > 64:
                    rbuf.pop(0)
            if rank_head is not None and micro % args.rank_every == 0:
                # PROMPT-AWARE RANKING TERM (see train/rank_utils.py and docs/rank/README.md)
                _t0 = time.time()
                _negs = rank_negs.get(int(rec["idx"]), [])
                _prompts = [rec["prompt"]] + _negs
                with torch.no_grad():
                    _emb_all, _, _pool_all, _ = pipe.encode_prompt(prompt=_prompts, prompt_2=_prompts, prompt_3=_prompts,
                                                                 do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
                _transfer_on = args.rank_head_freeze_step == 0 or rank_frozen
                _grad_all = args.rank_mode == "backprop" and (args.rank_lambda > 0 or args.rank_share > 0) and _transfer_on
                _align_on = align is not None and _transfer_on and args.rank_input == "rollout"
                _uref = ref_emb[int(rec["idx"])].to(device).float()[None]
                if args.rank_input == "rollout":
                    _z0 = rollout_noise(args.seed, rank, micro, (1, lat_c, h_lat, h_lat), device).expand(len(_prompts), -1, -1, -1).contiguous()
                    # every rank rolls out on every ranking micro-step (DDP needs the same number of forwards with grad on every rank)
                    _zK = student_rollout(student_ddp, _z0, _emb_all, _pool_all, rank_sig, rank_ts, grad=_grad_all,
                                          grad_last=_align_on, hook=(align if _align_on else None), hook_step=args.align_step)
                    with torch.autocast("cuda", torch.bfloat16):
                        _e = proj(_zK).float()
                    _e_pos, _e_negs = _e[:1], (_e[1:, None, :] if len(_negs) else None)
                    _bufz = _zK.detach()
                else:
                    # RANK ON THE REWARD'S INPUTS: the positive's clean estimates at the reward states are
                    # `_xr` / `_ehat` (grad -> student through the consistency forward); the negatives'
                    # estimates come from the same states with only the text swapped, one no-grad forward
                    # through the bare module (a DDP forward here would deadlock ranks without negatives)
                    _S = len(_order)
                    _z_all = torch.cat([states[s].float() for s in stu_idx], 0)          # the student inputs of every supervised state
                    if len(_negs):
                        _emb_n, _pool_n = _emb_all[1:], _pool_all[1:]
                        _z_in = _z_all[_order].to(torch.bfloat16).repeat(len(_negs), 1, 1, 1)
                        _t_n = t_in[_order].repeat(len(_negs))
                        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                            _v_n = student_module(hidden_states=_z_in, timestep=_t_n,
                                                  encoder_hidden_states=_emb_n.repeat_interleave(_S, 0),
                                                  pooled_projections=_pool_n.repeat_interleave(_S, 0), return_dict=False)[0]
                        _x_neg = (_z_all[_order].repeat(len(_negs), 1, 1, 1) - sig_stu[_order].repeat(len(_negs), 1, 1, 1) * _v_n.float())
                        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                            _e_negs = proj(_x_neg).float().view(len(_negs), _S, -1)
                        _bufz = torch.cat([_xr[:1].detach(), _x_neg.view(len(_negs), _S, *_x_neg.shape[1:])[:, 0]], 0)
                    else:
                        _e_negs = None; _bufz = _xr[:1].detach()
                    _e_pos = _ehat
                    _e = torch.cat([_e_pos] + ([_e_negs.transpose(0, 1).reshape(-1, _e_pos.shape[-1])] if _e_negs is not None else []), 0)
                _R_raw = rank_rows(None, _e_pos.detach(), (_e_negs.detach() if _e_negs is not None else None), _uref, False)
                _Rh = rank_rows(rank_head, _e_pos.detach(), (_e_negs.detach() if _e_negs is not None else None), _uref, True)
                _l_head = rows_loss(_Rh, args.rank_kappa)
                if not rank_frozen:
                    (_l_head / args.accum).backward()
                _cal_now = rank_frozen and not rank_calibrated and is_main and len(_negs) > 0
                if _grad_all:
                    _negs_th = None if _e_negs is None else (_e_negs if (args.rank_theta_side == "both" and args.rank_input == "rollout") else _e_negs.detach())
                    _Rs = rank_rows(rank_head, _e_pos, _negs_th, _uref, False)
                    _l_rank = rows_loss(_Rs, args.rank_kappa)
                    if _cal_now and args.rank_share > 0:
                        rank_scale["rank"] = args.rank_share * _cal_n_cd / max(_gnorm(_l_rank), 1e-9)
                    if is_main and args.reward_grad_probe_every and gstep % args.reward_grad_probe_every == 0 and len(_negs):
                        rank_stats["g_rank"] = _gnorm(_l_rank)
                        print(f"[rank-probe] step {gstep} ||grad rank||={rank_stats['g_rank']:.4f} (x lambda {rank_scale['rank']:.3f}) "
                              f"||grad CD||={reward_stats['g_cd']:.2f} ratio={rank_scale['rank'] * rank_stats['g_rank'] / max(reward_stats['g_cd'], 1e-9):.4f} "
                              f"rank_loss={float(_l_rank):.4f} acc={rows_acc(_Rs.detach()):.2f}", flush=True)
                    loss = loss + rank_scale["rank"] * _l_rank
                if _align_on:
                    _f = align.features()
                    _tgt = head_apply(rank_head, _e.detach(), False).detach()
                    _l_align = (1.0 - (_f * _tgt).sum(-1)).mean()
                    if _cal_now and args.align_share > 0:
                        rank_scale["align"] = args.align_share * _cal_n_cd / max(_gnorm(_l_align), 1e-9)
                    if is_main and args.reward_grad_probe_every and gstep % args.reward_grad_probe_every == 0:
                        rank_stats["g_align"] = _gnorm(_l_align)
                        print(f"[align-probe] step {gstep} ||grad align||={rank_stats['g_align']:.4f} (x lambda {rank_scale['align']:.3f}) "
                              f"||grad CD||={reward_stats['g_cd']:.2f} ratio={rank_scale['align'] * rank_stats['g_align'] / max(reward_stats['g_cd'], 1e-9):.4f} "
                              f"align_loss={float(_l_align):.4f}", flush=True)
                    loss = loss + rank_scale["align"] * _l_align
                    rank_stats["align"] += float(_l_align); rank_stats["n_align"] += 1
                if rank_frozen and not rank_calibrated:
                    # every rank takes rank 0's calibration (a collective on every rank, every micro-step
                    # until rank 0 has seen a caption with negatives)
                    _t = torch.tensor([1.0 if _cal_now else 0.0, rank_scale["shaped"], rank_scale["rank"], rank_scale["align"]], device=device)
                    if world > 1:
                        import torch.distributed as dist
                        dist.broadcast(_t, src=0)
                    if float(_t[0]) > 0:
                        rank_scale = {"shaped": float(_t[1]), "rank": float(_t[2]), "align": float(_t[3])}; rank_calibrated = True
                        if is_main:
                            print(f"[rank] switch calibrated at update {gstep}: shaped_scale={rank_scale['shaped']:.3f} lambda_rank={rank_scale['rank']:.3f} "
                                  f"lambda_align={rank_scale['align']:.3f} (||grad CD|| {_cal_n_cd:.2f} on the probe subset)", flush=True)
                if len(_negs):
                    rank_stats["loss"] += float(_l_head); rank_stats["n"] += 1
                    rank_stats["acc"] += rows_acc(_Rh.detach()); rank_stats["acc_raw"] += rows_acc(_R_raw)
                    rank_stats["margin"] += rows_margin(_Rh.detach()); rank_stats["margin_raw"] += rows_margin(_R_raw)
                    if is_main:
                        rank_buf.append((_bufz.to(torch.bfloat16), int(rec["idx"])))
                        if len(rank_buf) > 16:
                            rank_buf.pop(0)
                rank_stats["r_pos"] += float(_Rh[:, 0].mean()); rank_stats["nneg"] += len(_negs); rank_stats["n_cap"] += 1
                rank_stats["t"] += time.time() - _t0
                del _emb_all, _pool_all, _e, _e_pos, _e_negs, _bufz
            (loss / args.accum).backward()
        _wl_sum += float(loss.detach()); _wl_n += 1
        _wk_sum = per_k.detach().float().clone() if _wk_sum is None else _wk_sum + per_k.detach().float()
        if loss_cd_v is not None:
            _wcd_sum += loss_cd_v; _wcd_n += 1
        micro += 1
        if micro % args.accum != 0:
            continue                      # accumulate: no clip, no step, no logging until the window closes
        gn = torch.nn.utils.clip_grad_norm_(student_module.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        gstep += 1
        pbar.update(1)
        if rank_head is not None:
            # the head (and the alignment projector) step with the student: gradients averaged over the
            # ranks so every rank keeps an identical copy; frozen after --rank_head_freeze_step
            if not rank_frozen:
                all_reduce_grads(rank_head, world); torch.nn.utils.clip_grad_norm_(rank_head.parameters(), 1.0)
                rank_head_opt.step(); rank_head_opt.zero_grad(set_to_none=True)
            if align is not None and (args.rank_head_freeze_step == 0 or rank_frozen):
                all_reduce_grads(align.proj, world); torch.nn.utils.clip_grad_norm_(align.proj.parameters(), 1.0)
                align_opt.step(); align_opt.zero_grad(set_to_none=True)
            if args.rank_head_freeze_step and gstep == args.rank_head_freeze_step and not rank_frozen:
                rank_frozen = True
                for _p in rank_head.parameters():
                    _p.requires_grad_(False)
                if is_main:
                    print(f"[rank] head frozen at update {gstep} (staged recipe); the transfer terms switch on and are calibrated on the next captions", flush=True)
            if is_main and rank_buf and args.rank_monitor_every and gstep % args.rank_monitor_every == 0:
                # TRUE-DINO CHECK: decode recent rollouts and rank them with the offline scorer, raw and through g
                with torch.no_grad():
                    _a, _ah, _nb = 0.0, 0.0, 0
                    for _zk, _i in rank_buf[-8:]:
                        _et = scorer.embed_latents(pipe.vae, _zk)
                        _ur = ref_emb[_i].to(device).float()[None]
                        _a += rows_acc(rank_rows(None, _et[:1], _et[1:, None, :] if len(_et) > 1 else None, _ur, False))
                        _ah += rows_acc(rank_rows(rank_head, _et[:1], _et[1:, None, :] if len(_et) > 1 else None, _ur, False)); _nb += 1
                    rank_stats["acc_rgb"] = _a / max(_nb, 1); rank_stats["acc_rgb_shaped"] = _ah / max(_nb, 1)
            if is_main and gstep % args.log_every == 0 and rank_stats["n"]:
                _n = rank_stats["n"]
                print(f"[rank] step {gstep} loss={rank_stats['loss'] / _n:.4f} acc_shaped={rank_stats['acc'] / _n:.3f} acc_raw={rank_stats['acc_raw'] / _n:.3f} "
                      f"margin_shaped={rank_stats['margin'] / _n:+.4f} margin_raw={rank_stats['margin_raw'] / _n:+.4f} "
                      f"acc_rgb={rank_stats['acc_rgb']:.3f} acc_rgb_shaped={rank_stats['acc_rgb_shaped']:.3f} "
                      f"negs/caption={rank_stats['nneg'] / max(rank_stats['n_cap'], 1):.2f} "
                      f"align={rank_stats['align'] / max(rank_stats['n_align'], 1):.4f} t_rank/caption={rank_stats['t'] / max(rank_stats['n_cap'], 1):.2f}s "
                      f"mem={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB", flush=True)
        if reward_on and is_main and rbuf and ((args.reward_refresh_every and gstep % args.reward_refresh_every == 0)
                                                or (args.reward_monitor_every and gstep % args.reward_monitor_every == 0)):
            # HACKING MONITOR: decode the recent predictions and score them with the offline RGB
            # scorer against the same references; log both scores and their correlation
            _xs = torch.cat([x for x, _ in rbuf[-32:]], 0); _ids = [i for _, i in rbuf[-32:]]
            with torch.no_grad():
                _et = scorer.embed_latents(pipe.vae, _xs)
                _er = torch.stack([ref_emb[i] for i in _ids]).to(device).float()
                _rgb = (_et * _er).sum(-1)
                if args.reward_mode == "rgb":
                    _pr = rgb_reward.score(_xs.float(), _er)                 # the differentiable path on the same latents
                else:
                    with torch.autocast("cuda", torch.bfloat16):
                        _pr = (proj(_xs.float()).float() * _er).sum(-1)
            reward_stats["rgb"] = float(_rgb.mean()); reward_stats["proj"] = float(_pr.mean())
            reward_stats["corr"] = float(np.corrcoef(_rgb.cpu().numpy(), _pr.cpu().numpy())[0, 1]) if len(_ids) > 2 else float("nan")
            if args.reward_refresh_every and gstep % args.reward_refresh_every == 0:
                # REFRESH: fit the projector to the RGB scorer on the student's own predictions, with
                # replay of teacher candidates (cosine + ranking KL) so it keeps ranking those too
                for _p in proj.parameters():
                    _p.requires_grad_(True)
                proj.train(); _tot = 0.0
                _rr = np.random.default_rng(gstep)
                for _ in range(args.reward_refresh_steps):
                    with torch.autocast("cuda", torch.bfloat16):
                        _es = proj(_xs.float()).float()
                    _ls = (1 - (_es * _et).sum(-1)).mean()
                    _rb = _rr.choice(replay["z"].shape[0], size=8, replace=False)
                    _zr = replay["z"][_rb].to(device).float().flatten(0, 1); _ec = replay["e_cand"][_rb].to(device).float().flatten(0, 1)
                    _rref = replay["e_ref"][_rb].to(device).float(); _cr = replay["cos"][_rb].to(device)
                    with torch.autocast("cuda", torch.bfloat16):
                        _ep = proj(_zr).float()
                    _lrp = (1 - (_ep * _ec).sum(-1)).mean() + F.kl_div(F.log_softmax((_ep.view(len(_rb), 4, -1) * _rref[:, None]).sum(-1) / 0.04, -1),
                                                                      F.softmax(_cr / 0.04, -1), reduction="batchmean")
                    _l = _ls + _lrp
                    proj_opt.zero_grad(); _l.backward(); torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0); proj_opt.step(); _tot += float(_l)
                proj.eval()
                for _p in proj.parameters():
                    _p.requires_grad_(False)
                reward_stats["refresh_loss"] = _tot / args.reward_refresh_steps
        if (reward_on and proj is not None and world > 1
                and args.reward_refresh_every and gstep % args.reward_refresh_every == 0):
            # The refresh above ran on rank 0 only; every other rank's copy of `proj` is stale until
            # this runs. The condition is rank-agnostic (gstep and the flags are identical on every
            # rank), so every rank reaches this collective together.
            import torch.distributed as dist
            for _p in proj.parameters():
                dist.broadcast(_p.data, src=0)

        if is_main and gstep % args.log_every == 0:
            lv, g = _wl_sum / max(_wl_n, 1), float(gn)
            pbar.set_postfix({"loss": f"{lv:.3g}", "g": f"{g:.2g}"})
            if args.wandb_project:
                import wandb
                now = time.time()
                n = max(sel_stats["n"], 1)
                with torch.no_grad():
                    # how far the student has moved from its initialisation (the teacher's weights):
                    # a random walk on a plateau keeps growing, a converging run flattens
                    _d2 = sum(float((_p.detach().float() - _q.detach().float()).pow(2).sum())
                              for _p, _q in zip(student_module.parameters(), teacher.parameters()))
                payload = {"train/loss": lv, "train/grad_norm": g, "train/lr": sched.get_last_lr()[0],
                           "train/epoch": epoch, "train/samples_seen": micro * world,
                           "train/steps_per_s": args.log_every / max(now - t_last, 1e-6),
                           "train/gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                           "train/dist_from_teacher": _d2 ** 0.5,
                           "train/clip_coef": min(1.0, args.grad_clip / max(g, 1e-12)),
                           "sel/weight_ess": sel_stats["ess"] / n, "sel/weight_entropy": sel_stats["entropy"] / n,
                           "sel/target_churn": (sel_stats["churn"] / sel_stats["revisits"]) if sel_stats["revisits"] else float("nan")}
                t_last = now
                for k, v in zip(score_idxs, (_wk_sum / max(_wl_n, 1)).tolist()):
                    payload[f"train/loss_k{k}"] = v
                for f, (s, c) in gain.items():
                    payload[f"sel/gain_{f}"] = s / max(c, 1)
                if reward_on:
                    payload.update({"reward/r_mean": reward_stats["r"] / max(reward_stats["n"], 1), "reward/rgb_score": reward_stats["rgb"],
                                    "reward/proj_score": reward_stats["proj"], "reward/rgb_proj_corr": reward_stats["corr"],
                                    "reward/refresh_loss": reward_stats["refresh_loss"],
                                    "train/loss_cd": (_wcd_sum / _wcd_n) if _wcd_n else loss_cd_v,
                                    "reward/grad_norm_cd": reward_stats["g_cd"], "reward/grad_norm_reward": reward_stats["g_r"],
                                    "reward/grad_ratio": reward_stats["g_r"] / reward_stats["g_cd"] if reward_stats["g_cd"] == reward_stats["g_cd"] else float("nan"),
                                    "reward/grad_cos": reward_stats["g_cos"]})
                    reward_stats["r"] = 0.0; reward_stats["n"] = 0
                if rank_head is not None:
                    _n = max(rank_stats["n"], 1); _nc = max(rank_stats["n_cap"], 1)
                    payload.update({"rank/loss": rank_stats["loss"] / _n, "rank/acc_shaped": rank_stats["acc"] / _n, "rank/acc_raw": rank_stats["acc_raw"] / _n,
                                    "rank/margin_shaped": rank_stats["margin"] / _n, "rank/margin_raw": rank_stats["margin_raw"] / _n,
                                    "rank/r_pos_shaped": rank_stats["r_pos"] / _nc, "rank/negatives_per_caption": rank_stats["nneg"] / _nc,
                                    "rank/acc_rgb": rank_stats["acc_rgb"], "rank/acc_rgb_shaped": rank_stats["acc_rgb_shaped"],
                                    "rank/grad_norm_rank": rank_stats["g_rank"], "rank/grad_norm_align": rank_stats["g_align"],
                                    "rank/seconds_per_caption": rank_stats["t"] / _nc, "rank/head_frozen": float(rank_frozen),
                                    "rank/shaped_scale": rank_scale["shaped"], "rank/lambda_rank": rank_scale["rank"], "rank/lambda_align": rank_scale["align"]})
                    if rank_stats["n_align"]:
                        payload["rank/align_loss"] = rank_stats["align"] / rank_stats["n_align"]
                try:
                    wandb.log(payload, step=gstep)
                except Exception as e:      # a dead wandb service must not take a 13-hour run down
                    _wandb_fail += 1
                    if _wandb_fail <= 5 or gstep % (args.log_every * 50) == 0:
                        import traceback
                        print(f"[wandb] log failed at step {gstep} (failure #{_wandb_fail}): {type(e).__name__}: {e}", flush=True)
                        traceback.print_exc()
        if gstep % args.log_every == 0:   # every rank: start a fresh logging window
            _wl_sum, _wl_n, _wk_sum, _wcd_sum, _wcd_n = 0.0, 0, None, 0.0, 0
            if rank_head is not None:
                for _k in ("loss", "acc", "acc_raw", "margin", "margin_raw", "r_pos", "nneg", "align", "t"):
                    rank_stats[_k] = 0.0
                rank_stats["n"] = rank_stats["n_cap"] = rank_stats["n_align"] = 0
        if (is_main and sample_prompts and args.sample_every > 0 and gstep % args.sample_every == 0):
            if gstep == args.sample_every:          # once: the guided 28-step teacher as reference
                log_samples(teacher, pipe, sample_prompts, 28, args.cfg, args.height, device, gstep,
                            "samples/teacher_28step", neg_emb, neg_pool)
            log_samples(student_module, pipe, sample_prompts, args.sample_steps, args.sample_cfg,
                        args.height, device, gstep, "samples/student", neg_emb, neg_pool)
        if gstep % args.save_every == 0 and gstep != args.num_steps:
            if is_main:
                torch.save({"model": student_module.state_dict(), "step": gstep, "selector": args.selector},
                           Path(args.output_dir) / f"checkpoint_step{gstep}.pt")
                if rank_head is not None:
                    torch.save({"head": rank_head.state_dict(), "align_proj": (align.proj.state_dict() if align is not None else None),
                                "proj": proj.state_dict(), "step": gstep,
                                "rank_args": {k: v for k, v in vars(args).items() if k.startswith(("rank_", "align_"))}},
                               Path(args.output_dir) / f"rank_head_step{gstep}.pt")
            barrier()

    if is_main:
        torch.save({"model": student_module.state_dict(), "step": gstep, "selector": args.selector},
                   Path(args.output_dir) / "checkpoint_final.pt")
        if rank_head is not None:
            torch.save({"head": rank_head.state_dict(), "align_proj": (align.proj.state_dict() if align is not None else None),
                        "proj": proj.state_dict(), "step": gstep,          # the REFRESHED projector the head was trained on
                        "rank_args": {k: v for k, v in vars(args).items() if k.startswith(("rank_", "align_"))}},
                       Path(args.output_dir) / "rank_head_final.pt")
        if args.wandb_project:
            import wandb
            try:
                wandb.finish()
            except Exception as e:
                print(f"[wandb] finish failed: {type(e).__name__}: {e}", flush=True)
    teardown()


if __name__ == "__main__":
    main()
