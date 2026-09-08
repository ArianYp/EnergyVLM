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
Every --reward_monitor_every updates the 32 most recent predictions are decoded and scored by the
OFFLINE scorer (8-bit image, HF processor) and logged next to the reward: the hacking monitor.
Note: loading the scorer (and constructing the projector) advances the global torch generator
before the sampler's first draw, so a reward arm visits the captions in a different order from a
reward-free run with the same --seed (initial weights, noise-level draws and candidate seeds are
unchanged). Kept as is so the released trainer reproduces the reported runs bit for bit.

--accum M accumulates M captions per optimizer update (batch M per GPU at the same per-update
cost); --num_steps counts optimizer updates, so the data budget is num_steps * accum * world.

Multi-GPU under torchrun: a DistributedSampler shows every caption once per global epoch, so
num_steps = epochs * n_captions / (world_size * accum).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

SELECTORS = ("random", "dino_patch", "latent", "boltzmann", "boltzmann_sample", "boltzmann_frozen", "boltzmann_mc",
             "uniform_visit")
# score field each selector ranks by; the boltzmann selectors take theirs from --score_field
FIELD = {"dino_patch": "dino_patch_cos", "latent": "latent_cos"}


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
    if selector in ("dino_patch", "latent"):
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


SCORE_FIELDS = ("dino_patch_cos", "latent_cos", "dino_cos", "clip_cos", "endpoint_vqa")


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
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, args.num_warmup_steps)))

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

    reward_on = args.reward_mode != "none" and args.reward_lambda != 0.0
    loss_cd_v = None
    proj = rgb_reward = scorer = None
    if reward_on:
        assert not exact_soft, "the reward arms are single-candidate arms"
        assert world == 1 or args.reward_refresh_every == 0, "projector refresh is implemented for a single GPU"
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
        reward_stats = {"r": 0.0, "n": 0, "rgb": float("nan"), "proj": float("nan"), "corr": float("nan"), "refresh_loss": float("nan")}
        if is_main:
            print(f"[reward] mode={args.reward_mode} projector {args.reward_proj if proj is not None else None} lambda={args.reward_lambda} "
                  f"states={args.reward_states} refresh_every={args.reward_refresh_every} monitor_every={args.reward_monitor_every} refs={len(ref_emb)}", flush=True)

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
                    _r = (_ehat * _eref[None]).sum(-1)
                _lr = -args.reward_lambda * _r.mean()
                if is_main and gstep < args.reward_grad_probe:
                    _ps = [p for p in student_module.parameters() if p.requires_grad]
                    _g1 = torch.autograd.grad(loss, _ps, retain_graph=True, allow_unused=True)
                    _g2 = torch.autograd.grad(_lr, _ps, retain_graph=True, allow_unused=True)
                    _n1 = float(torch.sqrt(sum((g.float() ** 2).sum() for g in _g1 if g is not None)))
                    _n2 = float(torch.sqrt(sum((g.float() ** 2).sum() for g in _g2 if g is not None)))
                    _dot = float(sum((a.float() * b.float()).sum() for a, b in zip(_g1, _g2) if a is not None and b is not None))
                    print(f"[reward-probe] step {gstep} ||grad CD||={_n1:.2f} ||grad reward||={_n2:.4f} ratio={_n2 / max(_n1, 1e-9):.4f} "
                          f"cos(CD,reward)={_dot / max(_n1 * _n2, 1e-9):.4f} r={float(_r.mean()):.4f}", flush=True)
                    del _g1, _g2
                loss_cd_v = float(loss.detach())
                loss = loss + _lr
                reward_stats["r"] += float(_r.mean()); reward_stats["n"] += 1
                rbuf.append((_xr.detach()[:1].clone(), int(rec["idx"])))
                if len(rbuf) > 64:
                    rbuf.pop(0)
            (loss / args.accum).backward()
        micro += 1
        if micro % args.accum != 0:
            continue                      # accumulate: no clip, no step, no logging until the window closes
        gn = torch.nn.utils.clip_grad_norm_(student_module.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        gstep += 1
        pbar.update(1)
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

        if is_main and gstep % args.log_every == 0:
            lv, g = float(loss.detach().item()), float(gn)
            pbar.set_postfix({"loss": f"{lv:.3g}", "g": f"{g:.2g}"})
            if args.wandb_project:
                import wandb
                now = time.time()
                n = max(sel_stats["n"], 1)
                payload = {"train/loss": lv, "train/grad_norm": g, "train/lr": sched.get_last_lr()[0],
                           "train/epoch": epoch, "train/samples_seen": micro * world,
                           "train/steps_per_s": args.log_every / max(now - t_last, 1e-6),
                           "train/gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                           "sel/weight_ess": sel_stats["ess"] / n, "sel/weight_entropy": sel_stats["entropy"] / n,
                           "sel/target_churn": (sel_stats["churn"] / sel_stats["revisits"]) if sel_stats["revisits"] else float("nan")}
                t_last = now
                for k, v in zip(score_idxs, per_k.detach().tolist()):
                    payload[f"train/loss_k{k}"] = v
                for f, (s, c) in gain.items():
                    payload[f"sel/gain_{f}"] = s / max(c, 1)
                if reward_on:
                    payload.update({"reward/r_mean": reward_stats["r"] / max(reward_stats["n"], 1), "reward/rgb_score": reward_stats["rgb"],
                                    "reward/proj_score": reward_stats["proj"], "reward/rgb_proj_corr": reward_stats["corr"],
                                    "reward/refresh_loss": reward_stats["refresh_loss"], "train/loss_cd": loss_cd_v})
                    reward_stats["r"] = 0.0; reward_stats["n"] = 0
                wandb.log(payload, step=gstep)
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
            barrier()

    if is_main:
        torch.save({"model": student_module.state_dict(), "step": gstep, "selector": args.selector},
                   Path(args.output_dir) / "checkpoint_final.pt")
        if args.wandb_project:
            import wandb
            wandb.finish()
    teardown()


if __name__ == "__main__":
    main()
