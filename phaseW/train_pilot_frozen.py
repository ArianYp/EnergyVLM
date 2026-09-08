#!/usr/bin/env python3
"""
Phase C1 pilot trainer — B2 / B4 / B5 variants on the frozen-teacher M1 recipe.

Tests the paper's core question: does distilling an ORACLE-selected candidate
trajectory (B4) beat random-trajectory distillation (B2) and strong
selected-endpoint pseudo-data fine-tuning (B5)? Frozen base teacher, pure
consistency (no EMA / no encoder / no KL), 4-step-capable student.

Per step: re-roll N frozen-teacher candidates with the SAME deterministic seeds
as build_selection.py, select ONE by the cached index, and apply the variant's
loss. B2 & B4 see IDENTICAL candidate pools (fairness) — only the selected index
differs. All variants supervise K_w target pairs/update (matched).

--variant:
  B2         : distill the random_idx candidate's full trajectory (consistency)
  B4         : distill the oracle_idx candidate's full trajectory (consistency)
  B5_latent  : take the oracle endpoint clean latent, re-noise over the full
               timestep marginal, flow-match (no decode/re-encode round trip)
  B5_pixel   : same but decode->VAE-reencode the endpoint first (RFMI/CRAFT-style
               selected synthetic image pseudo-data; the realistic baseline)

Consistency target: x̃0_k = z_k − σ_k·v_k derived from consecutive cached states
(v_k = (z_{k+1}−z_k)/(σ_{k+1}−σ_k) is the CFG-mixed velocity). Student at the
noisier state z_{k−Δ} predicts x̂0; loss = pseudo-Huber(x̂0, sg[x̃0]).
"""
from __future__ import annotations
import argparse, copy, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_self_distill import (setup_distributed, dist_barrier, vae_decode,
                                 sample_teacher_images)


class SelectionDataset(Dataset):
    def __init__(self, records):
        self.items = records
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def rollout_states(teacher, scheduler, z0, prompt_embeds, pooled, neg_embeds, neg_pool,
                   K, cfg, device):
    """Frozen-teacher CFG Euler rollout; returns z_at_list [K+1] each [N,C,H,W].
    Identical math to build_selection's euler_cfg_sample (so the same seed → same
    candidate that was VQA-scored)."""
    N = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sigmas = scheduler.sigmas.to(device, torch.float32)
    ts = scheduler.timesteps.to(device)
    emb = torch.cat([neg_embeds.repeat(N, 1, 1), prompt_embeds.repeat(N, 1, 1)], 0)
    pool = torch.cat([neg_pool.repeat(N, 1), pooled.repeat(N, 1)], 0)
    z = z0
    zs = [z]
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v_all = teacher(hidden_states=torch.cat([z, z], 0), timestep=ts[k].expand(2 * N),
                            encoder_hidden_states=emb, pooled_projections=pool, return_dict=False)[0]
        v_u, v_c = v_all.chunk(2, 0)
        v = v_u + cfg * (v_c - v_u)
        z = (z.float() + (sigmas[k + 1] - sigmas[k]) * v.float()).to(torch.bfloat16)
        zs.append(z)
    return zs, sigmas




@torch.no_grad()
def log_student_samples(model, pipe, prompts, steps, cfg, height, device, gstep, tag, neg_emb, neg_pool):
    """Sample a fixed prompt set from `model` and log an image grid + a table to wandb.

    Uses only private generators (seeded by each prompt's idx, the evaluation generator's
    convention), so it never advances the RNG streams that decide the data order or the Delta
    draws: a run with sampling on is bit-identical to one with it off.
    """
    import wandb
    was_training = model.training
    model.eval()
    lat_c = model.config.in_channels
    h_lat = height // pipe.vae_scale_factor
    grid, rows = [], []
    for p in prompts:
        emb, _, pooled, _ = pipe.encode_prompt(prompt=[p["prompt"]], prompt_2=[p["prompt"]], prompt_3=[p["prompt"]],
                                               do_classifier_free_guidance=False, device=device,
                                               num_images_per_prompt=1)
        g = torch.Generator(device=device).manual_seed(int(p["idx"]))
        z0 = torch.randn(1, lat_c, h_lat, h_lat, device=device, dtype=torch.bfloat16, generator=g)
        zs, _ = rollout_states(model, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, steps, cfg, device)
        img = vae_decode(pipe.vae, zs[steps])
        u8 = ((img + 1) / 2).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        cap = f"[{p.get('bench', '')} {p['idx']}] {p['prompt'][:70]}"
        grid.append(wandb.Image(u8, caption=cap))
        rows.append([gstep, int(p["idx"]), p.get("bench", ""), p.get("category", ""), p["prompt"], wandb.Image(u8)])
    if was_training:
        model.train()
    wandb.log({f"{tag}/grid": grid,
               f"{tag}/table": wandb.Table(columns=["step", "idx", "bench", "category", "prompt", "image"], data=rows)},
              step=gstep)


DEFAULT_ASSIGN_SEED = 982_451_653


def assign_rng(assign_seed, idx):
    """Independent generator for the caption->candidate map.

    default_rng(assign_seed + idx) is WRONG for varying the map: shifting assign_seed by one gives
    the same stream one caption over, so two "different" maps agree on about half the captions and
    the variance they expose is far too small. Distinct seeds must be mixed, not added.

    The historical scalar path is kept for the default seed so every run already completed stays
    bit-reproducible.
    """
    if assign_seed == DEFAULT_ASSIGN_SEED:
        return np.random.default_rng(assign_seed + int(idx))
    return np.random.default_rng([int(assign_seed), int(idx)])


def target_schedule(idx, N, m, mode, n_visits, seed=DEFAULT_ASSIGN_SEED):
    # `seed` is the ASSIGNMENT seed. It is deliberately separate from the run seed so
    # that, by default, every seed of an arm trains on the same caption->candidate map
    # and the seeds isolate optimiser noise. Varying it resamples the map, which is the
    # variance component a seed-only bootstrap cannot see.
    """The full sequence of candidate indices one caption will train on, one entry per visit.

    Every mode with the same `m` returns the SAME MULTISET -- n_visits/m copies of each of m
    candidates -- and differs only in the ORDER. That is what separates temporal churn from the
    number of distinct targets, which the fixed-versus-resampled comparison cannot do: a resampled
    arm changes both at once.

      one_fixed  m=1, one candidate for every visit
      block      m>1, each candidate used for a long run of consecutive visits (fewest switches)
      cycle      m>1, round robin (most switches, n_visits-1 of them)
      shuffle    m>1, a fixed random permutation of the multiset

    The support (which m of the N candidates) is drawn once per caption from a fixed seed, so all
    modes at a given m see the same candidates and no label is involved anywhere.
    """
    rng = assign_rng(seed, idx)
    support = rng.permutation(N)[:m]
    reps = max(1, n_visits // m)
    if mode == "one_fixed" or m == 1:
        return [int(support[0])] * max(n_visits, 1)
    if mode == "block":
        seq = [int(c) for c in support for _ in range(reps)]
    elif mode == "cycle":
        seq = [int(support[v % m]) for v in range(reps * m)]
    elif mode == "shuffle":
        seq = [int(c) for c in support for _ in range(reps)]
        rng.shuffle(seq)
    else:
        raise ValueError(mode)
    while len(seq) < n_visits:                      # n_visits not divisible by m
        seq.append(seq[len(seq) % (reps * m)])
    return seq[:max(n_visits, 1)]


def pick_index(rec, variant, rng, alpha, temp, visit=0, sched_mode=None, support_m=1,
               n_visits=1, assign_seed=DEFAULT_ASSIGN_SEED):
    """Which cached candidate this update trains on.

    B2 and B4/B5_* keep their original behaviour exactly: the cached uniform draw and the cached
    VQAScore argmax. The W_* variants instead SAMPLE the index from a soft weighting

        w_i = (1 - alpha)/N + alpha * softmax(z_i / temp),      z = (s - mean(s)) / std(s)

    Sampling rather than weighting the loss is what keeps compute identical to B2/B4: the trainer
    rolls out exactly one trajectory per update, so drawing the index from w is an unbiased
    Monte-Carlo estimator of the weighted loss sum_i w_i d_i at the same cost.

    Two deliberate choices. Scores are z-scored by their within-prompt spread, so `temp` is
    scale-free and the same value means the same concentration for the energy (a cosine, spread
    ~1e-2) and for VQAScore (spread ~1e-1). And the (1-alpha)/N floor keeps every candidate at
    positive weight, so the objective can never improve by starving one -- the loser-degradation
    route that took most of the margin growth in this project's earlier pairwise run.
    """
    # A schedule arm ignores the label entirely: its candidates come from target_schedule and its
    # position in that sequence is the global visit number, so it is the same on every rank.
    if sched_mode is not None:
        seq = target_schedule(int(rec["idx"]), int(rec.get("N", 4)), support_m, sched_mode,
                              n_visits, seed=assign_seed)
        return int(seq[visit % len(seq)])
    w = selection_weights(rec, variant, alpha, temp)
    if int((w > 0).sum()) == 1:            # deterministic arms: B2, B4, B5_*, W_energy_hard
        return int(w.argmax())
    if variant.endswith("_freeze"):
        # ONE draw per caption, fixed for the whole run. The distribution is the matching _soft
        # arm's, so label quality is equal in expectation; only the target stops changing. This
        # is the control that separates target persistence from label quality.
        fixed = assign_rng(assign_seed, rec["idx"])
        return int(fixed.choice(len(w), p=w))
    return int(rng.choice(len(w), p=w))


def selection_weights(rec, variant, alpha, temp):
    """The probability this update trains on each cached candidate.

    Returned rather than consumed internally so the policy itself is loggable: without it, a null
    result cannot be attributed to the label, the weighting, or the student.
    """
    # NOT rec.get("N", len(rec["endpoint_vqa"])): Python builds the default eagerly, so that
    # form touches endpoint_vqa even when N is present and raises on a score-free cache.
    n = int(rec["N"]) if "N" in rec else len(rec["endpoint_vqa"])
    if variant in ("B2", "B4", "B5_latent", "B5_pixel"):
        w = np.zeros(n)
        w[int(rec["random_idx"] if variant == "B2" else rec["oracle_idx"])] = 1.0
        return w
    if variant.endswith("_freeze"):
        # weights are the soft ones; the DRAW is made deterministic per caption in pick_index.
        # This rewrite MUST come before the W_uniform_soft test: otherwise W_uniform_freeze misses
        # that test, is rewritten to W_uniform_soft, and then falls through to a key lookup for
        # "uniform", which raises. Ordering, not logic, was the bug.
        variant = variant.replace("_freeze", "_soft")
    if variant == "W_uniform_soft":
        return np.full(n, 1.0 / n)
    # W_<label>_{soft,hard}: <label> names the cached score this arm ranks candidates by. The
    # baselines all need a decode, so an arm that beats the energy is not a free win -- it is a
    # win that costs a VAE pass. LPIPS is cached pre-negated, so argmax is correct for every key.
    key = {"energy": "energy", "oracle": "endpoint_vqa", "dino": "dino_cos",
           "clip": "clip_cos", "lpips": "lpips_neg",
           # verifier-swap arms. NOTE neither is evaluator-independent: ImageReward is BLIP-based
           # (shares a family with CompBench's BLIP-VQA categories) and PickScore is a fine-tuned
           # CLIP-H (shares a family with CLIPScore, and plausibly CLIP-feature CMMD). They provide
           # verifier-family triangulation, not circularity elimination.
           "pick": "pick_score", "imgrwd": "imgrwd_score",
           # dinop = mean-pooled DINOv2 PATCH tokens (CLS dropped). The older "dino" key
           # is the CLS token, which is the weaker global-semantics summary and the one
           # every other DINO use in this project deliberately avoids.
           "dinop": "dino_patch_cos"}.get(variant.split("_")[1])
    if key is None or key not in rec:
        raise KeyError(f"variant {variant!r} needs cached field for {variant.split('_')[1]!r}; "
                       f"record has {sorted(k for k in rec if isinstance(rec[k], list))}")
    s = np.asarray(rec[key], dtype=float)
    if variant.endswith("_hard"):
        w = np.zeros(n)
        w[int(s.argmax())] = 1.0
        return w
    spread = s.std()
    logits = (s - s.mean()) / (spread * temp if spread > 0 and temp > 0 else 1.0)
    q = np.exp(logits - logits.max())
    q /= q.sum()
    w = (1.0 - alpha) / n + alpha * q
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["B2", "B4", "B5_latent", "B5_pixel",
                             "W_energy_soft", "W_oracle_soft", "W_energy_hard", "W_uniform_soft",
                             "W_energy_freeze", "W_oracle_freeze",
                             # B4 is NOT the ceiling of this family: B2 and B4 take the consistency
                             # branch (see is_consistency below) while every W_* arm takes the
                             # flow-matching branch. W_oracle_hard is B4's label under THIS
                             # objective, and W_uniform_freeze is the true persistence control for
                             # W_uniform_soft. Without them the ceiling and the control are
                             # cross-objective comparisons.
                             "W_oracle_hard", "W_uniform_freeze",
                             # reference-similarity baselines; each needs its cached field
                             "W_dino_hard", "W_dino_soft", "W_clip_hard", "W_clip_soft",
                             "W_lpips_hard", "W_lpips_soft",
                             # consistency objective with a non-oracle label: completes the
                             # objective x label matrix whose CD row was previously B2/B4 only
                             "CD_energy_hard", "CD_oracle_hard", "CD_uniform_freeze",
                             # Consistency-branch arms for the reference-similarity scorers. These
                             # need a cache carrying the matching field (coco_selection_108348 has
                             # clip_cos / dino_cos / lpips_neg; coco_selection_106573 does NOT).
                             # No new code path: `is_consistency` already matches CD_*, and
                             # selection_weights resolves the key from variant.split("_")[1].
                             "CD_clip_hard", "CD_dino_hard", "CD_lpips_hard",
                             "CD_pick_hard", "CD_imgrwd_hard", "CD_dinop_hard"])
    # Soft-weighting knobs. Only the W_* variants read them; B2/B4/B5_* are untouched.
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="preference mass. w_i = (1-alpha)/N + alpha*q_i, so alpha=0 is uniform "
                         "and every candidate keeps positive weight at any alpha (pull-only)")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="softmax temperature, in units of the within-prompt score spread")
    ap.add_argument("--sel_seed", type=int, default=0,
                    help="seed for the W_* candidate sampler only; does not touch data order")
    ap.add_argument("--selection_dir", default="phaseC")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--num_steps", type=int, default=6000)
    ap.add_argument("--num_warmup_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0, help="teacher rollout guidance scale")
    # 0.0 keeps the original behaviour EXACTLY: one conditional student forward, no guidance, i.e.
    # guidance distillation (the student internalises CFG and samples at cfg=1). Setting it to the
    # teacher's scale instead makes the student's own prediction CFG-mixed, so the only thing it
    # has to learn is the step reduction. That is a different operating point: the student then
    # needs two forwards at inference, so a 4-step student costs 8 NFE, not 4.
    ap.add_argument("--assign_seed", type=int, default=DEFAULT_ASSIGN_SEED,
                    help="seed for the caption->candidate map used by the _freeze arms and by "
                         "--target_schedule. It defaults to a constant, so every run seed shares "
                         "one map and the seed spread measures optimiser noise only. Vary it to "
                         "resample the map and expose that variance component.")
    ap.add_argument("--target_schedule", default=None,
                    choices=["one_fixed", "block", "cycle", "shuffle"],
                    help="overrides the label entirely. All modes at one --support_m train on the "
                         "SAME multiset of candidates per caption and differ only in the order, "
                         "which separates temporal churn from the number of distinct targets.")
    ap.add_argument("--support_m", type=int, default=1,
                    help="how many distinct candidates a caption may train on, 1..N")
    ap.add_argument("--coupling", default="fresh",
                    choices=["fresh", "fresh_shared", "teacher", "perm_within", "perm_global",
                             "fixed_unrelated"],
                    help="flow-matching noise. 'fresh' draws new noise for the selected endpoint, "
                         "discarding the teacher's own noise-to-image pairing. 'teacher' reuses the "
                         "very noise whose rollout produced that endpoint, so the interpolation "
                         "lies on the teacher's own path.")
    ap.add_argument("--paired_sigma", action="store_true",
                    help="draw the flow-matching sigmas from a generator keyed by "
                         "(seed, caption, visit) instead of the global one. The global stream is "
                         "consumed by --coupling fresh (which calls torch.randn for its noise) and "
                         "NOT by the coupled modes (which use a private generator), so from update "
                         "2 onward a fresh arm and a coupled arm see different sigma sequences. "
                         "That is fine when the effect is large but it is unpaired noise in "
                         "exactly the contrast a small-effect screen has to resolve. Off by "
                         "default so every completed run stays bit-reproducible.")
    ap.add_argument("--onpolicy_warmup", type=int, default=1000,
                    help="steps of teacher-trajectory training before switching to the student's "
                         "own states. On-policy from step 0 supervises a student whose trajectory "
                         "is still poor, which is a known way to make this diverge.")
    ap.add_argument("--cfg_mode", default="composed",
                    choices=["composed", "branch", "branch_onpolicy"],
                    help="composed: supervise only v^- + w(v^+ - v^-), which does not pin down "
                         "either branch. branch: give each branch its own target via a common "
                         "shift, preserving the teacher's CFG direction exactly. Only read when "
                         "--student_cfg > 0.")
    ap.add_argument("--student_cfg", type=float, default=0.0,
                    help="guidance scale for the STUDENT's own forward; 0 = no guidance")
    ap.add_argument("--scoring_window", default="0.4,0.9")
    ap.add_argument("--delta_min", type=int, default=1)
    ap.add_argument("--delta_max", type=int, default=3)
    ap.add_argument("--distill_loss", default="x0_huber",
                    choices=["x0_huber", "segment_velocity"],
                    help="x0_huber: our consistency target -- student at z_{k-Delta} regresses the "
                         "teacher's Tweedie x0 at k, pseudo-Huber in x0 space. segment_velocity: "
                         "the REST/progressive target -- student ON z_k regresses the chord to "
                         "z_{k+1}, plain MSE, no Delta. Consistency branch only.")
    ap.add_argument("--huber_c", type=float, default=0.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--sched_shift", type=float, default=3.0)
    ap.add_argument("--num_train_ts", type=float, default=1000.0)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--sample_prompts", default=None,
                    help="json list of {idx, prompt[, bench]} sampled by the student every "
                         "--sample_every steps and logged to wandb as an image grid and a table; "
                         "noise is seeded by idx, matching the evaluation generator")
    ap.add_argument("--sample_steps", type=int, default=4)
    ap.add_argument("--sample_cfg", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="sd-phaseC-pilot")
    ap.add_argument("--wandb_run_name", default=None)
    args = ap.parse_args()

    rank, world, local_rank, device, is_main = setup_distributed(0)
    torch.manual_seed(args.seed + rank * 1009)
    random.seed(args.seed + rank * 1009)
    # The Delta draws get a PRIVATE generator: wandb.init() consumes one draw from the global
    # `random` stream, so with the global generator a run with logging on trains on different
    # supervised states from step 1 than the same run with logging off (measured 2026-09-02).
    delta_rng = random.Random(args.seed + rank * 1009)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    dist_barrier()

    # ── merged selection cache ──
    recs = []
    for f in sorted(Path(args.selection_dir).glob("selection_rank*.jsonl")):
        for ln in f.read_text().splitlines():
            try: recs.append(json.loads(ln))
            except Exception: pass
    recs.sort(key=lambda r: r["idx"])
    if is_main:
        # A score-free cache has no endpoint_vqa, so there is no headroom to report. Say so rather
        # than crash: the arms that use such a cache never read a label.
        if recs and "endpoint_vqa" in recs[0]:
            gains = [r["endpoint_vqa"][r["oracle_idx"]] - sum(r["endpoint_vqa"]) / len(r["endpoint_vqa"])
                     for r in recs]
            print(f"[r{rank}] variant={args.variant} | {len(recs)} prompts | "
                  f"mean oracle-vs-random endpoint gain {sum(gains)/len(gains):+.4f}")
        else:
            print(f"[r{rank}] variant={args.variant} | {len(recs)} prompts | score-free cache")

    # ── SD3.5: student (fp32 master, trainable) + frozen teacher (bf16) ──
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters(): p.requires_grad = False
    student = pipe.transformer
    student.to(dtype=torch.float32).train()
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    teacher = copy.deepcopy(student).to(dtype=torch.bfloat16).eval()
    for p in teacher.parameters(): p.requires_grad = False

    student_module = student; student_ddp = student
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        student_ddp = DDP(student, device_ids=[device.index], find_unused_parameters=False)
        student_module = student_ddp.module

    opt = torch.optim.AdamW([p for p in student_module.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, args.num_warmup_steps)))

    lat_c = student_module.config.in_channels
    H_lat = args.height // pipe.vae_scale_factor
    huber_c = args.huber_c if args.huber_c > 0 else 0.00054 * (lat_c * H_lat * H_lat) ** 0.5
    win_lo, win_hi = (float(x) for x in args.scoring_window.split(","))
    K = args.K
    score_idxs = list(range(max(1, round(win_lo * K)), min(K - 1, round(win_hi * K)) + 1))
    # CD_* arms take the consistency branch with a NON-oracle label, which B2 and B4 cannot express:
    # B2 is fixed-random and B4 is VQAScore top-1, so the consistency row of the objective x label
    # matrix had no PRS cell. Naming them CD_* keeps the objective visible in the arm name.
    is_consistency = args.variant in ("B2", "B4") or args.variant.startswith("CD_")

    # The two branches get their timestep from DIFFERENT places, and only one of them can drift.
    # The consistency branch reads pipe.scheduler.timesteps, so it is correct by construction. The
    # flow-matching branch rebuilds the schedule itself from --sched_shift and --num_train_ts, which
    # are argparse defaults. If either stops matching the loaded scheduler, the student is
    # conditioned on the wrong timestep and nothing announces it -- the same silent-drift failure
    # that cost a run in an earlier phase. Assert the two agree instead of trusting the defaults.
    _sc = pipe.scheduler.config
    assert abs(float(_sc.num_train_timesteps) - args.num_train_ts) < 1e-6, (
        f"--num_train_ts {args.num_train_ts} != scheduler num_train_timesteps "
        f"{_sc.num_train_timesteps}; the flow-matching timestep would be mis-scaled")
    assert not getattr(_sc, "use_dynamic_shifting", False), (
        "scheduler uses dynamic shifting, so a single --sched_shift cannot reproduce its sigmas")
    assert abs(float(_sc.shift) - args.sched_shift) < 1e-6, (
        f"--sched_shift {args.sched_shift} != scheduler shift {_sc.shift}; the flow-matching "
        f"branch would sample sigmas the teacher never visits")
    if is_main:
        print(f"[schedule] shift={_sc.shift} num_train_timesteps={_sc.num_train_timesteps} "
              f"match the trainer flags", flush=True)

    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)

    sampler = DistributedSampler(SelectionDataset(recs), num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(SelectionDataset(recs), batch_size=1, sampler=sampler,
                        shuffle=(sampler is None), drop_last=True, collate_fn=lambda b: b[0])
    data_iter = iter(loader); epoch = 0

    sample_prompts = None
    if is_main and not args.no_wandb and args.sample_prompts and args.sample_every > 0:
        sample_prompts = json.loads(Path(args.sample_prompts).read_text())
    if is_main and not args.no_wandb:
        import wandb
        cfg_all = dict(vars(args))
        # Make the run self-describing: the cache's generation settings decide what the cached
        # energies and VQAScores even refer to, so they belong in the run config, not just on disk.
        try:
            meta = json.loads((Path(args.selection_dir) / "cache_meta.json").read_text())
            cfg_all.update({f"cache_{k}": v for k, v in meta.items()})
        except Exception:
            pass
        cfg_all["n_prompts"] = len(recs)
        cfg_all["world_size"] = world
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=cfg_all, save_code=True)
        # Snapshot the code that produced this run: the git commit, the working-tree diff (this
        # tree is rarely clean) and every tracked-or-not source file as a code artifact.
        try:
            import subprocess
            _root = Path(__file__).resolve().parent.parent
            _git = lambda *a: subprocess.run(["git", *a], cwd=_root, capture_output=True, text=True).stdout  # noqa: E731
            _sha = _git("rev-parse", "HEAD").strip()
            _diff = _git("diff", "HEAD", "--", "*.py", "*.lsf", "*.sh")
            wandb.config.update({"git_commit": _sha, "git_dirty": bool(_diff.strip()),
                                 "torch": torch.__version__, "cuda": torch.version.cuda}, allow_val_change=True)
            _patch = Path(args.output_dir) / "git_diff.patch"
            _patch.write_text(_diff)
            wandb.save(str(_patch), base_path=str(Path(args.output_dir)), policy="now")
            # An explicit file list, NOT wandb.run.log_code(root=repo): log_code walks the whole
            # tree before filtering, and this repo root holds caches, checkpoints and tens of
            # thousands of evaluation images on Lustre (a run hung >25 min in that walk).
            import glob as _glob
            _pats = ["phaseC/*.py", "train_self_distill.py", "exp0/*.py", "phaseN/build_coco_*.py",
                     "phaseW/*.py", "ablations/phaseW_*.lsf", "ablations/phaseN_eval_*.lsf",
                     "ablations/phaseT_fid_*.lsf", "ablations/phaseV_train.lsf"]
            _art = wandb.Artifact(f"source-{wandb.run.id}", type="source",   # 'code' is reserved by wandb
                                  metadata={"git_commit": _sha, "git_dirty": bool(_diff.strip())})
            for _pat in _pats:
                for _f in sorted(_glob.glob(str(_root / _pat))):
                    _art.add_file(_f, name=str(Path(_f).relative_to(_root)))
            _art.add_file(str(_patch), name="git_diff.patch")
            wandb.run.log_artifact(_art)
        except Exception as _e:
            print(f"[wandb] code snapshot skipped: {_e}", flush=True)

    # Independent of the data order, so two arms see the same prompts in the same order and differ
    # only in which candidate each update trains on.
    # Global visits per caption. DistributedSampler reshuffles the whole set each epoch and then
    # splits it, so every caption is seen exactly ONCE per global epoch regardless of rank. The
    # local epoch counter therefore IS the global visit index, and a schedule indexed by it is the
    # same on every rank without any communication.
    n_visits = max(1, int(round(args.num_steps * world / max(len(recs), 1))))
    if is_main and args.target_schedule is not None:
        print(f"[schedule] {args.target_schedule} m={args.support_m} over {n_visits} visits/caption",
              flush=True)
    if is_main:
        # Self-certify the two things a paired comparison silently gets wrong. Kw is the treatment
        # in the Phase-X window study, and the caption->candidate map is the field that made every
        # earlier fresh-vs-coupled contrast unpaired: the freeze path and the one_fixed schedule
        # path draw from the same generator but with different calls, so they agree at chance.
        # Printing the resolved map hash lets two arms be checked from their logs alone.
        import hashlib
        _h = hashlib.sha256()
        for _r in recs:
            _j = pick_index(_r, args.variant, np.random.default_rng(0), args.alpha, args.temp,
                            visit=0, sched_mode=args.target_schedule, support_m=args.support_m,
                            n_visits=n_visits, assign_seed=args.assign_seed)
            _h.update(f"{int(_r['idx'])}:{int(_j)}\n".encode())
        print(f"[window] scoring_window={args.scoring_window} K={K} score_idxs={score_idxs} "
              f"Kw={len(score_idxs)}", flush=True)
        print(f"[pairing] coupling={args.coupling} paired_sigma={args.paired_sigma} "
              f"assign_seed={args.assign_seed} target_map_sha256={_h.hexdigest()}", flush=True)
    sel_rng = np.random.default_rng(args.sel_seed + 7919 * rank)
    # Running selection diagnostics. `rho_running` is the headroom recovery of the policy this arm
    # is ACTUALLY executing, accumulated over the updates it performs. It is the number that makes
    # a null interpretable: if the student does not improve but rho_running is high, the failure is
    # in the student or the objective, not in the label. Rank 0 only, over its own shard.
    acc = {"num": 0.0, "den": 0.0, "n": 0, "agree": 0.0, "ess": 0.0,
           "w_on_oracle": 0.0, "vqa_sel": 0.0, "energy_sel": 0.0, "idx_hist": np.zeros(args.N),
           "churn": 0.0, "revisits": 0.0,
           # per cached score field: sum of (score of the selected candidate - mean over the N
           # candidates) and a count, i.e. what THIS arm's selection buys under EVERY scorer
           "gain": {}}
    seen_idx: dict[int, int] = {}   # caption -> the candidate its last visit trained on
    pbar = tqdm(total=args.num_steps, disable=not is_main, desc=f"pilot-{args.variant}")
    gstep = 0
    t_last = time.time()
    while gstep < args.num_steps:
        try: rec = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None: sampler.set_epoch(epoch)
            data_iter = iter(loader); rec = next(data_iter)
        # B2 = cached random candidate; B4/B5_* = oracle-selected candidate (B5 uses its
        # endpoint); W_* = drawn from a soft weighting over candidates, see pick_index.
        prompt = rec["prompt"]
        sel = pick_index(rec, args.variant, sel_rng, args.alpha, args.temp,
                         visit=epoch, sched_mode=args.target_schedule,
                         support_m=args.support_m, n_visits=n_visits,
                         assign_seed=args.assign_seed)
        if is_main and "endpoint_vqa" in rec:
            # A score-free cache (phaseN/build_caption_records.py) carries no VQAScore, so the
            # headroom diagnostics do not exist. Skip them rather than invent a value; the arms
            # that use such a cache do not read a label at all.
            _w = selection_weights(rec, args.variant, args.alpha, args.temp)
            _R = np.asarray(rec["endpoint_vqa"], dtype=float)
            _E = np.asarray(rec.get("energy", [float("nan")] * len(_R)), dtype=float)
            _den = float(_R.max() - _R.mean())
            _p = _w[_w > 0]
            acc["num"] += float(_R[sel] - _R.mean())
            acc["den"] += _den
            acc["n"] += 1
            acc["agree"] += float(sel == int(rec["oracle_idx"]))
            acc["ess"] += float(np.exp(-(_p * np.log(_p)).sum()))
            acc["w_on_oracle"] += float(_w[int(rec["oracle_idx"])])
            acc["vqa_sel"] += float(_R[sel])
            acc["energy_sel"] += float(_E[sel]) if np.isfinite(_E[sel]) else 0.0
            acc["idx_hist"][sel] += 1.0
            for _f in ("dino_patch_cos", "dino_cos", "clip_cos", "endpoint_vqa", "energy",
                       "pick_score", "imgrwd_score", "lpips_neg"):
                if _f in rec:
                    _s = np.asarray(rec[_f], dtype=float)
                    if np.all(np.isfinite(_s)):
                        _g = acc["gain"].setdefault(_f, [0.0, 0])
                        _g[0] += float(_s[sel] - _s.mean()); _g[1] += 1
            # TARGET churn, which is the statistic the _freeze arms exist to control. It is NOT the
            # ESS: ESS describes the weight distribution, and a _freeze arm keeps the soft weights
            # while making the draw once per caption, so its ESS is that of the matching _soft arm.
            # Only this counter separates them from W&B alone. Churn is measured over REVISITS, so
            # the first visit to a caption cannot contribute.
            _prev = seen_idx.get(rec["idx"])
            if _prev is not None:
                acc["revisits"] += 1.0
                acc["churn"] += float(_prev != sel)
            seen_idx[rec["idx"]] = sel
        seed_base = rec["seed_base"]

        with torch.no_grad():
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            # Roll out ONLY the selected candidate (its seed = seed_base + sel), 4× cheaper
            # than re-rolling all N and identical to what build_selection scored. The N-pool
            # fairness lives in the offline selection; here both B2 and B4 roll a single
            # trajectory from the same seed family, differing only in which index.
            g = torch.Generator(device=device).manual_seed(seed_base + sel)
            z0 = torch.randn(1, lat_c, H_lat, H_lat, device=device, dtype=torch.bfloat16, generator=g)
            zs, sigmas = rollout_states(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool,
                                        K, args.cfg, device)   # zs[k]: [1,C,H,W]
            z_sel = [zs[k].float() for k in range(K + 1)]   # each [1,C,H,W]

            # ON-POLICY STATES.
            #
            # Every state above lies on the TEACHER's trajectory. At sampling time the student walks
            # its OWN trajectory, which drifts away from that. With one conditional call the drift
            # costs little. With CFG the two branches are unconstrained off-trajectory and are then
            # multiplied by +w and 1-w, so the same drift is amplified w-fold. A diagnostic
            # (phaseN/diag_cfg_branch.py) ruled out loss scale, gradient magnitude and gradient
            # cancellation as causes of the branch-mode failure, which leaves this.
            #
            # So: replace the states with ones the STUDENT produces, and rebuild the target there by
            # letting the teacher advance from that state. Same Tweedie construction as the
            # off-policy path, evaluated where the student actually goes.
            if args.cfg_mode == "branch_onpolicy" and gstep >= args.onpolicy_warmup:
                sched_ts = pipe.scheduler.timesteps.to(device)
                z_cur = z0.float()
                for k in range(K):
                    with torch.autocast("cuda", torch.bfloat16):
                        v2 = student_module(
                            hidden_states=torch.cat([z_cur, z_cur], 0).to(torch.bfloat16),
                            timestep=sched_ts[k].expand(2),
                            encoder_hidden_states=torch.cat([neg_emb, emb], 0),
                            pooled_projections=torch.cat([neg_pool, pooled], 0),
                            return_dict=False)[0]
                    vu2, vc2 = v2.float().chunk(2, 0)
                    z_cur = z_cur + (sigmas[k + 1] - sigmas[k]) * (
                        vu2 + args.student_cfg * (vc2 - vu2))
                    z_sel[k + 1] = z_cur
                # z_sel[0] is the shared noise; z_sel[1..K] now lie on the student's own path.

        opt.zero_grad(set_to_none=True)
        per_k = None                       # per-supervised-state loss, x0_huber branch only
        if is_consistency:
            # one Δ per teacher score index; batch the K_w student forwards
            #
            # --distill_loss segment_velocity reproduces the REST/progressive-distillation target
            # instead: the student sits ON the supervised state z_k and regresses the teacher's
            # realised chord across ITS OWN step, v_k = (z_{k+1}-z_k)/(sigma_{k+1}-sigma_k).
            # Composing K such steps retraces the teacher's subsampled trajectory. There is no Δ
            # randomisation because the student's step and the target segment must span the same
            # interval, so Δ is pinned to 0 here rather than sampled.
            if args.distill_loss == "segment_velocity":
                deltas = {kt: 0 for kt in score_idxs}
            else:
                deltas = {kt: delta_rng.randint(args.delta_min, args.delta_max) for kt in score_idxs}
            stu_idx = [kt - deltas[kt] for kt in score_idxs]
            z_in = torch.cat([z_sel[ks] for ks in stu_idx], 0).to(torch.bfloat16)   # [K_w,C,H,W]
            t_in = torch.cat([pipe.scheduler.timesteps.to(device)[ks].reshape(1) for ks in stu_idx], 0)
            n_w = len(score_idxs)
            if args.student_cfg > 0:
                # Student predicts its OWN CFG-mixed velocity, matching the teacher's guidance, so
                # the only gap left to close is the step reduction. Costs two forwards per state.
                with torch.autocast("cuda", torch.bfloat16):
                    v_both = student_ddp(
                        hidden_states=torch.cat([z_in, z_in], 0),
                        timestep=torch.cat([t_in, t_in], 0),
                        encoder_hidden_states=torch.cat(
                            [neg_emb.repeat(n_w, 1, 1), emb.repeat(n_w, 1, 1)], 0),
                        pooled_projections=torch.cat(
                            [neg_pool.repeat(n_w, 1), pooled.repeat(n_w, 1)], 0),
                        return_dict=False)[0]
                v_u, v_c = v_both.chunk(2, 0)
                v_stu = v_u + args.student_cfg * (v_c - v_u)
            else:
                with torch.autocast("cuda", torch.bfloat16):
                    v_stu = student_ddp(hidden_states=z_in, timestep=t_in,
                                        encoder_hidden_states=emb.repeat(n_w, 1, 1),
                                        pooled_projections=pooled.repeat(n_w, 1),
                                        return_dict=False)[0]
            sig_stu = torch.stack([sigmas[ks] for ks in stu_idx]).view(-1, 1, 1, 1)
            z_stu = torch.cat([z_sel[ks] for ks in stu_idx], 0)
            x_hat = z_stu - sig_stu * v_stu.float()
            # teacher Tweedie x̃0_k from consecutive states (CFG-mixed velocity)
            x_tea = []
            for kt in score_idxs:
                if args.cfg_mode == "branch_onpolicy" and gstep >= args.onpolicy_warmup:
                    # z_sel[kt+1] is now the STUDENT's next state, which carries the student's own
                    # error and would make the target self-referential. Advance the TEACHER one step
                    # out of the student's state instead, and take the Tweedie estimate from that.
                    with torch.no_grad():
                        zk = z_sel[kt]
                        with torch.autocast("cuda", torch.bfloat16):
                            vt1 = teacher(hidden_states=torch.cat([zk, zk], 0).to(torch.bfloat16),
                                          timestep=pipe.scheduler.timesteps.to(device)[kt].expand(2),
                                          encoder_hidden_states=torch.cat([neg_emb, emb], 0),
                                          pooled_projections=torch.cat([neg_pool, pooled], 0),
                                          return_dict=False)[0]
                        vu1, vc1 = vt1.float().chunk(2, 0)
                        dv = vu1 + args.cfg * (vc1 - vu1)
                else:
                    dv = (z_sel[kt + 1] - z_sel[kt]) / (sigmas[kt + 1] - sigmas[kt])
                x_tea.append((z_sel[kt] - sigmas[kt] * dv))
            x_tea = torch.cat(x_tea, 0).detach()
            if args.student_cfg > 0 and args.cfg_mode == "branch":
                # COMMON-SHIFT BRANCH MATCHING.
                #
                # Supervising only the CFG composition g_S = v^- + w(v^+ - v^-) does not pin down
                # either branch: any pair whose weighted difference is right satisfies it, and the
                # sampler then amplifies the individual errors by w and 1-w. So we build a target
                # for EACH branch instead.
                #
                # Take the teacher's two branches at the student's own state, and the shortcut
                # velocity the consistency target demands:
                #     g* = (z_s - x_tea) / sigma_s,      Delta = sg(g* - g_teacher).
                # Add the SAME Delta to both branches:
                #     v+_target = v_T^+ + Delta,   v-_target = v_T^- + Delta.
                # Two properties hold exactly, and both are asserted below on the first step:
                #     (v+_target - v-_target) == (v_T^+ - v_T^-)          direction preserved
                #     v-_target + w (v+_target - v-_target) == g*         composition is the target
                with torch.no_grad():
                    with torch.autocast("cuda", torch.bfloat16):
                        vt = teacher(hidden_states=torch.cat([z_in, z_in], 0),
                                     timestep=torch.cat([t_in, t_in], 0),
                                     encoder_hidden_states=torch.cat(
                                         [neg_emb.repeat(n_w, 1, 1), emb.repeat(n_w, 1, 1)], 0),
                                     pooled_projections=torch.cat(
                                         [neg_pool.repeat(n_w, 1), pooled.repeat(n_w, 1)], 0),
                                     return_dict=False)[0]
                    vt_neg, vt_pos = vt.float().chunk(2, 0)
                    g_tea_c = vt_neg + args.student_cfg * (vt_pos - vt_neg)
                    g_star = (z_stu - x_tea) / sig_stu
                    corr = (g_star - g_tea_c).detach()
                    tgt_pos, tgt_neg = vt_pos + corr, vt_neg + corr
                    if gstep == 0 and is_main:
                        _d = (tgt_pos - tgt_neg) - (vt_pos - vt_neg)
                        _c = (tgt_neg + args.student_cfg * (tgt_pos - tgt_neg)) - g_star
                        assert _d.abs().max() < 1e-3, f"direction not preserved: {_d.abs().max()}"
                        assert _c.abs().max() < 1e-2, f"composition != g_star: {_c.abs().max()}"
                        print(f"[cfg_mode=branch] identities hold: direction {_d.abs().max():.2e}, "
                              f"composition {_c.abs().max():.2e}", flush=True)

                def _ph(a, b):
                    s = (a - b).pow(2).sum(dim=(1, 2, 3))
                    return (torch.sqrt(s + huber_c * huber_c) - huber_c).mean()

                loss = 0.5 * (_ph(z_stu - sig_stu * v_c.float(), z_stu - sig_stu * tgt_pos)
                              + _ph(z_stu - sig_stu * v_u.float(), z_stu - sig_stu * tgt_neg))
            elif args.distill_loss == "segment_velocity":
                # REST-style per-step imitation: plain velocity MSE against the teacher's realised
                # segment chord. Note this is NOT the same vector as our x0 target. Writing our
                # objective in velocity space gives v* = (z_s - x_tea)/sigma_s, a shortcut toward
                # the denoised estimate; the target below is the chord to the NEXT state. They
                # coincide only when k+1 is the endpoint.
                v_seg = torch.cat(
                    [((z_sel[kt + 1] - z_sel[kt]) / (sigmas[kt + 1] - sigmas[kt])) for kt in score_idxs], 0
                ).detach()
                loss = (v_stu.float() - v_seg).pow(2).mean()
            else:
                diff = x_hat - x_tea
                sq = diff.pow(2).sum(dim=(1, 2, 3))
                d = torch.sqrt(sq + huber_c * huber_c) - huber_c
                loss = d.mean()
                per_k = d.detach()
        else:
            # B5: selected endpoint clean latent -> re-noise over full marginal -> flow-match
            x0 = z_sel[K]                                             # endpoint (σ≈0) clean latent
            if args.variant == "B5_pixel":
                with torch.no_grad():
                    img = vae_decode(pipe.vae, x0)                    # [-1,1]
                    post = pipe.vae.encode(img.to(pipe.vae.dtype)).latent_dist.mean
                    x0 = ((post - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor).float()
            x0 = x0.detach()
            Kw = len(score_idxs)
            with torch.no_grad():
                if args.paired_sigma:
                    # Keyed by (run seed, caption, visit) and NOT by the arm, so two arms that
                    # differ only in --coupling walk the same sigma sequence. Without this the
                    # fresh branch's torch.randn advances the global generator and the coupled
                    # branches' private generator does not, so the sigma streams diverge after
                    # update 1. SeedSequence mixing, not addition: adding the three numbers would
                    # collide across (caption, visit) pairs with the same sum.
                    # int() BEFORE the shift: numpy refuses >> on a uint64 scalar, and the shift
                    # keeps the value inside the int64 range manual_seed accepts.
                    gsig = torch.Generator(device=device).manual_seed(
                        int(np.random.SeedSequence(
                            [int(args.seed), int(rec["idx"]), int(epoch)]
                        ).generate_state(1, dtype=np.uint64)[0]) >> 1)
                    u = torch.rand(Kw, device=device, generator=gsig)
                else:
                    u = torch.rand(Kw, device=device)
                s = (args.sched_shift * u) / (1 + (args.sched_shift - 1) * u)
                s = s.clamp(1e-4, 1 - 1e-4)
                # WHICH noise is paired with the chosen endpoint. Reusing the endpoint's OWN
                # source makes (noise, x0) a pair the teacher realised, so the interpolation is the
                # straight chord of that pair. The other modes hold "a fixed noise per target"
                # constant while destroying the source-endpoint IDENTITY, which separates the two
                # explanations: is the gain from the pairing, or merely from the noise not changing?
                if args.coupling == "fresh":
                    noise = torch.randn(Kw, lat_c, H_lat, H_lat, device=device)
                elif args.coupling == "fresh_shared":
                    # ONE fresh draw shared by all Kw supervised states, redrawn on the next visit.
                    # This is the control that separates the two things every coupled mode does at
                    # once: sharing a chord within the update, and using the endpoint's true source.
                    # Note v_target = noise - x0 has no sigma dependence, so sharing the noise gives
                    # all Kw states an IDENTICAL regression target on one interpolation line, while
                    # `fresh` gives Kw different targets on Kw different lines.
                    # Deliberately drawn from the GLOBAL generator with leading dim 1: at Kw=1 that
                    # is byte-for-byte the same call `fresh` makes, so the two arms are bit-identical
                    # there and the Kw=1 cell is a built-in null rather than merely a matched
                    # distribution.
                    noise = torch.randn(1, lat_c, H_lat, H_lat, device=device
                                        ).expand(Kw, -1, -1, -1).contiguous()
                else:
                    if args.coupling == "teacher":
                        nseed = seed_base + sel                    # the true source of this x0
                    elif args.coupling == "perm_within":
                        nseed = seed_base + (sel + 1) % args.N     # another candidate, same caption
                    elif args.coupling == "perm_global":
                        nseed = seed_base + 7_919_000 + sel        # another caption's family
                    elif args.coupling == "fixed_unrelated":
                        nseed = seed_base + sel + 500_000_000      # fixed, never a rollout source
                    else:
                        raise ValueError(args.coupling)
                    gnz = torch.Generator(device=device).manual_seed(int(nseed))
                    one = torch.randn(1, lat_c, H_lat, H_lat, device=device, generator=gnz)
                    noise = one.expand(Kw, -1, -1, -1).contiguous()
                sb = s.view(-1, 1, 1, 1)
                z_t = (1 - sb) * x0 + sb * noise                      # [Kw,C,H,W]
                v_target = noise - x0
                t_fm = (s * args.num_train_ts)
            with torch.autocast("cuda", torch.bfloat16):
                v_fm = student_ddp(hidden_states=z_t.to(torch.bfloat16), timestep=t_fm,
                                   encoder_hidden_states=emb.repeat(Kw, 1, 1),
                                   pooled_projections=pooled.repeat(Kw, 1), return_dict=False)[0]
            loss = F.mse_loss(v_fm.float(), v_target.float())

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student_module.parameters(), args.grad_clip)
        opt.step(); sched.step()
        gstep += 1; pbar.update(1)

        if is_main and gstep % args.log_every == 0:
            lv = float(loss.detach().item()); g = float(gn.item() if torch.is_tensor(gn) else gn)
            pbar.set_postfix({"loss": f"{lv:.3g}", "g": f"{g:.2g}"})
            if not args.no_wandb:
                import wandb
                _n = max(acc["n"], 1)
                payload = {
                    "train/loss": lv, "train/grad_norm": g,
                    "train/lr": sched.get_last_lr()[0],
                    # --- what the selection policy is actually doing ---
                    "sel/rho_running": (acc["num"] / acc["den"]) if acc["den"] > 0 else float("nan"),
                    "sel/headroom_num_mean": acc["num"] / _n,
                    "sel/headroom_den_mean": acc["den"] / _n,
                    "sel/agree_oracle": acc["agree"] / _n,
                    "sel/vqa_of_selected": acc["vqa_sel"] / _n,
                    "sel/energy_of_selected": acc["energy_sel"] / _n,
                    "sel/weight_ess": acc["ess"] / _n,          # effective candidates, 1..N
                    "sel/weight_on_oracle": acc["w_on_oracle"] / _n,
                    "sel/updates_counted": acc["n"],
                    # 0.0 for every fixed-target arm (B2, B4, B5_*, *_hard, *_freeze) by design.
                    "sel/target_churn": (acc["churn"] / acc["revisits"]
                                         if acc["revisits"] > 0 else float("nan")),
                    "sel/revisits": acc["revisits"],
                }
                # slot histogram catches a policy that has collapsed onto one candidate index
                for _j in range(args.N):
                    payload[f"sel/frac_idx{_j}"] = float(acc["idx_hist"][_j] / _n)
                # --- run health: throughput, memory, where in the trajectory the loss sits ---
                _now = time.time()
                payload.update({
                    "train/epoch": epoch,
                    "train/samples_seen": gstep * world,
                    "train/steps_per_s": args.log_every / max(_now - t_last, 1e-6),
                    "train/gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                })
                t_last = _now
                if per_k is not None:
                    for _kt, _v in zip(score_idxs, per_k.tolist()):
                        payload[f"train/loss_k{_kt}"] = _v
                # selection gain of THIS arm's choice under every cached scorer, running mean
                for _f, (_s, _c) in acc["gain"].items():
                    payload[f"sel/gain_{_f}"] = _s / max(_c, 1)
                wandb.log(payload, step=gstep)

        # Periodic qualitative track: the student on a fixed prompt set (and, once, the 28-step
        # guided teacher on the same prompts as the reference). Private RNG, so training is
        # unaffected; rank 0 only, the other ranks wait at the next all-reduce.
        if (is_main and not args.no_wandb and sample_prompts
                and args.sample_every > 0 and gstep % args.sample_every == 0):
            if gstep == args.sample_every:
                log_student_samples(teacher, pipe, sample_prompts, 28, args.cfg, args.height, device,
                                    gstep, "samples/teacher_28step", neg_emb, neg_pool)
            log_student_samples(student_module, pipe, sample_prompts, args.sample_steps, args.sample_cfg,
                                args.height, device, gstep, "samples/student", neg_emb, neg_pool)

        # The final step is deliberately excluded: `checkpoint_final.pt` is written below with
        # identical contents, so the old `or gstep == args.num_steps` clause stored the same 9.88 GB
        # twice for every run. Verified byte-identical on a completed run before this was changed.
        if gstep % args.save_every == 0 and gstep != args.num_steps:
            if is_main:
                torch.save({"model": student_module.state_dict(), "step": gstep, "variant": args.variant},
                           Path(args.output_dir) / f"checkpoint_step{gstep}.pt")
                print(f"\n[{args.variant}] saved step {gstep}", flush=True)
            dist_barrier()

    if is_main:
        torch.save({"model": student_module.state_dict(), "step": gstep, "variant": args.variant},
                   Path(args.output_dir) / "checkpoint_final.pt")
        if not args.no_wandb:
            import wandb; wandb.finish()
    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
