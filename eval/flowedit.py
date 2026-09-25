#!/usr/bin/env python3
"""FlowEdit: inversion-free text-guided editing of one image with a distilled SD3.5 student.

FlowEdit (Kulikov et al., 2025) never inverts the source image. It keeps the clean source latent
z_src and an edited latent z (initialised to z_src), and at each sigma it couples the two branches
through the SAME noise draw:

    z_t_src  = (1 - sigma) * z_src + sigma * eps          the source ON the source's own flow
    z_t_edit = z + (z_t_src - z_src)                      the SAME displacement applied to the edit
    dv       = v_theta(z_t_edit, c_tgt) - v_theta(z_t_src, c_src)
    z       <- z + (sigma_next - sigma) * dv              Euler on the DIFFERENCE field only

The second line is the point of the method and is NOT the same as (1 - sigma) * z + sigma * eps once
z has drifted away from z_src: the edit branch is offset by the source's own noise displacement, so
the two velocities are evaluated at states that differ exactly by the accumulated edit. Integrating
only their difference is what leaves everything the two prompts agree about untouched.

This is a CUSTOM loop, not the stock pipeline. Our students are 4-step, guidance-free (the teacher's
CFG is internalised, so they are sampled at guidance 1.0) and trained at 512x512 -- see
docs/CHECKPOINTS.md. The defaults here are the student's operating point, NOT the 50-step / high-CFG
configuration the SoftREPA-style FlowEdit scripts use.

Grid A
------
The default sigma grid is grid A, `1, 0.882788, 0.693793, 0.337972, 0` -- states 0, 2, 4, 6 of the
8-step training grid, worth +0.006 CompBench at zero cost (README). These are POST-shift values and
are used verbatim: `pipe(..., sigmas=...)` would re-apply the scheduler shift (3.0) to them, which is
why this script never goes through the pipeline call. Timesteps follow the scheduler convention
t = sigma * num_train_timesteps.

n_max / n_min
-------------
The grid defines four intervals. `n_max` is the number of ACTIVE editing intervals counted from the
END (the clean end) of the schedule, `n_min` how many of the last intervals to leave out:

    active = range(num_intervals - n_max, num_intervals - n_min)

The default n_max=3, n_min=0 skips the pure-noise interval at sigma 1.0 (where the source structure
carries no information) and edits on 0.882788 -> 0.693793 -> 0.337972 -> 0.

Examples
--------
Final student (default, best checkpoint):

    python eval/flowedit.py \
      --checkpoint /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472/checkpoint_avg_last5.pt \
      --source_image /path/to/source.png \
      --source_prompt "a photo of a cat sitting on a sofa" \
      --target_prompt "a photo of a dog sitting on a sofa" \
      --output_dir out/flowedit/smoke/ours \
      --seed 0 \
      --n_max 3

Matched baseline (same recipe, no scored selection):

    python eval/flowedit.py \
      --checkpoint /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/phaseW_B2_3k-k10-hp1-acc4_s0_153468/checkpoint_avg_last5.pt \
      --source_image /path/to/source.png \
      --source_prompt "a photo of a cat sitting on a sofa" \
      --target_prompt "a photo of a dog sitting on a sofa" \
      --output_dir out/flowedit/smoke/baseline \
      --seed 0 \
      --n_max 3

The seed fixes the VAE posterior draw and the whole noise bank before any transformer runs, so the
two commands above differ in the student weights and in nothing else.

Pure-function checks that need neither a GPU nor the model:

    python eval/flowedit.py --self_test

Output: {output_dir}/source.png, edited.png, comparison.png, metadata.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import encode_prompt, vae_decode  # noqa: E402

# Post-shift grid A (states 0, 2, 4, 6 of the 8-step training grid). Four intervals.
GRID_A = [1.0, 0.882788, 0.693793, 0.337972, 0.0]
GRID_A_STR = "1,0.882788,0.693793,0.337972,0"
SIGMA_GRID_LEN = len(GRID_A)          # the students are 4-step: a 5-value grid, nothing else
TRAINED_RESOLUTION = 512              # docs/CHECKPOINTS.md: trained and evaluated at 512x512

# Seed derivation. One CLI seed feeds two disjoint, independently seeded streams, so that the VAE
# posterior draw cannot shift the noise bank (or vice versa) when either side changes. SEED_STRIDE
# is the same convention eval/generate.py uses to keep its per-candidate streams disjoint.
SEED_STRIDE = 1_000_000
VAE_STREAM = 0                        # seed + 0 * SEED_STRIDE -> the VAE posterior sample
NOISE_STREAM = 1                      # seed + 1 * SEED_STRIDE -> the FlowEdit noise bank


# --------------------------------------------------------------------------------------- schedule

def parse_sigmas(text: str) -> list[float]:
    """Parse and validate a sigma grid '1,s1,s2,s3,0' (POST-shift values, used verbatim).

    Requires exactly SIGMA_GRID_LEN strictly decreasing values from 1 to 0: these students are
    4-step, and a grid of any other length is a mistake, not a configuration.
    """
    sigmas = [float(x) for x in text.split(",") if x.strip()]
    if len(sigmas) != SIGMA_GRID_LEN:
        raise ValueError(f"--sigmas needs exactly {SIGMA_GRID_LEN} values "
                         f"({SIGMA_GRID_LEN - 1} intervals, the students are 4-step), got {len(sigmas)}: {sigmas}")
    if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError(f"--sigmas must start at 1 and end at 0, got {sigmas[0]} ... {sigmas[-1]}")
    if not all(b < a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError(f"--sigmas must be strictly decreasing, got {sigmas}")
    return sigmas


def active_intervals(num_intervals: int, n_min: int, n_max: int) -> list[int]:
    """Indices of the intervals FlowEdit actually integrates, counted from the clean end.

    n_max intervals are active, minus the last n_min. n_max=3, n_min=0 on a 4-interval grid gives
    [1, 2, 3]: the pure-noise interval at sigma 1.0 is skipped.
    """
    if not (0 <= n_min < n_max <= num_intervals):
        raise ValueError(f"need 0 <= n_min < n_max <= {num_intervals}, got n_min={n_min}, n_max={n_max}")
    return list(range(num_intervals - n_max, num_intervals - n_min))


def scheduler_grid(scheduler, num_steps: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The scheduler's own shifted grid of `num_steps` intervals, as common.sampling.rollout uses it."""
    scheduler.set_timesteps(num_steps, device=device)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    assert sigmas.numel() == num_steps + 1 and timesteps.numel() == num_steps, (sigmas.shape, timesteps.shape)
    return sigmas, timesteps


def need_uncond(args) -> bool:
    """Whether any branch runs a classifier-free-guidance unconditional forward."""
    return args.source_guidance > 1.0 or args.target_guidance > 1.0


def forward_budget(n_active: int, n_avg: int, source_guidance: float, target_guidance: float) -> dict[str, int]:
    """Transformer forwards this configuration will run. A guidance of 1.0 skips its uncond branch."""
    per_target = 2 if target_guidance > 1.0 else 1
    per_source = 2 if source_guidance > 1.0 else 1
    target = n_active * n_avg * per_target
    source = n_active * n_avg * per_source
    return {"target": target, "source": source, "total": target + source}


def build_noise_bank(num_intervals: int, n_avg: int, shape: tuple[int, ...],
                     device: torch.device, seed: int) -> list[list[torch.Tensor]]:
    """The complete FlowEdit noise bank, drawn before any transformer forward.

    One generator seeded from the CLI seed fills bank[i][a] in a fixed (interval, draw) order, for
    EVERY interval of the grid rather than only the active ones: the noise at interval i is then the
    same whatever n_max/n_min select, so an n_max sweep varies only the intervals it integrates. The
    bank is float32 because the FlowEdit state arithmetic is float32.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed) + NOISE_STREAM * SEED_STRIDE)
    return [[torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
             for _ in range(n_avg)] for _ in range(num_intervals)]


# ------------------------------------------------------------------------------------------ input

def load_source_image(path: Path, height: int, width: int) -> torch.Tensor:
    """PIL -> RGB -> bicubic (width, height) -> float32 [1, 3, H, W] in [-1, 1]."""
    if not path.is_file():
        raise SystemExit(f"[flowedit] source image not found: {path}")
    img = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())         # [H, W, 3] uint8
    x = x.permute(2, 0, 1).float().div_(255.0)                           # [3, H, W] in [0, 1]
    return (x * 2.0 - 1.0).unsqueeze(0)                                  # [1, 3, H, W] in [-1, 1]


def load_pipeline(model_id: str, device: torch.device):
    """Stock SD3.5 pipeline, every component explicitly cast to bfloat16 and frozen.

    torch_dtype alone leaves CLIP's text_projection in fp16 and prompt encoding dies with
    "Half != BFloat16" in this environment, so each module is cast by hand (docs/CHECKPOINTS.md).
    """
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    names = ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3")
    for name in names:
        m = getattr(pipe, name)
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        bad = [k for k, v in m.named_parameters() if v.dtype != torch.bfloat16]
        assert not bad, f"{name}: {len(bad)} parameters are not bfloat16, e.g. {bad[:3]}"
    print(f"[flowedit] pipeline components bfloat16: {', '.join(names)}", flush=True)
    return pipe


def load_student(pipe, checkpoint_path: Path) -> dict:
    """Swap the student transformer weights into the stock pipeline. Fails loudly on any mismatch."""
    if not checkpoint_path.is_file():
        raise SystemExit(f"[flowedit] checkpoint not found: {checkpoint_path}")
    if checkpoint_path.name == "checkpoint_final.pt":
        print(f"[flowedit] WARNING: {checkpoint_path.name} -- the averaged checkpoint_avg_last5.pt is "
              f"the one every number in this repository describes", flush=True)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise SystemExit(f"[flowedit] {checkpoint_path} has no 'model' entry "
                         f"(keys: {list(ck)[:10] if isinstance(ck, dict) else type(ck).__name__})")
    n_tensors = len(ck["model"])
    missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (missing[:10], unexpected[:10])
    pipe.transformer.to(dtype=torch.bfloat16).eval()
    meta = {"step": ck.get("step"), "variant": ck.get("variant"), "n_tensors": n_tensors}
    del ck                                        # the fp32 state dict is ~9.9 GB of host memory
    return meta


@torch.no_grad()
def encode_source_latent(pipe, image: torch.Tensor, device: torch.device, seed: int,
                         mode: str = "sample", explicit_seed: int | None = None) -> torch.Tensor:
    """VAE-encode the [-1, 1] source image to the SD3 latent space, deterministically.

    SoftREPA samples the posterior rather than taking its mode; that behaviour is kept, but the draw
    comes from a dedicated generator seeded off the CLI seed so it is reproducible and independent of
    the FlowEdit noise bank. `mode="mode"` takes latent_dist.mode() instead (also deterministic, and
    what the PIE-Bench brief prefers); `explicit_seed` overrides the CLI-derived seed, which the
    PIE-Bench runner uses to feed its own SHA-256 per-record seed.
    """
    posterior = pipe.vae.encode(image.to(device=device, dtype=pipe.vae.dtype)).latent_dist
    if mode == "mode":
        sample = posterior.mode()
    elif mode == "sample":
        s = int(explicit_seed) if explicit_seed is not None else int(seed) + VAE_STREAM * SEED_STRIDE
        gen = torch.Generator(device=device).manual_seed(s)
        try:
            sample = posterior.sample(generator=gen)
        except TypeError:                         # older diffusers: no generator= on sample()
            eps = torch.randn(posterior.mean.shape, generator=gen,
                              device=posterior.mean.device, dtype=posterior.mean.dtype)
            sample = posterior.mean + posterior.std * eps
    else:
        raise ValueError(f"vae encode mode must be 'sample' or 'mode', got {mode!r}")
    # SD3 latent transformation, the inverse of common.sampling.vae_decode.
    return (sample.float() - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


# --------------------------------------------------------------------------------------- flowedit

@torch.no_grad()
def flowedit(pipe, z_src: torch.Tensor, sigmas: torch.Tensor, timesteps: torch.Tensor,
             active: list[int], noise_bank: list[list[torch.Tensor]],
             src_emb, src_pooled, tgt_emb, tgt_pooled, neg_emb, neg_pooled,
             source_guidance: float, target_guidance: float,
             device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    """Integrate the FlowEdit difference field over the active intervals. Returns (z, forward counts).

    State arithmetic is float32 and only the transformer inputs are bfloat16: the update accumulates
    over several intervals and n_avg draws, and there is no reason to round it every time.
    """
    transformer = pipe.transformer
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else contextlib.nullcontext())
    counts = {"target": 0, "source": 0, "total": 0}

    def call(latent: torch.Tensor, t: torch.Tensor, emb, pooled, which: str) -> torch.Tensor:
        counts[which] += 1
        counts["total"] += 1
        with autocast:
            v = transformer(hidden_states=latent.to(torch.bfloat16), timestep=t,
                            encoder_hidden_states=emb, pooled_projections=pooled,
                            return_dict=False)[0]
        return v.float()

    z = z_src.clone()                              # the edited latent; z_src stays the clean source
    for i in active:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        t = timesteps[i].expand(z.shape[0])
        deltas = []
        for noise in noise_bank[i]:
            # The source ON its own flow at this sigma, and the edit displaced by exactly the same
            # amount. NOT (1 - sigma) * z + sigma * noise: once z has drifted these differ.
            z_t_source = (1.0 - sigma) * z_src + sigma * noise
            z_t_edit = z + (z_t_source - z_src)

            v_target = call(z_t_edit, t, tgt_emb, tgt_pooled, "target")
            if target_guidance > 1.0:              # conventional CFG: v_u + s * (v_c - v_u)
                v_u = call(z_t_edit, t, neg_emb, neg_pooled, "target")
                v_target = v_u + target_guidance * (v_target - v_u)

            v_source = call(z_t_source, t, src_emb, src_pooled, "source")
            if source_guidance > 1.0:
                v_u = call(z_t_source, t, neg_emb, neg_pooled, "source")
                v_source = v_u + source_guidance * (v_source - v_u)

            deltas.append(v_target - v_source)
        mean_delta_v = torch.stack(deltas).mean(dim=0) if len(deltas) > 1 else deltas[0]
        z = z + (sigma_next - sigma) * mean_delta_v      # Euler on the difference field
        print(f"[flowedit] interval {i}: sigma {float(sigma):.6f} -> {float(sigma_next):.6f} "
              f"(t={float(timesteps[i]):.2f}, {len(deltas)} draw(s))", flush=True)
    return z, counts


# ------------------------------------------------------------------------------------- self-tests

def run_self_test() -> None:
    """Pure-function checks: no GPU, no pipeline, no checkpoint. Raises AssertionError on failure."""
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    def rejects(fn, msg):
        try:
            fn()
        except (ValueError, AssertionError):
            return check(True, msg)
        raise AssertionError(f"expected a rejection: {msg}")

    # 1. the sigma grid parses to grid A and is strictly decreasing from 1 to 0
    parsed = parse_sigmas(GRID_A_STR)
    check(parsed == GRID_A, f"grid A round-trip: {parsed}")
    check(all(b < a for a, b in zip(parsed, parsed[1:])), "grid A strictly decreasing")
    check(parsed[0] == 1.0 and parsed[-1] == 0.0, "grid A endpoints are 1 and 0")
    check(len(parsed) - 1 == 4, "grid A defines four intervals")

    # 2. invalid grids are rejected
    rejects(lambda: parse_sigmas("1,0.5,0"), "rejects a 3-value grid")
    rejects(lambda: parse_sigmas("1,0.9,0.7,0.3,0.1,0"), "rejects a 6-value grid")
    rejects(lambda: parse_sigmas("0.9,0.8,0.6,0.3,0"), "rejects a grid not starting at 1")
    rejects(lambda: parse_sigmas("1,0.9,0.7,0.3,0.1"), "rejects a grid not ending at 0")
    rejects(lambda: parse_sigmas("1,0.7,0.7,0.3,0"), "rejects a non-strictly-decreasing grid")
    rejects(lambda: parse_sigmas("1,0.3,0.7,0.9,0"), "rejects an increasing grid")

    # 3. active interval selection
    check(active_intervals(4, 0, 3) == [1, 2, 3], "n_max=3, n_min=0 -> [1, 2, 3]")
    check(active_intervals(4, 0, 4) == [0, 1, 2, 3], "n_max=4, n_min=0 -> all intervals")
    check(active_intervals(4, 0, 2) == [2, 3], "n_max=2, n_min=0 -> [2, 3]")
    check(active_intervals(4, 1, 3) == [1, 2], "n_max=3, n_min=1 -> [1, 2]")
    rejects(lambda: active_intervals(4, 0, 5), "rejects n_max=5 on a 4-interval grid")
    rejects(lambda: active_intervals(4, 0, 0), "rejects n_max=0")
    rejects(lambda: active_intervals(4, 3, 3), "rejects n_min == n_max")
    rejects(lambda: active_intervals(4, -1, 3), "rejects n_min < 0")

    # 3b. the stock-model reference configuration: 50 scheduler intervals, FlowEdit over 25..49
    ref = active_intervals(50, 0, 25)
    check(ref == list(range(25, 50)), f"--steps 50 --n_max 25 -> intervals 25..49, got {ref[:3]}..{ref[-1]}")
    check(len(ref) == 25, "25 active intervals")
    check(active_intervals(50, 0, 33)[0] == 17, "the FlowEdit paper's n_max=33 starts at interval 17")
    rejects(lambda: active_intervals(50, 0, 51), "rejects n_max > the interval count")

    # 4. the default active transitions are the three non-pure-noise ones
    trans = [(GRID_A[i], GRID_A[i + 1]) for i in active_intervals(4, 0, 3)]
    check(trans == [(0.882788, 0.693793), (0.693793, 0.337972), (0.337972, 0.0)],
          f"default active transitions: {trans}")

    # 5. forward budget
    b = forward_budget(len(active_intervals(4, 0, 3)), 1, 1.0, 1.0)
    check(b == {"target": 3, "source": 3, "total": 6}, f"default forward budget: {b}")
    check(forward_budget(3, 1, 3.5, 1.0)["total"] == 9, "source CFG adds three forwards")
    check(forward_budget(3, 1, 3.5, 2.0)["total"] == 12, "both CFG branches add six forwards")
    check(forward_budget(3, 2, 1.0, 1.0)["total"] == 12, "n_avg=2 doubles the forwards")

    # 6. the noise bank is a deterministic function of the seed alone
    shape, dev = (1, 16, 8, 8), torch.device("cpu")
    a = build_noise_bank(4, 2, shape, dev, 0)
    b2 = build_noise_bank(4, 2, shape, dev, 0)
    c = build_noise_bank(4, 2, shape, dev, 1)
    check(all(torch.equal(x, y) for xs, ys in zip(a, b2) for x, y in zip(xs, ys)),
          "same seed -> identical noise bank")
    check(not torch.equal(a[0][0], c[0][0]), "a different seed changes the noise bank")
    check(not torch.equal(a[0][0], a[0][1]), "the n_avg draws at one sigma are independent")
    check(not torch.equal(a[0][0], a[1][0]), "different intervals draw different noise")
    check(len(a) == 4 and len(a[0]) == 2 and a[0][0].shape == shape, "noise bank shape")
    # the bank covers every interval, so n_max does not shift the noise of interval i
    check(torch.equal(build_noise_bank(4, 1, shape, dev, 0)[3][0],
                      build_noise_bank(4, 1, shape, dev, 0)[3][0]), "interval noise is n_max-independent")
    # the VAE stream and the noise stream are disjoint for the same CLI seed
    check(VAE_STREAM * SEED_STRIDE != NOISE_STREAM * SEED_STRIDE, "VAE and noise streams are disjoint")

    print(f"[self-test] ok: {n} checks", flush=True)


# ------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="FlowEdit with a 4-step guidance-free SD3.5 student (see docs/CHECKPOINTS.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", help="student .pt (use checkpoint_avg_last5.pt, not checkpoint_final.pt), "
                                         "or 'base' for the stock undistilled SD3.5 transformer")
    ap.add_argument("--source_image", help="image to edit; resized to --height x --width, bicubic")
    ap.add_argument("--source_prompt", help="what the source image shows")
    ap.add_argument("--target_prompt", help="what the edited image should show")
    ap.add_argument("--output_dir", help="written: source.png, edited.png, comparison.png, metadata.json")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium",
                    help="stock pipeline; only the transformer weights are replaced")
    ap.add_argument("--height", type=int, default=TRAINED_RESOLUTION,
                    help="source images are resized to this, not kept at their own resolution")
    ap.add_argument("--width", type=int, default=TRAINED_RESOLUTION,
                    help="the students are trained and evaluated at 512 and score worse elsewhere")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the VAE posterior draw and the FlowEdit noise bank on disjoint streams")
    ap.add_argument("--sigmas", default=GRID_A_STR,
                    help="POST-shift sigma grid, used verbatim: exactly 5 strictly decreasing values from 1 to 0. "
                         "The default is grid A (states 0, 2, 4, 6 of the 8-step training grid)")
    ap.add_argument("--steps", type=int, default=None,
                    help="use the scheduler's own N-interval grid instead of --sigmas. For the stock model "
                         "reference configuration: --checkpoint base --steps 50 --n_max 25 "
                         "--source_guidance 3.5 --target_guidance 13.5. Default: the 4-step grid A")
    ap.add_argument("--n_min", type=int, default=0,
                    help="editing intervals to DROP at the clean end of the schedule")
    ap.add_argument("--n_max", type=int, default=3,
                    help="number of active editing intervals counted from the end of the schedule; 3 skips the "
                         "pure-noise interval at sigma 1.0 on the 4-step grid. With --steps 50, n_max 25 means "
                         "FlowEdit runs over intervals 25..49")
    ap.add_argument("--n_avg", type=int, default=1,
                    help="independent noise draws per interval; their velocity differences are averaged")
    ap.add_argument("--source_guidance", type=float, default=1.0,
                    help="CFG on the source branch; 1.0 skips the unconditional forward entirely")
    ap.add_argument("--target_guidance", type=float, default=1.0,
                    help="CFG on the target branch; the student internalises guidance, so leave it at 1.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self_test", action="store_true",
                    help="run the pure-function checks (no GPU, no model, no checkpoint) and exit")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    required = ["checkpoint", "source_image", "source_prompt", "target_prompt", "output_dir"]
    missing_args = [f"--{a}" for a in required if getattr(args, a) is None]
    if missing_args:
        raise SystemExit(f"[flowedit] missing required arguments: {', '.join(missing_args)}")
    if args.n_avg < 1:
        raise SystemExit(f"[flowedit] --n_avg must be >= 1, got {args.n_avg}")

    # The interval COUNT is known without the scheduler, so n_min/n_max are validated before the
    # (slow) pipeline load; only the sigma VALUES of a --steps grid need the scheduler.
    try:
        if args.steps is not None:
            if args.sigmas != GRID_A_STR:
                raise ValueError("--steps and --sigmas are mutually exclusive")
            if args.steps < 1:
                raise ValueError(f"--steps must be >= 1, got {args.steps}")
            sigma_list, num_intervals = None, args.steps
        else:
            sigma_list = parse_sigmas(args.sigmas)
            num_intervals = len(sigma_list) - 1
        active = active_intervals(num_intervals, args.n_min, args.n_max)
    except ValueError as e:
        raise SystemExit(f"[flowedit] {e}")
    budget = forward_budget(len(active), args.n_avg, args.source_guidance, args.target_guidance)

    if args.height != TRAINED_RESOLUTION or args.width != TRAINED_RESOLUTION:
        print(f"\n{'!' * 88}\n[flowedit] WARNING: {args.height}x{args.width} -- the students are trained and "
              f"evaluated at {TRAINED_RESOLUTION}x{TRAINED_RESOLUTION} and score worse anywhere else.\n"
              f"{'!' * 88}\n", flush=True)
    if args.target_guidance != 1.0 and args.checkpoint != "base":
        print(f"[flowedit] WARNING: --target_guidance {args.target_guidance}: this student was distilled "
              f"against a guided teacher and has the guidance internalised in its conditional prediction. "
              f"1.0 is the operating point; above it is not a recommended default.", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)   # set_device rejects an index-less "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[flowedit] model={args.model_id} device={device} dtype=bfloat16 "
          f"resolution={args.height}x{args.width} seed={args.seed}", flush=True)
    print(f"[flowedit] schedule: {num_intervals} intervals "
          f"({'scheduler grid, --steps ' + str(args.steps) if args.steps else 'grid A'}), active "
          f"{active[0]}..{active[-1]} (n_max={args.n_max}, n_min={args.n_min}, n_avg={args.n_avg})", flush=True)
    cfg_note = ("no unconditional forward" if not need_uncond(args) else
                f"CFG active on {'source' if args.source_guidance > 1.0 else ''}"
                f"{'+' if args.source_guidance > 1.0 and args.target_guidance > 1.0 else ''}"
                f"{'target' if args.target_guidance > 1.0 else ''}")
    print(f"[flowedit] guidance: source={args.source_guidance} target={args.target_guidance} -> {cfg_note}; "
          f"{budget['target']} target + {budget['source']} source = {budget['total']} transformer forwards",
          flush=True)

    pipe = load_pipeline(args.model_id, device)
    if args.checkpoint == "base":
        ck_meta = {"step": None, "variant": "base", "n_tensors": None}
        print("[flowedit] stock SD3.5 transformer, no student checkpoint loaded", flush=True)
    else:
        ck_meta = load_student(pipe, Path(args.checkpoint))
        print(f"[flowedit] checkpoint {Path(args.checkpoint).name}: {ck_meta['n_tensors']} tensors loaded, "
              f"0 missing, 0 unexpected (step={ck_meta['step']}, variant={ck_meta['variant']})", flush=True)

    # Timesteps follow the scheduler convention t = sigma * num_train_timesteps. An explicit grid is
    # used verbatim -- passing it to the pipeline would re-apply the scheduler shift (3.0) to it.
    if sigma_list is None:
        sigmas, timesteps = scheduler_grid(pipe.scheduler, args.steps, device)
        sigma_list = [float(x) for x in sigmas]
    else:
        sigmas = torch.tensor(sigma_list, device=device, dtype=torch.float32)
        timesteps = sigmas[:-1] * float(pipe.scheduler.config.num_train_timesteps)
    show = (lambda v: v if len(v) <= 9 else v[:4] + ["..."] + v[-4:])
    print(f"[flowedit] sigmas    = {show([round(x, 6) for x in sigma_list])}", flush=True)
    print(f"[flowedit] timesteps = {show([round(float(t), 3) for t in timesteps])}", flush=True)
    print(f"[flowedit] active transitions ({len(active)}): "
          f"{sigma_list[active[0]]:.6f} -> ... -> {sigma_list[active[-1] + 1]:.6f}", flush=True)
    for i in (active if len(active) <= 6 else active[:3] + active[-3:]):
        print(f"[flowedit]   [{i}] {sigma_list[i]:.6f} -> {sigma_list[i + 1]:.6f}", flush=True)

    t0 = time.time()
    source_image = load_source_image(Path(args.source_image), args.height, args.width)
    z_src = encode_source_latent(pipe, source_image, device, args.seed)
    expected_c = pipe.transformer.config.in_channels
    expected_h = args.height // pipe.vae_scale_factor
    assert z_src.shape == (1, expected_c, expected_h, args.width // pipe.vae_scale_factor), z_src.shape

    with torch.no_grad():
        src_emb, src_pooled = encode_prompt(pipe, args.source_prompt, device)
        tgt_emb, tgt_pooled = encode_prompt(pipe, args.target_prompt, device)
        # The unconditional embeddings are only needed by the optional CFG branches: at the default
        # guidance of 1.0 there is no empty prompt anywhere in this script.
        neg_emb, neg_pooled = encode_prompt(pipe, "", device) if need_uncond(args) else (None, None)

    # The whole noise bank is drawn before any transformer forward, so two checkpoints compared at
    # the same seed see byte-identical noise whatever they do with it.
    noise_bank = build_noise_bank(num_intervals, args.n_avg, tuple(z_src.shape), device, args.seed)

    z, counts = flowedit(pipe, z_src, sigmas, timesteps, active, noise_bank,
                         src_emb, src_pooled, tgt_emb, tgt_pooled, neg_emb, neg_pooled,
                         args.source_guidance, args.target_guidance, device)
    assert counts == budget, (counts, budget)

    with torch.no_grad():
        edited = (vae_decode(pipe.vae, z) + 1.0) / 2.0          # [-1, 1] -> [0, 1]
    edited = edited.clamp(0.0, 1.0).cpu()
    assert torch.isfinite(edited).all(), "the edited image is not finite"
    runtime = time.time() - t0

    from torchvision.utils import save_image
    source_01 = ((source_image + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
    save_image(source_01[0], out_dir / "source.png")            # the preprocessed source, as encoded
    save_image(edited[0], out_dir / "edited.png")
    save_image(torch.cat([source_01, edited], dim=0), out_dir / "comparison.png", nrow=2, padding=4)

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": ck_meta["step"],
        "checkpoint_variant": ck_meta["variant"],
        "checkpoint_n_tensors": ck_meta["n_tensors"],
        "model_id": args.model_id,
        "source_image": str(Path(args.source_image).resolve()),
        "source_prompt": args.source_prompt,
        "target_prompt": args.target_prompt,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "vae_seed": args.seed + VAE_STREAM * SEED_STRIDE,
        "noise_seed": args.seed + NOISE_STREAM * SEED_STRIDE,
        "steps": args.steps,
        "grid": "scheduler" if args.steps is not None else "explicit",
        "sigmas": sigma_list,
        "timesteps": [float(t) for t in timesteps],
        "active_indices": active,
        "active_transitions": [[sigma_list[i], sigma_list[i + 1]] for i in active],
        "n_min": args.n_min,
        "n_max": args.n_max,
        "n_avg": args.n_avg,
        "source_guidance": args.source_guidance,
        "target_guidance": args.target_guidance,
        "dtype": "bfloat16",
        "device": str(device),
        "runtime_seconds": round(runtime, 3),
        "transformer_forwards": counts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[flowedit] {counts['total']} transformer forwards, {runtime:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
