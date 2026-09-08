#!/usr/bin/env python3
"""End-to-end pairing check for the 118k cache against the TRAINER's own code path.

The cache (phaseN/build_coco_selection.py) scores four candidates per caption and records which one
wins. The trainer (phaseC/train_pilot.py) never reads those candidates: it RE-ROLLS the winner from
`seed_base + sel` with its own `rollout_states`. If the two disagree -- a different sampler, a
different noise convention, a different scheduler, a different decode -- the student is trained on
a trajectory the scorer never saw, and nothing crashes. That exact failure occurred once in this
project (the caption->candidate map agreed at chance), so it is checked here directly rather than
inferred from the two functions "looking the same".

What this does, for records sampled from several rank files:
  1. encode the prompt exactly as train_pilot does;
  2. draw z0 with the trainer's convention  manual_seed(seed_base + j), randn(1,C,H,W, bf16);
  3. roll out with train_pilot.rollout_states (imported, not copied), K=8, cfg 7;
  4. decode with train_self_distill.vae_decode, quantise like torchvision.save_image;
  5. score mean-pooled DINOv2-base patches and CLS against the reference photo;
  6. compare with the cache's `dino_patch_cos` / `dino_cos`.
A NEGATIVE control scores each record against the NEXT record's cached values, so the test is
shown to be able to fail (agreement should fall to ~1/N).

Pass criterion: max|cos diff| below --tol on both channels, and 100% argmax agreement among records
whose cached top-2 margin exceeds 3x the measured drift. The builder samples the 4 candidates in one
batch while the trainer rolls one, so bf16 kernel tiling moves a cosine by ~1e-3; a hard argmax over
two candidates closer than that is decided by rounding and may flip without any pairing fault (a
real fault scrambles all four values, which is what the shifted control demonstrates). Near-ties are
counted and reported, not scored. Measured 2026-09-02 (job 128552): drift 2.2e-3 / 3.2e-3 max,
10/10 and 9/9 clear records agree, shifted control 8.3%.
"""
from __future__ import annotations

import argparse, glob, json, os, random, re, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "phaseC"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="phaseN/coco_selection_118k")
    ap.add_argument("--ranks", default="0,9,18,27")
    ap.add_argument("--per_rank", type=int, default=3)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default="phaseW/verify_cache_pairing.json")
    args = ap.parse_args()
    device = torch.device("cuda")

    from train_pilot import rollout_states                     # the trainer's function itself
    from train_self_distill import vae_decode                  # the trainer's decode

    recs = []
    for r in (int(x) for x in args.ranks.split(",")):
        f = Path(args.cache) / f"selection_rank{r}.jsonl"
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rng = random.Random(1234 + r)
        recs += [dict(rec, _rank=r) for rec in rng.sample(rows, min(args.per_rank, len(rows)))]
    print(f"[pair] {len(recs)} records from ranks {args.ranks}", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
    pipe.set_progress_bar_config(disable=True)
    assert abs(float(pipe.scheduler.config.shift) - 3.0) < 1e-6, pipe.scheduler.config.shift
    teacher = pipe.transformer
    lat_c = teacher.config.in_channels
    h_lat = args.height // pipe.vae_scale_factor

    from transformers import AutoImageProcessor, AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    @torch.no_grad()
    def dino_scores(ref_img, cand_imgs):
        px = proc(images=[ref_img] + list(cand_imgs), return_tensors="pt")["pixel_values"].to(device)
        h = dino(pixel_values=px).last_hidden_state
        cls = F.normalize(h[:, 0].float(), dim=-1)
        pat = F.normalize(h[:, 1:].float().mean(1), dim=-1)
        return ((cls[0:1] * cls[1:]).sum(-1).cpu().numpy(), (pat[0:1] * pat[1:]).sum(-1).cpu().numpy())

    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
                                                     do_classifier_free_guidance=False, device=device,
                                                     num_images_per_prompt=1)

    rows = []
    for rec in recs:
        prompt, seed_base = rec["prompt"], int(rec["seed_base"])
        N = int(rec["N"])
        with torch.no_grad():
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                                                   do_classifier_free_guidance=False, device=device,
                                                   num_images_per_prompt=1)
            cands = []
            for j in range(N):
                # EXACTLY train_pilot's draw
                g = torch.Generator(device=device).manual_seed(seed_base + j)
                z0 = torch.randn(1, lat_c, h_lat, h_lat, device=device, dtype=torch.bfloat16, generator=g)
                zs, _ = rollout_states(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool,
                                       args.K, args.cfg, device)
                img = vae_decode(pipe.vae, zs[args.K])                        # [-1,1] float
                u8 = ((img + 1) / 2).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)  # save_image
                cands.append(Image.fromarray(u8[0].permute(1, 2, 0).cpu().numpy()))
        ref = Image.open(rec["reference"]).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
        cls, pat = dino_scores(ref, cands)
        rows.append(dict(idx=int(rec["idx"]), rank=rec["_rank"], seed_base=seed_base,
                         cached_pat=rec["dino_patch_cos"], mine_pat=pat.tolist(),
                         cached_cls=rec["dino_cos"], mine_cls=cls.tolist()))
        print(f"[pair] idx={rec['idx']:6d} r{rec['_rank']:2d} patch cached={np.round(rec['dino_patch_cos'],4).tolist()} "
              f"mine={np.round(pat,4).tolist()}", flush=True)

    def agree(a, b): return int(np.argmax(a) == np.argmax(b))
    res = {}
    for ch in ("pat", "cls"):
        c = np.array([r[f"cached_{ch}"] for r in rows]); m = np.array([r[f"mine_{ch}"] for r in rows])
        res[ch] = dict(max_abs_diff=float(np.abs(c - m).max()), mean_abs_diff=float(np.abs(c - m).mean()),
                       argmax_agree=float(np.mean([agree(c[k], m[k]) for k in range(len(rows))])),
                       # negative control: my scores vs the NEXT record's cached scores
                       shifted_argmax_agree=float(np.mean([agree(np.roll(c, 1, 0)[k], m[k]) for k in range(len(rows))])))
    # A hard argmax is undefined to within the numerical drift when two candidates are within
    # a few 1e-3 of each other (the builder samples 4 candidates in one bf16 batch, the trainer
    # rolls one). Near-ties therefore may flip WITHOUT any pairing fault, and a pairing fault
    # would scramble all four values (see the shifted control), not swap two close ones. The
    # criterion is: every record whose cached top-2 margin exceeds the noise floor must agree.
    floor = 3.0 * max(res["pat"]["max_abs_diff"], res["cls"]["max_abs_diff"])
    def clear(ch):
        c = np.array([r[f"cached_{ch}"] for r in rows]); m = np.array([r[f"mine_{ch}"] for r in rows])
        marg = np.sort(c, 1)[:, -1] - np.sort(c, 1)[:, -2]
        idx = np.where(marg > floor)[0]
        return float(np.mean([agree(c[k], m[k]) for k in idx])) if len(idx) else 1.0, int(len(rows) - len(idx))
    for ch in ("pat", "cls"):
        res[ch]["argmax_agree_clear"], res[ch]["near_ties"] = clear(ch)
    res["noise_floor"] = floor
    ok = (res["pat"]["argmax_agree_clear"] == 1.0 and res["cls"]["argmax_agree_clear"] == 1.0
          and res["pat"]["max_abs_diff"] < args.tol and res["cls"]["max_abs_diff"] < args.tol)
    print("\n  channel  max|diff|  mean|diff|  argmax agree  shifted-control agree")
    for ch in ("pat", "cls"):
        r = res[ch]
        print(f"  {ch:7s}  {r['max_abs_diff']:.2e}   {r['mean_abs_diff']:.2e}   {100*r['argmax_agree']:6.1f}%        "
              f"{100*r['shifted_argmax_agree']:6.1f}%  (chance {100/4:.0f}%)")
    print(f"  near-ties (cached top-2 margin < {floor:.1e}): pat {res['pat']['near_ties']}, cls {res['cls']['near_ties']}; "
          f"agreement among clear records: pat {100*res['pat']['argmax_agree_clear']:.0f}%, cls {100*res['cls']['argmax_agree_clear']:.0f}%")
    print(f"\nPAIRING_RESULT {'PASS' if ok else 'FAIL'}  n={len(rows)} tol={args.tol}")
    json.dump(dict(result="PASS" if ok else "FAIL", summary=res, rows=rows), open(args.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
