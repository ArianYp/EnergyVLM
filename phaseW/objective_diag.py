#!/usr/bin/env python3
"""How good are the consistency targets the student is trained on?

For a sample of cached captions, re-roll the selected candidate's K=8 guided teacher trajectory
(exactly as the trainer does) and compare, at every supervised state k, the targets the trainer
COULD use against the trajectory's true endpoint z_K:

    tweedie1(k)   z_k - sigma_k (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k)     what we train on
    tweedie2(k)   z_k - sigma_k (z_{k+2} - z_k) / (sigma_{k+2} - sigma_k)     2-step (student grid) chord
    endpoint      z_K                                                        the final clean latent

Reports per k: relative L2 error to z_K and cosine with z_K (latent space), for both Tweedie
horizons. Then the "target conflict": a student state s = k - Delta can be assigned targets from
several k (Delta in {1,2,3}); the spread of those targets, relative to ||z_K||, is what the
student is asked to average over.
"""
from __future__ import annotations

import argparse, glob, json, random, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "phaseC"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="phaseN/coco_selection_dinopatch")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--field", default="dino_patch_cos")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", default="phaseW/objective_diag.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    from train_pilot import rollout_states
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
    teacher = pipe.transformer
    lat_c, h_lat = teacher.config.in_channels, args.height // pipe.vae_scale_factor
    K = args.K

    recs = []
    for f in sorted(glob.glob(str(Path(args.records) / "selection_rank*.jsonl"))):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    recs = random.Random(0).sample(recs, min(args.n, len(recs)))

    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
                                                     do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    rel1, rel2, cos1, cos2 = {k: [] for k in range(1, K)}, {k: [] for k in range(1, K - 1)}, {k: [] for k in range(1, K)}, {k: [] for k in range(1, K - 1)}
    conflict = {s: [] for s in range(0, K - 1)}
    win = list(range(max(1, round(0.4 * K)), min(K - 1, round(0.9 * K)) + 1))
    for n, rec in enumerate(recs):
        sel = int(np.argmax(rec[args.field]))
        with torch.no_grad():
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[rec["prompt"]], prompt_2=[rec["prompt"]], prompt_3=[rec["prompt"]],
                                                   do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            g = torch.Generator(device=device).manual_seed(int(rec["seed_base"]) + sel)
            z0 = torch.randn(1, lat_c, h_lat, h_lat, device=device, dtype=torch.bfloat16, generator=g)
            zs, sig = rollout_states(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, K, args.cfg, device)
        z = [s.float().flatten() for s in zs]
        zK = z[K]; nK = zK.norm()
        t1 = {}
        for k in range(1, K):
            x1 = z[k] - sig[k] * (z[k + 1] - z[k]) / (sig[k + 1] - sig[k]); t1[k] = x1
            rel1[k].append(float((x1 - zK).norm() / nK)); cos1[k].append(float(torch.dot(x1, zK) / (x1.norm() * nK)))
            if k + 2 <= K:
                x2 = z[k] - sig[k] * (z[k + 2] - z[k]) / (sig[k + 2] - sig[k])
                rel2[k].append(float((x2 - zK).norm() / nK)); cos2[k].append(float(torch.dot(x2, zK) / (x2.norm() * nK)))
        for s in range(0, K - 1):
            ks = [k for k in win if 1 <= k - s <= 3]          # which supervised k can land the student on state s
            if len(ks) >= 2:
                T = torch.stack([t1[k] for k in ks])
                d = torch.cdist(T, T)
                conflict[s].append(float(d[np.triu_indices(len(ks), 1)].mean() / nK))
        if (n + 1) % 64 == 0:
            print(f"[diag] {n+1}/{len(recs)}", flush=True)

    sigmas = sig.tolist()
    print(f"\n  supervised window k = {win}; sigma grid = {[round(s, 3) for s in sigmas]}")
    print(f"\n  {'k':>3s} {'sigma_k':>8s} {'rel err tweedie1':>17s} {'rel err tweedie2':>17s} {'cos tweedie1':>13s} {'cos tweedie2':>13s}")
    out = {"window": win, "sigmas": sigmas, "per_k": {}}
    for k in range(1, K):
        r2 = f"{np.mean(rel2[k]):17.4f}" if k in rel2 and rel2[k] else f"{'-':>17s}"
        c2 = f"{np.mean(cos2[k]):13.4f}" if k in cos2 and cos2[k] else f"{'-':>13s}"
        print(f"  {k:3d} {sigmas[k]:8.3f} {np.mean(rel1[k]):17.4f} {r2} {np.mean(cos1[k]):13.4f} {c2}")
        out["per_k"][k] = {"sigma": sigmas[k], "rel_err_tweedie1": float(np.mean(rel1[k])), "cos_tweedie1": float(np.mean(cos1[k])),
                           "rel_err_tweedie2": float(np.mean(rel2[k])) if rel2.get(k) else None,
                           "cos_tweedie2": float(np.mean(cos2[k])) if cos2.get(k) else None}
    print(f"\n  target conflict at student state s (mean pairwise ||x0(k_i) - x0(k_j)|| / ||z_K|| over the k that can supervise s):")
    out["conflict"] = {}
    for s in range(0, K - 1):
        if conflict[s]:
            out["conflict"][s] = float(np.mean(conflict[s]))
            print(f"    s={s} sigma={sigmas[s]:.3f}: {np.mean(conflict[s]):.4f}   (n={len(conflict[s])})")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
