#!/usr/bin/env python3
"""Gradient diagnostic for soft candidate weighting (report Section 10, reviewer step 1).

For held-out captions (118k cache minus the 3k pool) and a FROZEN model state, roll out all four
cached candidates of a caption with the frozen teacher and compute each candidate's consistency
gradient g_j (same Delta draws for every candidate). With the Boltzmann weights pi_j at --temp:

    g_F  = sum_j pi_j g_j                          exact (Full-4) gradient
    V_c  = sum_j pi_j ||g_j - g_F||^2              candidate-sampling variance of Cat-1
         = sum_j pi_j ||g_j||^2 - ||g_F||^2
    B_c  = || sum_j pi_j C(g_j) - C(g_F) ||        clipping-induced difference, C(g) = g min(1, gamma/||g||)

Everything is a function of the 4x4 Gram matrix of the candidate gradients and pi, so only dot
products are formed; gradients live on the CPU one at a time. Two references per caption:
    Delta-noise   : the argmax candidate's gradient under a second Delta draw (same trajectory)
    batch-shape   : the argmax candidate rolled out alone instead of in the 4-candidate batch
Also logged: candidate gradient norms and the fraction above the clip threshold, pairwise
cosines, cos(g_j, g_F), the Adam second-moment ratio sum_j pi_j ||g_j||^2 / ||g_F||^2, and the
weights' entropy ESS, Kish ESS and per-visit switching probability 1 - sum_j pi_j^2.

    python3 phaseW/grad_diag.py --ckpt base --out phaseW/grad_diag_base.json
    python3 phaseW/grad_diag.py --ckpt checkpoints/phaseS4/phaseS4_CD_dinop_hard_s0_130455/checkpoint_final.pt ...
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def rollout(teacher, scheduler, z0, emb, pooled, neg_emb, neg_pool, K, cfg, device):
    """Frozen-teacher CFG Euler rollout, identical to the trainer's rollout_states."""
    N = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sigmas = scheduler.sigmas.to(device, torch.float32)
    ts = scheduler.timesteps.to(device)
    e = torch.cat([neg_emb.repeat(N, 1, 1), emb.repeat(N, 1, 1)], 0)
    p = torch.cat([neg_pool.repeat(N, 1), pooled.repeat(N, 1)], 0)
    z = z0
    zs = [z]
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v_all = teacher(hidden_states=torch.cat([z, z], 0), timestep=ts[k].expand(2 * N),
                            encoder_hidden_states=e, pooled_projections=p, return_dict=False)[0]
        v_u, v_c = v_all.chunk(2, 0)
        v = v_u + cfg * (v_c - v_u)
        z = (z.float() + (sigmas[k + 1] - sigmas[k]) * v.float()).to(torch.bfloat16)
        zs.append(z)
    return zs, sigmas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student checkpoint or 'base' (the pretrained init)")
    ap.add_argument("--cache", default="phaseN/coco_selection_118k")
    ap.add_argument("--exclude", default="phaseN/coco_selection_dinopatch", help="cache whose prompts are excluded")
    ap.add_argument("--n_captions", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.04)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--window", default="0.4,0.9")
    ap.add_argument("--delta_min", type=int, default=1)
    ap.add_argument("--delta_max", type=int, default=3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    # held-out captions: 118k records whose prompt is not in the 3k pool
    excl = set()
    for f in sorted(Path(args.exclude).glob("selection_rank*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                excl.add(json.loads(ln)["prompt"])
    recs = []
    for f in sorted(Path(args.cache).glob("selection_rank*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r["prompt"] not in excl and "dino_patch_cos" in r:
                    recs.append(r)
    recs.sort(key=lambda r: r["idx"])
    rng = np.random.default_rng(args.seed)
    recs = [recs[i] for i in rng.choice(len(recs), size=args.n_captions, replace=False)]
    print(f"[diag] {len(recs)} held-out captions ({len(excl)} excluded prompts) | ckpt={args.ckpt} T={args.temp}", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    student = pipe.transformer
    teacher = copy.deepcopy(student).to(dtype=torch.bfloat16).eval()      # the frozen pretrained teacher
    for p in teacher.parameters():
        p.requires_grad = False
    student.to(dtype=torch.float32).train()
    student.enable_gradient_checkpointing()
    if args.ckpt != "base":
        ck = torch.load(args.ckpt, map_location="cpu", mmap=False, weights_only=False)
        missing, unexpected = student.load_state_dict(ck["model"], strict=True), None
        print(f"[diag] loaded student from {args.ckpt} (step {ck.get('step')})", flush=True)
    params = [p for p in student.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)

    lat_c = student.config.in_channels
    H = args.height // pipe.vae_scale_factor
    huber_c = 0.00054 * (lat_c * H * H) ** 0.5
    K = args.K
    lo, hi = (float(x) for x in args.window.split(","))
    score_idxs = list(range(max(1, round(lo * K)), min(K - 1, round(hi * K)) + 1))
    n_w = len(score_idxs)
    delta_rng = random.Random(args.seed + 17)
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
                                                     do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)

    def grad_of(z, deltas, emb, pooled, sigmas):
        """consistency loss of trajectory z (list of [1,C,H,W]) and its gradient (CPU fp32 tensors)."""
        stu_idx = [k - deltas[k] for k in score_idxs]
        z_in = torch.cat([z[s] for s in stu_idx], 0).to(torch.bfloat16)
        t_in = torch.cat([pipe.scheduler.timesteps.to(device)[s].reshape(1) for s in stu_idx], 0)
        for p in params:
            p.grad = None
        with torch.autocast("cuda", torch.bfloat16):
            v = student(hidden_states=z_in, timestep=t_in, encoder_hidden_states=emb.repeat(n_w, 1, 1),
                        pooled_projections=pooled.repeat(n_w, 1), return_dict=False)[0]
        sig = torch.stack([sigmas[s] for s in stu_idx]).view(-1, 1, 1, 1)
        x_hat = torch.cat([z[s] for s in stu_idx], 0) - sig * v.float()
        x_tea = torch.cat([z[k] - sigmas[k] * (z[k + 1] - z[k]) / (sigmas[k + 1] - sigmas[k]) for k in score_idxs], 0).detach()
        d = torch.sqrt((x_hat - x_tea).pow(2).sum(dim=(1, 2, 3)) + huber_c * huber_c) - huber_c
        loss = d.mean()
        loss.backward()
        sq = 0.0
        for p in params:
            sq += float(p.grad.detach().pow(2).sum(dtype=torch.float64))
        g = [p.grad.detach().to("cpu", copy=True) for p in params]
        for p in params:
            p.grad = None
        return float(loss.detach()), g, sq ** 0.5

    def dot(a, b):
        return float(sum(torch.dot(x.flatten(), y.flatten()).double() for x, y in zip(a, b)))

    out = {"ckpt": args.ckpt, "temp": args.temp, "grad_clip": args.grad_clip, "n_params": n_par,
           "captions": []}
    t0 = time.time()
    for ci, rec in enumerate(recs):
        S = np.asarray(rec["dino_patch_cos"], float)
        n = len(S)
        z = (S - S.max()) / args.temp
        pi = np.exp(z); pi /= pi.sum()
        jstar = int(S.argmax())
        with torch.no_grad():
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[rec["prompt"]], prompt_2=[rec["prompt"]], prompt_3=[rec["prompt"]],
                                                   do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            z0_all = torch.cat([torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16,
                                            generator=torch.Generator(device=device).manual_seed(rec["seed_base"] + j))
                                for j in range(n)], 0)
            zs_all, sigmas = rollout(teacher, pipe.scheduler, z0_all, emb, pooled, neg_emb, neg_pool, K, args.cfg, device)
            zs_single, _ = rollout(teacher, pipe.scheduler, z0_all[jstar:jstar + 1], emb, pooled, neg_emb, neg_pool, K, args.cfg, device)
            batch_shape_maxdiff = max(float((zs_all[k][jstar:jstar + 1].float() - zs_single[k].float()).abs().max()) for k in range(K + 1))
            batch_shape_reldiff = float((zs_all[K][jstar:jstar + 1].float() - zs_single[K].float()).norm() / zs_single[K].float().norm())
        d1 = {k: delta_rng.randint(args.delta_min, args.delta_max) for k in score_idxs}
        d2 = {k: delta_rng.randint(args.delta_min, args.delta_max) for k in score_idxs}
        losses, grads, norms = [], [], []
        for j in range(n):
            lj, gj, nj = grad_of([zs_all[k][j:j + 1].float() for k in range(K + 1)], d1, emb, pooled, sigmas)
            losses.append(lj); grads.append(gj); norms.append(nj)
        l_s, g_s, n_s = grad_of([zs_single[k].float() for k in range(K + 1)], d1, emb, pooled, sigmas)   # batch-shape reference
        l_d, g_d, n_d = grad_of([zs_all[k][jstar:jstar + 1].float() for k in range(K + 1)], d2, emb, pooled, sigmas)  # Delta reference
        G = np.zeros((n, n))
        for i in range(n):
            G[i, i] = norms[i] ** 2
            for j in range(i + 1, n):
                G[i, j] = G[j, i] = dot(grads[i], grads[j])
        gs_dot = dot(grads[jstar], g_s); gd_dot = dot(grads[jstar], g_d)
        del g_s, g_d
        # exact and sampled estimators from the Gram matrix
        gF_sq = float(pi @ G @ pi); gF = gF_sq ** 0.5
        Gpi = G @ pi
        V_c = float(sum(pi[j] * G[j, j] for j in range(n)) - gF_sq)
        gamma = args.grad_clip
        c = np.array([min(1.0, gamma / norms[j]) for j in range(n)]); c_F = min(1.0, gamma / gF)
        a = pi * (c - c_F)
        B_c = float(np.sqrt(max(a @ G @ a, 0.0)))
        b = pi * c                                           # E[C(g_J)] = sum_j pi_j c_j g_j
        Eclip_sq = float(b @ G @ b)
        cos_clip = float((b @ Gpi) * c_F / (np.sqrt(Eclip_sq) * c_F * gF))   # cos(E[C(g_J)], C(g_F))
        cos_jF = [float(Gpi[j] / (norms[j] * gF)) for j in range(n)]
        cosmat = [[float(G[i, j] / (norms[i] * norms[j])) for j in range(n)] for i in range(n)]
        offdiag = [cosmat[i][j] for i in range(n) for j in range(n) if i < j]
        rec_out = {
            "idx": int(rec["idx"]), "prompt": rec["prompt"], "scores": S.tolist(), "pi": pi.tolist(), "argmax": jstar,
            "ess_entropy": float(np.exp(-(pi * np.log(pi)).sum())), "ess_kish": float(1.0 / (pi ** 2).sum()),
            "p_switch": float(1.0 - (pi ** 2).sum()),
            "losses": losses, "grad_norms": norms, "frac_clipped": float(np.mean([nrm > gamma for nrm in norms])),
            "gF_norm": gF, "V_c": V_c, "V_c_rel": V_c / gF_sq, "adam_ratio": float(sum(pi[j] * G[j, j] for j in range(n)) / gF_sq),
            "B_c": B_c, "B_c_rel": B_c / (c_F * gF), "cos_Eclip_vs_clipF": cos_clip,
            "Eclip_norm_rel": float(np.sqrt(Eclip_sq) / (c_F * gF)),
            "cos_j_F": cos_jF, "cos_argmax_F": cos_jF[jstar], "cos_pairwise": cosmat, "cos_pairwise_mean": float(np.mean(offdiag)),
            "delta_ref": {"cos": float(gd_dot / (norms[jstar] * n_d)), "rel_sq_diff": float((norms[jstar] ** 2 + n_d ** 2 - 2 * gd_dot) / norms[jstar] ** 2),
                          "loss": l_d, "norm": n_d},
            "batch_shape": {"max_abs_state_diff": batch_shape_maxdiff, "rel_endpoint_diff": batch_shape_reldiff,
                            "cos_grad": float(gs_dot / (norms[jstar] * n_s)), "rel_loss_diff": float(abs(l_s - losses[jstar]) / max(abs(losses[jstar]), 1e-9)),
                            "rel_norm_diff": float(abs(n_s - norms[jstar]) / norms[jstar])},
        }
        out["captions"].append(rec_out)
        print(f"[diag] {ci+1}/{len(recs)} idx={rec['idx']} |g|={np.mean(norms):.1f} |gF|={gF:.1f} "
              f"Vc/|gF|^2={rec_out['V_c_rel']:.3f} Bc/|C(gF)|={rec_out['B_c_rel']:.3f} cos(EC,CF)={cos_clip:.4f} "
              f"cos_pair={rec_out['cos_pairwise_mean']:.3f} cos(argmax,F)={cos_jF[jstar]:.3f} "
              f"delta_ref cos={rec_out['delta_ref']['cos']:.3f} batch cos={rec_out['batch_shape']['cos_grad']:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del grads
    keys = ["ess_entropy", "ess_kish", "p_switch", "frac_clipped", "gF_norm", "V_c_rel", "adam_ratio", "B_c_rel",
            "cos_Eclip_vs_clipF", "Eclip_norm_rel", "cos_argmax_F", "cos_pairwise_mean"]
    summ = {k: {"mean": float(np.mean([c[k] for c in out["captions"]])), "median": float(np.median([c[k] for c in out["captions"]]))} for k in keys}
    summ["grad_norm_mean"] = float(np.mean([np.mean(c["grad_norms"]) for c in out["captions"]]))
    summ["delta_ref_cos"] = float(np.mean([c["delta_ref"]["cos"] for c in out["captions"]]))
    summ["delta_ref_rel_sq_diff"] = float(np.mean([c["delta_ref"]["rel_sq_diff"] for c in out["captions"]]))
    summ["batch_shape_cos_grad"] = float(np.mean([c["batch_shape"]["cos_grad"] for c in out["captions"]]))
    summ["batch_shape_max_abs_state_diff"] = float(np.max([c["batch_shape"]["max_abs_state_diff"] for c in out["captions"]]))
    summ["batch_shape_rel_loss_diff"] = float(np.mean([c["batch_shape"]["rel_loss_diff"] for c in out["captions"]]))
    out["summary"] = summ
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("[diag] SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
