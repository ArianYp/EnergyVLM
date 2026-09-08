#!/usr/bin/env python3
"""Invariance violation (x0-prediction drift) along each cached candidate's teacher trajectory.

For every caption in the cache and each of its N candidates, re-roll the K-step guided teacher
trajectory from its seed (exactly as the trainer does) and form the Tweedie clean-latent estimate
at every state, x0(k) = z_k - sigma_k (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k). Along an ideal
(straight) characteristic x0(k) would be constant; its movement,

    drift_total  = sum_k ||x0(k+1) - x0(k)|| / ||z_K||          over the whole trajectory
    drift_window = the same over the supervised states k = 3..6 (targets the student regresses)
    early_error  = ||x0(3) - z_K|| / ||z_K||                    how wrong the first supervised target is

measures how far that candidate's trajectory passed from a crossing of characteristics, i.e. how
curved -- and how hard to compress into 4 steps -- it is.

Two questions:
  Q1  do the DINO-selected candidates have lower drift than random / VQAScore-selected ones?
      (paired per caption: selected candidate's drift minus the caption's mean over candidates)
  Q2  does drift alone select well? argmin-drift's % of oracle VQAScore headroom, within-prompt
      rank correlation of -drift with VQAScore and with dino_patch_cos, and a blend with DINO.
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "phaseC"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="phaseN/coco_selection_dinopatch")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="phaseW/drift_selector.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    from train_pilot import rollout_states
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
    teacher = pipe.transformer
    lat_c, h_lat, K = teacher.config.in_channels, args.height // pipe.vae_scale_factor, args.K
    win = list(range(max(1, round(0.4 * K)), min(K - 1, round(0.9 * K)) + 1))      # [3..7]

    recs = []
    for f in sorted(glob.glob(str(Path(args.records) / "selection_rank*.jsonl"))):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    recs.sort(key=lambda r: r["idx"])
    if args.limit: recs = recs[: args.limit]
    N = int(recs[0]["N"])
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
                                                     do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    DT, DW, EE, PK = (np.zeros((len(recs), N)) for _ in range(4))
    for n, rec in enumerate(recs):
        with torch.no_grad():
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[rec["prompt"]], prompt_2=[rec["prompt"]], prompt_3=[rec["prompt"]],
                                                   do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            z0 = torch.cat([torch.randn(1, lat_c, h_lat, h_lat, device=device, dtype=torch.bfloat16,
                                        generator=torch.Generator(device=device).manual_seed(int(rec["seed_base"]) + j))
                            for j in range(N)], 0)
            zs, sig = rollout_states(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, K, args.cfg, device)
        z = [s.float().flatten(1) for s in zs]                                   # [N, D] each
        x0 = [z[k] - sig[k] * (z[k + 1] - z[k]) / (sig[k + 1] - sig[k]) for k in range(K)]
        nK = z[K].norm(dim=1)
        step = torch.stack([(x0[k + 1] - x0[k]).norm(dim=1) for k in range(K - 1)], 1) / nK[:, None]   # [N, K-1]
        DT[n] = step.sum(1).cpu().numpy()
        DW[n] = step[:, win[0]:win[-1]].sum(1).cpu().numpy()
        PK[n] = step.max(1).values.cpu().numpy()
        EE[n] = ((x0[win[0]] - z[K]).norm(dim=1) / nK).cpu().numpy()
        if (n + 1) % 250 == 0:
            print(f"[drift] {n+1}/{len(recs)}", flush=True)

    VQ = np.array([r["endpoint_vqa"] for r in recs]); DP = np.array([r["dino_patch_cos"] for r in recs])
    rnd = np.array([r["random_idx"] for r in recs]); sel_dp = DP.argmax(1); orac = VQ.argmax(1)
    P = len(recs); rows = np.arange(P)
    v_rand = VQ[rows, rnd].mean(); v_orac = VQ.max(1).mean()
    hr = lambda pick: 100 * (VQ[rows, pick].mean() - v_rand) / (v_orac - v_rand)
    out = {"n": P, "N": N, "window": win}

    print(f"\n{P} captions x {N} candidates; drift relative to ||z_K||; supervised window k={win}")
    print(f"  mean over all candidates: drift_total {DT.mean():.4f}  drift_window {DW.mean():.4f}  peak step {PK.mean():.4f}  early_error {EE.mean():.4f}")
    print("\nQ1. Is the selected candidate's trajectory straighter than the caption's average candidate?")
    print(f"  {'selector':22s} {'drift_total':>12s} {'vs cand mean':>13s} {'p (t / wilcoxon)':>20s} {'drift_window':>13s} {'early_error':>12s}")
    out["q1"] = {}
    for name, pick in (("random (B2)", rnd), ("DINO-patch (ours)", sel_dp), ("VQAScore oracle", orac)):
        d = DT[rows, pick] - DT.mean(1)
        tp, wp = ttest_rel(DT[rows, pick], DT.mean(1))[1], wilcoxon(d)[1]
        print(f"  {name:22s} {DT[rows, pick].mean():12.4f} {d.mean():+13.4f} {tp:9.1e} / {wp:8.1e} {DW[rows, pick].mean():13.4f} {EE[rows, pick].mean():12.4f}")
        out["q1"][name] = {"drift_total": float(DT[rows, pick].mean()), "delta_vs_mean": float(d.mean()), "t_p": float(tp), "wilcoxon_p": float(wp),
                           "drift_window": float(DW[rows, pick].mean()), "early_error": float(EE[rows, pick].mean())}
    rho_dp = np.nanmean([spearmanr(-DT[k], DP[k])[0] for k in range(P)])
    rho_vq = np.nanmean([spearmanr(-DT[k], VQ[k])[0] for k in range(P)])
    print(f"  within-prompt Spearman of -drift_total with dino_patch_cos: {rho_dp:+.4f}; with VQAScore: {rho_vq:+.4f}")
    out["q1"]["rho_negdrift_dinopatch"] = float(rho_dp); out["q1"]["rho_negdrift_vqa"] = float(rho_vq)

    print("\nQ2. Drift as a selector: % of oracle VQAScore headroom recovered (random = 0, oracle = 100)")
    print(f"  {'selector':38s} {'headroom':>9s} {'top1=VQA':>9s}")
    out["q2"] = {}
    def z(x): s = x.std(1, keepdims=True); return (x - x.mean(1, keepdims=True)) / np.where(s > 0, s, 1)
    cands = [("dino_patch_cos (reference)", DP.argmax(1)), ("argmin drift_total", DT.argmin(1)), ("argmin drift_window", DW.argmin(1)),
             ("argmin peak step", PK.argmin(1)), ("argmin early_error", EE.argmin(1)), ("argMAX drift_total (control)", DT.argmax(1))]
    for lam in (0.25, 0.5, 1.0):
        cands.append((f"z(dino_patch) - {lam}*z(drift_total)", (z(DP) - lam * z(DT)).argmax(1)))
    for lam in (0.25, 0.5, 1.0):
        cands.append((f"z(dino_patch) + {lam}*z(drift_total)", (z(DP) + lam * z(DT)).argmax(1)))
    cands.append(("argmax drift_window", DW.argmax(1))); cands.append(("argmax early_error", EE.argmax(1)))
    for name, pick in cands:
        h = hr(pick); a = float(np.mean(pick == orac)); out["q2"][name] = {"headroom_pct": float(h), "top1_agree_vqa": a}
        print(f"  {name:38s} {h:8.1f}% {a:9.3f}")
    json.dump(out, open(args.out, "w"), indent=1)
    np.savez(Path(args.out).with_suffix(".npz"), idx=np.array([r["idx"] for r in recs]), drift_total=DT, drift_window=DW,
             peak_step=PK, early_error=EE)
    # a manifest with the drift fields, so `CD_drift_hard` (argmax teacher_drift) can be trained on this cache
    od = Path(args.records + "_drift"); od.mkdir(exist_ok=True)
    for s_ in range(4):
        with open(od / f"selection_rank{s_}.jsonl", "w") as fh:
            for k, r in enumerate(recs):
                if k % 4 == s_:
                    rr = dict(r); rr["teacher_drift"] = DT[k].tolist(); rr["teacher_drift_argmax_idx"] = int(DT[k].argmax())
                    fh.write(json.dumps(rr) + "\n")
    print(f"wrote {od} (records + teacher_drift)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
