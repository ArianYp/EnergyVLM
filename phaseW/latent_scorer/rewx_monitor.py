#!/usr/bin/env python3
"""Pull the in-training reward monitor of the exact-reward arm (rewX: differentiable decode + DINO,
lambda from the gradient probe) and of the two projector-reward arms (rewF frozen, rewR refreshed)
from wandb, summarise start/end levels per seed and draw the comparison figure.

  reward/r_mean     mean reward seen by the optimizer (rewX: the differentiable RGB DINO score of the
                    two least-noisy clean estimates; rewF/rewR: the projector score)
  reward/rgb_score  the independent monitor: the same predictions decoded to uint8/PIL and scored by
                    the OFFLINE DINO scorer every 100 updates (32 recent latents)
  train/loss_cd     consistency loss alone (argmax baseline: train/loss, which has no reward term)

    python3 phaseW/latent_scorer/rewx_monitor.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_JSON = HERE / "rewx_monitor.json"
OUT_FIG = REPO / "reports" / "figs" / "reward_exact.pdf"
ARMS = {"rewX": "argmax + exact DINO reward", "rewF": "argmax + projector reward, frozen", "rewR": "argmax + projector reward, refreshed"}
COL = {"rewX": "#c0392b", "rewF": "#7f8c8d", "rewR": "#2c3e50", "argmax": "#2980b9"}


def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    runs = api.runs("mll_sut/sd-phaseW-s4", filters={"display_name": {"$regex": "^phaseS4_CD_dinop_hard(-rew[XFR])?_s[0-2]_"}}, order="-created_at")
    data = defaultdict(dict)
    for r in runs:
        name = r.name
        arm = next((a for a in ARMS if f"-{a}_" in name), "argmax")
        seed = int(name.split("_s")[-1].split("_")[0])
        if seed in data[arm]:
            continue  # newest run per arm/seed
        keys = ["_step", "reward/r_mean", "reward/rgb_score", "reward/rgb_proj_corr", "train/loss_cd", "train/loss"]
        rows = [row for row in r.scan_history(keys=None, page_size=2000)]
        h = {k: [] for k in keys}
        for row in rows:
            if any(row.get(k) is not None for k in keys[1:]):
                for k in keys:
                    v = row.get(k)
                    h[k].append(float(v) if isinstance(v, (int, float)) else float("nan"))
        data[arm][seed] = {"run": name, "id": r.id, "runtime_s": float(r.summary.get("_runtime", float("nan"))), **{k: h[k] for k in keys}}
        print(f"fetched {name}: {len(h['_step'])} rows", flush=True)
    return data


def window(step, val, lo, hi):
    s, v = np.asarray(step, float), np.asarray(val, float)
    m = (s >= lo) & (s <= hi) & np.isfinite(v)
    return float(v[m].mean()) if m.any() else float("nan")


def main():
    if OUT_JSON.exists():
        data = json.load(open(OUT_JSON))
        data = {a: {int(s): v for s, v in d.items()} for a, d in data.items()}
    else:
        data = fetch(); json.dump(data, open(OUT_JSON, "w"))
    summ = {}
    for arm in ["rewX", "rewF", "rewR"]:
        for seed, h in sorted(data.get(arm, {}).items()):
            st, rm, rgb = h["_step"], h["reward/r_mean"], h["reward/rgb_score"]
            summ[f"{arm}_s{seed}"] = {"reward_start": window(st, rm, 100, 300), "reward_end": window(st, rm, 5800, 6000),
                                      "rgb_start": window(st, rgb, 100, 300), "rgb_end": window(st, rgb, 5800, 6000),
                                      "corr_start": window(st, h["reward/rgb_proj_corr"], 100, 300), "corr_end": window(st, h["reward/rgb_proj_corr"], 5800, 6000),
                                      "loss_cd_end": window(st, h["train/loss_cd"], 5000, 6000), "runtime_s": h["runtime_s"]}
    for seed, h in sorted(data.get("argmax", {}).items()):
        summ[f"argmax_s{seed}"] = {"loss_end": window(h["_step"], h["train/loss"], 5000, 6000), "runtime_s": h["runtime_s"]}
    for arm in ["rewX", "rewF", "rewR"]:
        ks = [k for k in summ if k.startswith(arm)]
        if ks:
            summ[f"{arm}_mean"] = {f: float(np.nanmean([summ[k][f] for k in ks])) for f in summ[ks[0]]}
    json.dump(summ, open(HERE / "rewx_monitor_summary.json", "w"), indent=1)
    for k, v in summ.items():
        print(k, {a: round(b, 4) for a, b in v.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    # (a) true RGB score of decoded predictions (offline monitor) over training: 32 latents per point,
    #     so each run is shown as a running mean over 7 monitor points (700 updates); the bold line is the seed mean
    def rmean(v, w=7):
        v = np.asarray(v, float); out = np.full_like(v, np.nan)
        for i in range(len(v)):
            seg = v[max(0, i - w + 1):i + 1]; out[i] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
        return out
    for arm in ["rewF", "rewR", "rewX"]:
        runs = sorted(data.get(arm, {}).items())
        curves = []
        for i, (seed, h) in enumerate(runs):
            s, v = np.asarray(h["_step"]), np.asarray(h["reward/rgb_score"])
            m = np.isfinite(v); s, v = s[m], v[m]
            ax[0].plot(s, rmean(v), color=COL[arm], lw=0.8, ls="-" if arm == "rewX" else "--", alpha=0.35)
            curves.append((s, rmean(v)))
        L = min(len(c[1]) for c in curves)
        ax[0].plot(curves[0][0][:L], np.mean([c[1][:L] for c in curves], 0), color=COL[arm], lw=2.2, ls="-" if arm == "rewX" else "--", label=ARMS[arm] + " (seed mean)")
    ax[0].set_xlabel("update"); ax[0].set_ylabel("RGB DINO score of decoded $\\hat{x}_0$ (offline scorer)"); ax[0].set_title("(a) the true score of the predictions (700-update running mean)", fontsize=9); ax[0].legend(fontsize=7, loc="lower right")
    # (b) reward seen by the optimizer vs the monitor for rewX (should coincide) and for the projector arms (should not)
    for arm in ["rewF", "rewR", "rewX"]:
        for i, (seed, h) in enumerate(sorted(data.get(arm, {}).items())):
            s, r, v = np.asarray(h["_step"]), np.asarray(h["reward/r_mean"]), np.asarray(h["reward/rgb_score"])
            m = np.isfinite(v) & np.isfinite(r)
            ax[1].plot(s[m], r[m] - v[m], color=COL[arm], lw=1.6 if arm == "rewX" else 1.0, ls="-" if arm == "rewX" else "--", alpha=0.9 if arm == "rewX" else 0.6, label=ARMS[arm] if i == 0 else None)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("update"); ax[1].set_ylabel("reward $-$ true RGB score"); ax[1].set_title("(b) proxy gap: reward minus the monitor"); ax[1].legend(fontsize=7)
    # (c) consistency loss (smoothed) rewX vs argmax
    def smooth(s, v, w=200):
        s, v = np.asarray(s, float), np.asarray(v, float); m = np.isfinite(v); s, v = s[m], v[m]
        if len(v) < w: return s, v
        c = np.convolve(v, np.ones(w) / w, mode="valid"); return s[w - 1:], c
    for i, (seed, h) in enumerate(sorted(data.get("argmax", {}).items())):
        s, v = smooth(h["_step"], h["train/loss"]); ax[2].plot(s, v, color=COL["argmax"], lw=1.0, alpha=0.7, label="argmax (no reward)" if i == 0 else None)
    for i, (seed, h) in enumerate(sorted(data.get("rewX", {}).items())):
        s, v = smooth(h["_step"], h["train/loss_cd"]); ax[2].plot(s, v, color=COL["rewX"], lw=1.4, label=ARMS["rewX"] if i == 0 else None)
    ax[2].set_xlabel("update"); ax[2].set_ylabel("consistency loss (200-update mean)"); ax[2].set_title("(c) consistency loss with and without the reward"); ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout(); OUT_FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT_FIG); fig.savefig(OUT_FIG.with_suffix(".png"), dpi=130)
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    main()
