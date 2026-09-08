#!/usr/bin/env python3
"""The in-training monitor of the reward arms, pulled from wandb: the reward the optimizer sees
(`reward/r_mean`), the offline RGB DINO score of the same predictions (`reward/rgb_score`, every
--reward_monitor_every updates), their correlation, and the consistency loss alone
(`train/loss_cd`; `train/loss` for the reward-free argmax runs). Summarises start (updates 100-300)
against end (last 200 updates) per seed and draws the three-panel figure of report Section 11.7.

    python3 eval/reward_monitor.py --project <entity>/<project> \
        --run_regex '^dino_patch(-rew[XFR])?_3k_s[0-2]$' --out_json out/reward_monitor.json --fig docs/figs/reward_exact.pdf
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = {"rewX": "argmax + exact DINO reward", "rewF": "argmax + projector reward, frozen", "rewR": "argmax + projector reward, refreshed"}
COL = {"rewX": "#c0392b", "rewF": "#7f8c8d", "rewR": "#2c3e50", "argmax": "#2980b9"}
KEYS = ["_step", "reward/r_mean", "reward/rgb_score", "reward/proj_score", "reward/rgb_proj_corr", "train/loss_cd", "train/loss"]


def fetch(project, run_regex):
    import wandb
    api = wandb.Api(timeout=120)
    data = defaultdict(dict)
    for r in api.runs(project, order="-created_at"):
        if not re.search(run_regex, r.name):
            continue
        arm = next((a for a in ARMS if f"-{a}" in r.name), "argmax")
        m = re.search(r"_s(\d+)", r.name); seed = int(m.group(1)) if m else 0
        if seed in data[arm]:
            continue                                   # newest run per arm and seed
        h = {k: [] for k in KEYS}
        for row in r.scan_history(keys=None, page_size=2000):
            if any(row.get(k) is not None for k in KEYS[1:]):
                for k in KEYS:
                    v = row.get(k); h[k].append(float(v) if isinstance(v, (int, float)) else float("nan"))
        data[arm][seed] = {"run": r.name, "id": r.id, "runtime_s": float(r.summary.get("_runtime", float("nan"))), **h}
        print(f"fetched {r.name}: {len(h['_step'])} rows", flush=True)
    return data


def window(step, val, lo, hi):
    s, v = np.asarray(step, float), np.asarray(val, float); m = (s >= lo) & (s <= hi) & np.isfinite(v)
    return float(v[m].mean()) if m.any() else float("nan")


def rmean(v, w=7):
    v = np.asarray(v, float); out = np.full_like(v, np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - w + 1):i + 1]; out[i] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
    return out


def smooth(s, v, w=200):
    s, v = np.asarray(s, float), np.asarray(v, float); m = np.isfinite(v); s, v = s[m], v[m]
    if len(v) < w:
        return s, v
    return s[w - 1:], np.convolve(v, np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="<entity>/<project>")
    ap.add_argument("--run_regex", default=r"^dino_patch(-rew[XFR])?_3k_s[0-2]$")
    ap.add_argument("--out_json", default="out/reward_monitor.json")
    ap.add_argument("--fig", default="docs/figs/reward_exact.pdf")
    ap.add_argument("--last", type=int, default=6000, help="number of updates (end window = last 200)")
    args = ap.parse_args()
    oj = Path(args.out_json)
    if oj.exists():
        data = {a: {int(s): v for s, v in d.items()} for a, d in json.load(open(oj)).items()}
    else:
        data = fetch(args.project, args.run_regex); oj.parent.mkdir(parents=True, exist_ok=True); json.dump(data, open(oj, "w"))
    T = args.last
    summ = {}
    for arm in ARMS:
        for seed, h in sorted(data.get(arm, {}).items()):
            st = h["_step"]
            summ[f"{arm}_s{seed}"] = {"reward_start": window(st, h["reward/r_mean"], 100, 300), "reward_end": window(st, h["reward/r_mean"], T - 200, T),
                                      "rgb_start": window(st, h["reward/rgb_score"], 100, 300), "rgb_end": window(st, h["reward/rgb_score"], T - 200, T),
                                      "corr_start": window(st, h["reward/rgb_proj_corr"], 100, 300), "corr_end": window(st, h["reward/rgb_proj_corr"], T - 200, T),
                                      "loss_cd_end": window(st, h["train/loss_cd"], T - 1000, T), "runtime_s": h["runtime_s"]}
    for seed, h in sorted(data.get("argmax", {}).items()):
        summ[f"argmax_s{seed}"] = {"loss_end": window(h["_step"], h["train/loss"], T - 1000, T), "runtime_s": h["runtime_s"]}
    for arm in ARMS:
        ks = [k for k in summ if k.startswith(arm)]
        if ks:
            summ[f"{arm}_mean"] = {f: float(np.nanmean([summ[k][f] for k in ks])) for f in summ[ks[0]]}
    json.dump(summ, open(oj.with_name(oj.stem + "_summary.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, {a: round(b, 4) for a, b in v.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for arm in ["rewF", "rewR", "rewX"]:
        runs = sorted(data.get(arm, {}).items()); curves = []
        for seed, h in runs:
            s, v = np.asarray(h["_step"]), np.asarray(h["reward/rgb_score"]); m = np.isfinite(v); s, v = s[m], v[m]
            ax[0].plot(s, rmean(v), color=COL[arm], lw=0.8, ls="-" if arm == "rewX" else "--", alpha=0.35); curves.append((s, rmean(v)))
        if curves:
            L = min(len(c[1]) for c in curves)
            ax[0].plot(curves[0][0][:L], np.mean([c[1][:L] for c in curves], 0), color=COL[arm], lw=2.2, ls="-" if arm == "rewX" else "--", label=ARMS[arm] + " (seed mean)")
        for i, (seed, h) in enumerate(runs):
            # matched approximation gap: reward-path score minus offline RGB score on the SAME monitor
            # latents, one point per monitor evaluation (repeated logged values dropped)
            s, pr, v = np.asarray(h["_step"]), np.asarray(h["reward/proj_score"]), np.asarray(h["reward/rgb_score"]); m = np.isfinite(v) & np.isfinite(pr)
            s, pr, v = s[m], pr[m], v[m]; keep = np.r_[True, (np.diff(pr) != 0) | (np.diff(v) != 0)]
            ax[1].plot(s[keep], pr[keep] - v[keep], color=COL[arm], lw=1.6 if arm == "rewX" else 1.0, ls="-" if arm == "rewX" else "--", alpha=0.9 if arm == "rewX" else 0.6, label=ARMS[arm] if i == 0 else None)
    ax[0].set_xlabel("update"); ax[0].set_ylabel("RGB DINO score of decoded $\\hat{x}_0$ (offline scorer)"); ax[0].set_title("(a) the true score of the predictions (700-update running mean)", fontsize=9); ax[0].legend(fontsize=7, loc="lower right")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("update"); ax[1].set_ylabel("reward-path score $-$ offline RGB score, same latents"); ax[1].set_title("(b) approximation gap on the monitor latents", fontsize=9); ax[1].legend(fontsize=7)
    for i, (seed, h) in enumerate(sorted(data.get("argmax", {}).items())):
        s, v = smooth(h["_step"], h["train/loss"]); ax[2].plot(s, v, color=COL["argmax"], lw=1.0, alpha=0.7, label="argmax (no reward)" if i == 0 else None)
    for i, (seed, h) in enumerate(sorted(data.get("rewX", {}).items())):
        s, v = smooth(h["_step"], h["train/loss_cd"]); ax[2].plot(s, v, color=COL["rewX"], lw=1.4, label=ARMS["rewX"] if i == 0 else None)
    ax[2].set_xlabel("update"); ax[2].set_ylabel("consistency loss (200-update mean)"); ax[2].set_title("(c) consistency loss with and without the reward"); ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout(); Path(args.fig).parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.fig); fig.savefig(Path(args.fig).with_suffix(".png"), dpi=130)
    print("wrote", args.fig)


if __name__ == "__main__":
    main()
