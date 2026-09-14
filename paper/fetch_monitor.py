#!/usr/bin/env python3
"""Pull the in-training reward monitor (every 100 updates: mean reward seen by the optimiser, true
RGB DINOv2 score of the same decoded predictions, projector score, their correlation, refresh loss)
for the 3k reward arms from wandb into iclr2027/monitor.json.

    python3 iclr2027/fetch_monitor.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"rewRi-s16": "projector, refreshed every 100, 16 steps (ours)", "rewRi-e25": "projector, refreshed every 25",
        "rewR": "projector, refreshed every 100, 4 steps", "rewF": "projector, frozen", "rewX": "decode-based DINOv2 reward",
        "rewXi": "decode-based DINOv2 reward (matched order)"}
KEYS = ["reward/r_mean", "reward/rgb_score", "reward/proj_score", "reward/rgb_proj_corr", "reward/refresh_loss", "train/loss_cd", "train/loss"]


def main():
    import wandb
    api = wandb.Api(timeout=180)
    runs = api.runs("mll_sut/sd-phaseW-s4", filters={"display_name": {"$regex": "^phaseS4_CD_dinop_hard-rew[A-Za-z0-9.-]*_s[0-2]_"}}, order="-created_at")
    data = defaultdict(dict)
    for r in runs:
        name = r.name
        arm = None
        for a in sorted(ARMS, key=len, reverse=True):
            if f"-{a}_s" in name:
                arm = a; break
        if arm is None:
            continue
        seed = int(name.split("_s")[-1].split("_")[0])
        if seed in data[arm]:
            continue  # newest run per arm/seed
        hist = r.history(keys=["_step"] + KEYS, pandas=False, samples=100000)
        series = defaultdict(list)
        for row in hist:
            for k in KEYS:
                if row.get(k) is not None and row[k] == row[k]:
                    series[k].append((int(row["_step"]), float(row[k])))
        data[arm][seed] = {"run": name, "id": r.id, **{k: v for k, v in series.items()}}
        print(f"{arm} s{seed} {name}: " + ", ".join(f"{k}:{len(v)}" for k, v in series.items()), flush=True)
    json.dump({"arms": ARMS, "data": data}, open(os.path.join(ROOT, "iclr2027", "monitor.json"), "w"))
    print("wrote iclr2027/monitor.json")


if __name__ == "__main__":
    main()
