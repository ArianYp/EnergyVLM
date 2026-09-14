#!/usr/bin/env python3
"""Pull the training curves (consistency loss, per-noise-level losses, gradient norm, lr) of the 3k and
118k runs from wandb into iclr2027/loss_curves.json. Uses scan_history so runs logged by older trainer
versions (fewer keys) are included.

    python3 iclr2027/fetch_loss_curves.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WANT = ("train/loss", "train/loss_cd", "train/loss_k3", "train/loss_k4", "train/loss_k5", "train/loss_k6", "train/loss_k7",
        "train/grad_norm", "train/lr", "train/xhat_norm", "train/xtea_norm", "reward/r_mean", "reward/rgb_score")
RUNS = {  # project -> {arm name: regex on display name}
    "mll_sut/sd-phaseW-s4": {"3k naive": r"^phaseS4_B2_s[0-2]_\d+$", "3k scored": r"^phaseS4_CD_dinop_hard_s[0-2]_\d+$",
                             "3k ours": r"^phaseS4_CD_dinop_hard-rewRi-s16_s[0-2]_\d+$", "3k decode-reward": r"^phaseS4_CD_dinop_hard-rewXi_s[0-2]_\d+$"},
    "mll_sut/sd-phaseW-118k": {"118k naive": r"^phaseW_B2_118k_s[0-2]_\d+$", "118k scored": r"^phaseW_CD_dinop_hard_118k_s[0-2]_\d+$",
                               "118k decode-reward": r"^phaseW_CD_dinop_hard_118k-rewX_s[0-2]_\d+$", "118k ours (running)": r"^phaseW_CD_dinop_hard_118k-rewRi-s16_s[0-2]_\d+$"},
}


def main():
    import wandb
    api = wandb.Api(timeout=300)
    out = defaultdict(dict)
    for project, arms in RUNS.items():
        runs = list(api.runs(project, order="-created_at"))
        print(project, len(runs), "runs")
        for arm, pat in arms.items():
            for r in runs:
                if not re.match(pat, r.name):
                    continue
                seed = int(r.name.split("_s")[-1].split("_")[0])
                if seed in out[arm]:
                    continue  # newest run per arm/seed
                series = defaultdict(list)
                keys_seen = set()
                for row in r.scan_history(page_size=5000):
                    keys_seen.update(k for k in row if not k.startswith("_"))
                    st = row.get("_step")
                    if st is None:
                        continue
                    for k in WANT:
                        v = row.get(k)
                        if isinstance(v, (int, float)) and v == v:
                            series[k].append((int(st), float(v)))
                cfg = {k: r.config.get(k) for k in ("lr", "num_warmup_steps", "num_steps", "accum", "seed", "weight_decay", "grad_clip", "log_every")}
                out[arm][seed] = {"run": r.name, "state": r.state, "config": cfg, "keys": sorted(keys_seen), **{k: v for k, v in series.items()}}
                print(f"  {arm:22s} s{seed} {r.name} [{r.state}] " + ", ".join(f"{k.split('/')[-1]}:{len(v)}" for k, v in series.items())
                      + (f" | keys: {sorted(keys_seen)[:12]}" if not series else ""), flush=True)
    json.dump(out, open(os.path.join(ROOT, "iclr2027", "loss_curves.json"), "w"))
    print("wrote iclr2027/loss_curves.json")


if __name__ == "__main__":
    main()
