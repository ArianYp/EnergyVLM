#!/usr/bin/env python3
"""Log the exact-reward study (S4h) to wandb as one analysis run: seed statistics of the reward arms
(raw and averaged), the held-out DINO monitor, the fidelity rows, the in-training monitor summary
and the comparison figure.

    python3 phaseW/latent_scorer/log_rewx_to_wandb.py --project sd-phaseW-s4
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wandb

ap = argparse.ArgumentParser(); ap.add_argument("--project", default="sd-phaseW-s4"); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
run = wandb.init(project=args.project, name="exact-reward-arms-stats", job_type="analysis",
                 config={"arms": ["S4_CD_dinop_hard-rewX", "S4_CD_dinop_hard-rewF", "S4_CD_dinop_hard-rewR", "S4_CD_dinop_hard", "S4_B2"],
                         "lambda_rgb": 15.504, "lambda_proj": 80.0, "reward_states": 2, "seeds": [0, 1, 2], "jobs": "phaseW/s4h_jobs.txt"})


def md_table(path, pattern, columns):
    rows = []
    for ln in open(path):
        if re.search(pattern, ln) and ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= len(columns) and not set(cells[0]) <= {"-"}:
                rows.append(cells[:len(columns)])
    return wandb.Table(columns=columns, data=rows)


stats_md = HERE.parent / "s4_seed_stats.md"
cols_arms = ["arm", "CompBench per seed", "CompBench mean+-sd", "GenEval2 per seed", "GenEval2 mean+-sd", "GPU-h", "wall-h"]
cols_con = ["contrast", "CompBench per seed", "diff+-sd", "95% CI", "p", "GenEval2 per seed", "diff+-sd", "p"]
run.log({"seed_stats/arms": md_table(stats_md, r"reward|^\| argmax \||random \(fixed draw\) \|", cols_arms),
         "seed_stats/contrasts": md_table(stats_md, r"reward.* vs ", cols_con)})
run.log({"heldout/table": md_table(HERE / "heldout_summary.md", r"random pick|argmax", ["arm", "checkpoint", "seeds", "x0_dino mean+-sd", "sample_dino mean+-sd"])})
fid = Path(HERE.parent / "_equiv/clean-tree/docs/selection_rule_fidelity.md")
if fid.exists():
    run.log({"fidelity/rows": md_table(fid, r"reward|^\| argmax \||random \(fixed draw\)", ["arm", "FID", "CMMD", "precision", "recall"])})
summ = json.load(open(HERE / "rewx_monitor_summary.json"))
for k, v in summ.items():
    for a, b in v.items():
        run.summary[f"monitor/{k}/{a}"] = b
held = json.load(open(HERE / "heldout" / "CD_dinop_hard-rewX_s0@avg_last5.json"))["summary"] if (HERE / "heldout" / "CD_dinop_hard-rewX_s0@avg_last5.json").exists() else {}
for k, v in held.items():
    run.summary[f"heldout/rewX_s0_avg/{k}"] = v
run.summary.update({"compbench/rewX_avg_mean": 0.4984, "compbench/argmax_avg_mean": 0.4873, "compbench/rewX_minus_argmax_avg": 0.0111,
                    "compbench/rewX_minus_argmax_p_paired": 0.085, "compbench/rewX_minus_argmax_p_welch": 0.018,
                    "geneval2/rewX_avg_mean": 25.25, "geneval2/argmax_avg_mean": 23.73, "cmmd/rewX_avg": 0.69, "cmmd/argmax_avg": 0.80,
                    "heldout/x0_dino_rewX_minus_argmax_avg": 0.0080, "heldout/sample_dino_rewX_minus_argmax_avg": 0.0065})
fig = HERE.parents[1] / "reports" / "figs" / "reward_exact.png"
if fig.exists():
    run.log({"figures/reward_exact": wandb.Image(str(fig))})
art = wandb.Artifact("exact-reward-study", type="analysis")
for f in [stats_md, HERE / "heldout_summary.md", HERE / "rewx_monitor_summary.json", HERE / "rewx_monitor.json", fid, HERE.parents[1] / "reports" / "figs" / "reward_exact.pdf"]:
    if f.exists():
        art.add_file(str(f))
run.log_artifact(art)
run.finish()
print("logged", run.url)
