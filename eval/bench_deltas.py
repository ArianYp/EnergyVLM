#!/usr/bin/env python3
"""Benchmark-prompt campaign (docs/bench/): evaluator-argmax selection vs the random-pick control on
the SAME prompts, over seeds, per category, against CTCal's published deltas over its own base.

The comparison is delta-to-delta by design: their absolutes are SD3 (2B) at 1024 px with ~30 guided
steps, ours a 4-step guidance-free SD3.5-M student at 512 px. What is comparable is how much each
method's selection / fine-tuning moves its own starting point.

Reads the experimental tree's phaseN/eval_<label>_<job> records (common/artifacts.py).

    python eval/bench_deltas.py [--raw] [--artifacts <tree>]    -> docs/bench/DELTAS.md (DELTAS_raw.md)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402
from eval.nested_grid import paired  # noqa: E402

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
# CTCal (CVPR 2026) Table 1: SD3 (2B) + CTCal minus SD3 (2B)
CTCAL = {"color": 0.0311, "shape": 0.0083, "texture": 0.0247, "spatial": 0.0276,
         "3d_spatial": 0.0033, "numeracy": 0.0118, "non_spatial": 0.0085, "complex": 0.0043}
ARG = "W_CD_bench_hard-k10-hp1_bench-hp1{s}_s{i}"      # evaluator-argmax arm (--selector bench)
CTL = "W_B2-k10-hp1_bench-hp1{s}_s{i}"                 # random-pick control on the same cache


def find(label):
    ds = [d for d in sorted(glob.glob(f"phaseN/eval_{label}_*")) if os.path.exists(f"{d}/alignment.json")]
    return ds[-1] if ds else None


def per_prompt(d):
    out = {}
    for c in CATS:
        # exact '<label>_s<steps>_<category>' match: '*_spatial' also catches '_3d_spatial' and '_non_spatial'
        fs = [f for f in glob.glob(f"{d}/compbench_scores/*_{c}/scores.json")
              if re.search(rf"_s\d+_{re.escape(c)}$", os.path.basename(os.path.dirname(f)))]
        if fs:
            for r in json.load(open(fs[0]))["per_prompt"]:
                out[(c, int(r["idx"]))] = float(r["score"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--raw", action="store_true", help="raw final checkpoints instead of the averaged ones")
    ap.add_argument("--out", default=None, help="default docs/bench/DELTAS.md (DELTAS_raw.md with --raw)")
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    suf = "" if args.raw else "-avg"
    seeds, A, B = [], {}, {}
    for i in (0, 1, 2):
        da, db = find(ARG.format(s=suf, i=i)), find(CTL.format(s=suf, i=i))
        if da and db:
            seeds.append(i); A[i] = (per_prompt(da), json.load(open(f"{da}/alignment.json")))
            B[i] = (per_prompt(db), json.load(open(f"{db}/alignment.json")))
    if not seeds:
        print("no matched seed pairs yet"); return
    L = [f"# Benchmark-prompt selection: evaluator-argmax vs random pick ({'raw final' if args.raw else 'averaged'} checkpoints)", "",
         f"Seeds {seeds}. Both arms: the same 5,559 T2I-CompBench++ TRAIN prompts, 16 teacher candidates each, K=10 grid, "
         "converged schedule, 2,780 updates. The ONLY difference is which of the 16 trajectories the student distils: "
         "the one the category's own official evaluator scores highest, or a random one. Evaluated on the 2,398 held-out "
         "val prompts, 1 image per prompt.", "",
         "| | " + " | ".join(f"seed {i}" for i in seeds) + " | mean | CTCal delta | ours - theirs |", "|---|" + "---|" * (len(seeds) + 3)]
    dm = [A[i][1]["compbench_mean"] - B[i][1]["compbench_mean"] for i in seeds]
    L.append("| **CompBench mean** | " + " | ".join(f"{x:+.4f}" for x in dm) + f" | **{np.mean(dm):+.4f}** | {np.mean(list(CTCAL.values())):+.4f} | {np.mean(dm) - np.mean(list(CTCAL.values())):+.4f} |")
    gm = [A[i][1]["geneval2"] - B[i][1]["geneval2"] for i in seeds]
    L.append("| GenEval2 | " + " | ".join(f"{x:+.4f}" for x in gm) + f" | {np.mean(gm):+.4f} | – | – |")
    for c in CATS:
        d = [A[i][1]["compbench"][c] - B[i][1]["compbench"][c] for i in seeds]
        L.append(f"| {c} | " + " | ".join(f"{x:+.4f}" for x in d) + f" | {np.mean(d):+.4f} | {CTCAL[c]:+.4f} | {np.mean(d) - CTCAL[c]:+.4f} |")
    pa = {(i,) + k: v for i in seeds for k, v in A[i][0].items()}
    pb = {(i,) + k: v for i in seeds for k, v in B[i][0].items()}
    p = paired(pb, pa)
    L += ["", f"Pooled per-prompt paired test over {len(seeds)} seed(s): win rate {p['win']:.3f}, sign p {p['sign_p']:.3g}, "
              f"Wilcoxon p {p['wilcoxon_p']:.3g}, n = {p['n']}.",
          "", "Absolutes for reference:", "",
          "| arm | " + " | ".join(f"seed {i}" for i in seeds) + " |", "|---|" + "---|" * len(seeds),
          "| evaluator-argmax | " + " | ".join(f"{A[i][1]['compbench_mean']:.4f}" for i in seeds) + " |",
          "| random pick | " + " | ".join(f"{B[i][1]['compbench_mean']:.4f}" for i in seeds) + " |",
          "", "Our 28-step guided teacher scores 0.5053 on the same pool; our best COCO-trained student 0.4951."]
    out = args.out or str(ROOT / "docs" / "bench" / f"DELTAS{'_raw' if args.raw else ''}.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n"); print("\n".join(L)); print("\nwrote", out)


if __name__ == "__main__":
    main()
