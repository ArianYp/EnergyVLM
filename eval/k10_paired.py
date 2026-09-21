#!/usr/bin/env python3
"""K=10 teacher grid vs the K=8 recipe (docs/k10/), paired per prompt (same prompts, seeds and eval
pipeline), seed 0. Reads the experimental tree's phaseN/eval_*_<job> records (common/artifacts.py).

    python eval/k10_paired.py [--artifacts <tree>]     -> docs/k10/RESULTS.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402
from eval.nested_grid import CATS, load, paired  # noqa: E402

PAIRS = [  # name, reference job, test job
    ("naive K=10 avg vs naive K=8 avg", 145251, 153471),
    ("ours  K=10 avg vs ours  K=8 avg", 145253, 153475),
    ("ours  K=10 avg vs naive K=10 avg", 153471, 153475),
    ("ours  K=8  avg vs naive K=8  avg (reference contrast)", 145251, 145253),
    ("naive K=10 raw vs naive K=8 raw", 145250, 153469),
    ("ours  K=10 raw vs ours  K=8 raw", 145252, 153473),
    ("ours  K=10 raw vs naive K=10 raw", 153469, 153473),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--out", default=str(ROOT / "docs" / "k10" / "RESULTS.md"))
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    out = ["# K=10 teacher grid (2026-09-17), paired per prompt", "",
           "K=10 nests the 4-step inference grid (states 0, 3, 6, 9); supervised states 6-9 (sigma 0.602, 0.465, 0.278, 0.009), "
           "student inputs cover both deployment sigmas. Everything else is the HP1 3k recipe, seed 0. Cache: phaseN/coco_selection_153453.", "",
           "| contrast | CB ref | CB test | dCB | win | sign p | Wilcoxon p | GE2 ref | GE2 test | dGE2 | sign p | " + " | ".join(f"d {c}" for c in CATS) + " |",
           "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(CATS)]
    for name, ja, jb in PAIRS:
        ca, ga, aa = load(ja); cb, gb, ab = load(jb)
        p = paired(ca, cb); q = paired(ga, gb)
        pc = [paired({k: v for k, v in ca.items() if k[0] == c}, {k: v for k, v in cb.items() if k[0] == c}) for c in CATS]
        out.append(f"| {name} | {aa['compbench_mean']:.4f} | {ab['compbench_mean']:.4f} | {p['mean_diff']:+.4f} | {p['win']:.3f} | {p['sign_p']:.2g} | {p['wilcoxon_p']:.2g} | "
                   f"{aa['geneval2']:.4f} | {ab['geneval2']:.4f} | {q['mean_diff']:+.4f} | {q['sign_p']:.2g} | "
                   + " | ".join(f"{x['mean_diff']:+.4f}{'*' if x['sign_p'] < 0.05 else ''}" for x in pc) + " |")
        print(f"{name:52s} CB {aa['compbench_mean']:.4f} -> {ab['compbench_mean']:.4f} ({p['mean_diff']:+.4f}, win {p['win']:.3f}, sign p {p['sign_p']:.3g}, W {p['wilcoxon_p']:.3g}) "
              f"GE2 {aa['geneval2']:.4f} -> {ab['geneval2']:.4f} ({q['mean_diff']:+.4f}, p {q['sign_p']:.2g})")
    out += ["", "`*` = per-category sign test p < 0.05."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n"); print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
