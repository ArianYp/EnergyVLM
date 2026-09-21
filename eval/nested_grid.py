#!/usr/bin/env python3
"""Nested-grid evaluation (2026-09-17, docs/nested_grid/): the converged 3k averaged students (naive
145251, ours 145253) sampled at 4 steps on sub-grids of the 8-step training grid vs the scheduler's own
4-step grid. Same prompts, same seeds, same checkpoints, so every contrast is paired per prompt.

Also the home of `load(job)` and `paired(a, b)`, which eval/k10_*.py and eval/bench_deltas.py reuse.
Reads the experimental tree's phaseN/eval_*_<job> records (common/artifacts.py).

    python eval/nested_grid.py [--artifacts <tree>]      -> docs/nested_grid/RESULTS.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
GRIDS = {"scheduler": "1, 0.858, 0.602, 0.009, 0 (states 0,-,-,7: shares only the endpoints)",
         "A": "1, 0.883, 0.694, 0.338, 0 (states 0,2,4,6,8: one coarse last step)",
         "B": "1, 0.883, 0.548, 0.009, 0 (states 0,2,5,7,8: closest to deployment)",
         "C": "1, 0.883, 0.694, 0.009, 0 (states 0,2,4,7,8)"}
RUNS = {"naive": {"scheduler": 145251, "A": 153462, "B": 153464, "C": 153466},
        "ours": {"scheduler": 145253, "A": 153463, "B": 153465, "C": 153467}}


def load(job):
    """Per-prompt CompBench {(category, idx): score}, GenEval2 {idx: score} and alignment.json of one
    evaluation job (1-image eval_* or official 10-image eval10_* directory), keyed by its LSF job id."""
    d = glob.glob(f"phaseN/eval_*_{job}") or glob.glob(f"phaseN/eval10_*_{job}")
    assert len(d) == 1, (job, d)
    d = d[0]
    cb = {}
    for c in CATS:
        f = glob.glob(f"{d}/compbench_scores/*_s4_{c}/scores.json")[0]
        for r in json.load(open(f))["per_prompt"]:
            cb[(c, int(r["idx"]))] = float(r["score"])
    g = json.load(open(glob.glob(f"{d}/geneval2_scores/*_s4/scores.json")[0]))
    ge = {int(r["idx"]): float(r["score"]) for r in g["per_prompt"]}
    return cb, ge, json.load(open(f"{d}/alignment.json"))


def paired(a: dict, b: dict):
    """Paired per-prompt statistics of b - a over the shared keys: mean difference, win rate over the
    non-tied prompts, sign test, paired t-test and Wilcoxon."""
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    d = y - x
    nz = d[d != 0]
    from scipy import stats
    sign_p = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
    t_p = stats.ttest_rel(y, x).pvalue if len(d) > 2 else 1.0
    w_p = stats.wilcoxon(nz).pvalue if len(nz) > 10 else 1.0
    return {"n": len(ks), "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)), "win": float((nz > 0).mean()) if len(nz) else 0.5,
            "ties": int((d == 0).sum()), "sign_p": float(sign_p), "t_p": float(t_p), "wilcoxon_p": float(w_p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--out", default=str(ROOT / "docs" / "nested_grid" / "RESULTS.md"))
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    out = ["# Nested-grid evaluation (2026-09-17)", "",
           "Converged 3k averaged students (seed 0) sampled at 4 steps on sub-grids of the 8-step training grid. Paired per prompt "
           "against the same checkpoint on the scheduler's 4-step grid (the reported numbers). Single-run floor for an unpaired "
           "contrast is 0.0065 CompBench; here the pairing removes the prompt variance, so the sign / Wilcoxon tests are the "
           "relevant ones.", "", "Grids:", ""]
    for k, v in GRIDS.items():
        out.append(f"- {k}: sigma = {v}")
    out += ["", "| model | grid | CompBench | dCB vs scheduler | win rate | sign p | Wilcoxon p | GenEval2 | dGE2 | sign p | " + " | ".join(f"d {c}" for c in CATS) + " |",
            "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(CATS)]
    for model, jobs in RUNS.items():
        ref_cb, ref_ge, ref_al = load(jobs["scheduler"])
        out.append(f"| {model} | scheduler | {ref_al['compbench_mean']:.4f} | – | – | – | – | {ref_al['geneval2']:.4f} | – | – | " + " | ".join("–" for _ in CATS) + " |")
        for grid in ("A", "B", "C"):
            cb, ge, al = load(jobs[grid])
            p = paired(ref_cb, cb); q = paired(ref_ge, ge)
            per_cat = []
            for c in CATS:
                pc = paired({k: v for k, v in ref_cb.items() if k[0] == c}, {k: v for k, v in cb.items() if k[0] == c})
                per_cat.append(f"{pc['mean_diff']:+.4f}" + ("*" if pc["sign_p"] < 0.05 else ""))
            out.append(f"| {model} | {grid} | {al['compbench_mean']:.4f} | {p['mean_diff']:+.4f} | {p['win']:.3f} | {p['sign_p']:.2g} | {p['wilcoxon_p']:.2g} | "
                       f"{al['geneval2']:.4f} | {q['mean_diff']:+.4f} | {q['sign_p']:.2g} | " + " | ".join(per_cat) + " |")
            print(f"{model:6s} grid {grid}: CB {al['compbench_mean']:.4f} ({p['mean_diff']:+.4f}, win {p['win']:.3f}, sign p {p['sign_p']:.3g}, wilcoxon {p['wilcoxon_p']:.3g}, ties {p['ties']}/{p['n']}) "
                  f"GE2 {al['geneval2']:.4f} ({q['mean_diff']:+.4f}, sign p {q['sign_p']:.3g})")
    out += ["", "`*` = per-category sign test p < 0.05. Win rate = share of non-tied prompts where the sub-grid scores higher."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
