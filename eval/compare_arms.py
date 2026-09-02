#!/usr/bin/env python3
"""Paired comparison of evaluated models: per-prompt differences, averaged over training seeds.

Every model was evaluated on the same prompts from the same initial noise, so the comparison is
paired per prompt (joined on category and prompt text). Reported per evaluator family:

    UniDet detection   spatial, 3d_spatial, numeracy
    BLIP-VQA           color, shape, texture
    CLIPScore          non_spatial
    CompBench overall  all eight categories
    GenEval2           Soft-TIFA

Evaluation directories are the output of scripts/eval_alignment.lsf, one per (model, seed), named
{label}_s{seed}, each holding alignment.json plus compbench_scores/*/scores.json and
geneval2_scores/*/scores.json.

    python eval/compare_arms.py --root out/eval --baseline naive --arms dino_patch
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

FAMILIES = {
    "UniDet detection": ("spatial", "3d_spatial", "numeracy"),
    "BLIP-VQA": ("color", "shape", "texture"),
    "CLIPScore": ("non_spatial",),
    "CompBench overall": ("spatial", "3d_spatial", "numeracy", "color", "shape", "texture", "non_spatial", "complex"),
}


def load(root: str) -> dict:
    runs = {}
    for d in sorted(glob.glob(f"{root}/eval_*")):
        a = Path(d) / "alignment.json"
        if not a.exists():
            continue
        label = json.loads(a.read_text())["label"]
        cb, ge = {}, {}
        for p in glob.glob(str(Path(d) / "compbench_scores" / "*" / "scores.json")):
            for r in json.loads(Path(p).read_text())["per_prompt"]:
                cb[(r["category"], r["prompt"])] = float(r["score"])
        for p in glob.glob(str(Path(d) / "geneval2_scores" / "*" / "scores.json")):
            for r in json.loads(Path(p).read_text())["per_prompt"]:
                ge[r["prompt"]] = float(r["score"])
        if cb:
            runs[label] = {"compbench": cb, "geneval2": ge}
    return runs


def seeds_of(runs: dict, arm: str) -> dict:
    out = {}
    for label, data in runs.items():
        if label.startswith(arm + "_s") and label[len(arm) + 2:].isdigit():
            out[int(label[len(arm) + 2:])] = data
    return out


def contrast(A: dict, B: dict, bench: str, keep) -> str:
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return "no shared seeds"
    keys = None
    for s in seeds:
        for D in (A, B):
            ks = {k for k in D[s][bench] if keep(k)}
            keys = ks if keys is None else keys & ks
    keys = sorted(keys)
    if not keys:
        return "no shared prompts"
    a = np.mean([[A[s][bench][k] for k in keys] for s in seeds], 0)
    b = np.mean([[B[s][bench][k] for k in keys] for s in seeds], 0)
    d = a - b
    per_seed = "/".join(f"{np.mean([A[s][bench][k] for k in keys]) - np.mean([B[s][bench][k] for k in keys]):+.4f}"
                        for s in seeds)
    wp = wilcoxon(a, b)[1] if np.any(d != 0) else 1.0
    return (f"n={len(keys):5d}  delta={d.mean():+.4f}  per-seed {per_seed}  "
            f"t p={ttest_rel(a, b)[1]:.2e}  wilcoxon p={wp:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/eval")
    ap.add_argument("--baseline", default="naive")
    ap.add_argument("--arms", default="dino_patch", help="comma-separated labels to compare to the baseline")
    args = ap.parse_args()

    runs = load(args.root)
    B = seeds_of(runs, args.baseline)
    if not B:
        raise SystemExit(f"no evaluations for baseline {args.baseline!r}; found {sorted(runs)}")
    print(f"baseline {args.baseline}: seeds {sorted(B)}")
    for arm in args.arms.split(","):
        A = seeds_of(runs, arm)
        print(f"\n{arm} vs {args.baseline}   seeds {sorted(A)}")
        for name, cats in FAMILIES.items():
            print(f"    {name:18s} {contrast(A, B, 'compbench', lambda k, c=cats: k[0] in c)}")
        print(f"    {'GenEval2':18s} {contrast(A, B, 'geneval2', lambda k: True)}")


if __name__ == "__main__":
    main()
