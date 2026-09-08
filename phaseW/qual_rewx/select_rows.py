#!/usr/bin/env python3
"""Row selection for the exact-reward qualitative figure, from three evaluated models (averaged
checkpoints, seed 0): random pick, argmax, argmax + exact DINO reward.

Rule, stated on the figure: the 8 prompts with the largest (exact reward - argmax) per-prompt
benchmark margin, at most 2 per CompBench category; plus, for honesty, the 4 prompts with the most
NEGATIVE margin (where the reward hurts) under the same cap. Writes selection.json (consumed by
build_figure.py) and <bench>_subset.json prompt lists (consumed by the generation job, which reuses
each prompt's ORIGINAL pool index so every column shares initial noise).
"""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict
from pathlib import Path


def per_prompt(eval_dir: str, bench: str, pool: dict) -> dict:
    out = {}
    if bench == "compbench":
        for p in glob.glob(f"{eval_dir}/compbench_scores/*/scores.json"):
            for r in json.load(open(p))["per_prompt"]:
                out[pool[(r["category"], r["prompt"])]] = (float(r["score"]), r["category"], r["prompt"])
    else:
        for p in glob.glob(f"{eval_dir}/geneval2_scores/*/scores.json"):
            for r in json.load(open(p))["per_prompt"]:
                out[int(r["idx"])] = (float(r["score"]), "geneval2", r["prompt"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", required=True); ap.add_argument("--argmax", required=True); ap.add_argument("--reward", required=True)
    ap.add_argument("--pool", default="phaseFP/eval_pool_101203")
    ap.add_argument("--out_root", default="phaseW/qual_rewx")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--n_neg", type=int, default=4); ap.add_argument("--per_category", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    cb_pool = {(r["category"], r["prompt"]): int(r["idx"]) for r in json.load(open(f"{args.pool}/compbench_prompts.json"))}
    sel = {}
    for bench in ("compbench", "geneval2"):
        R, A, X = (per_prompt(d, bench, cb_pool) for d in (args.random, args.argmax, args.reward))
        rows = []
        for i in sorted(set(R) & set(A) & set(X)):
            a, cat, prompt = A[i]
            rows.append({"idx": i, "prompt": prompt, "category": cat, "margin": X[i][0] - a, "random": R[i][0], "argmax": a, "reward": X[i][0]})
        picked = []
        for sign, n in ((-1, args.n), (1, args.n_neg)):
            used = defaultdict(int); k = 0
            for r in sorted(rows, key=lambda r: sign * r["margin"]):
                if sign == 1 and r["margin"] >= 0:
                    break
                if bench == "compbench" and used[r["category"]] >= args.per_category:
                    continue
                r = dict(r, panel="gain" if sign == -1 else "loss"); picked.append(r); used[r["category"]] += 1; k += 1
                if k == n:
                    break
        sel[bench] = picked
        (root / f"{bench}_subset.json").write_text(json.dumps([{"idx": r["idx"], "category": r["category"], "prompt": r["prompt"]} for r in picked], indent=1))
        print(f"{bench}: " + "; ".join(f"p{r['idx']:05d} [{r['category']}] {r['panel']} rnd {r['random']:.2f} arg {r['argmax']:.2f} rew {r['reward']:.2f}" for r in picked))
        n_pos = sum(r["margin"] > 0 for r in rows); n_neg = sum(r["margin"] < 0 for r in rows)
        print(f"  {bench}: reward beats argmax on {n_pos} prompts, loses on {n_neg}, ties on {len(rows) - n_pos - n_neg} (of {len(rows)})")
    (root / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"wrote {root}/selection.json and subset files")


if __name__ == "__main__":
    main()
