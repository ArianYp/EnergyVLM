#!/usr/bin/env python3
"""Rows for a GenEval2 qualitative sheet restricted to prompts with a given atom count (default 9):
the N prompts with the largest (reward student - naive student) per-prompt margin, then the N_NEG
most negative, from the geneval2 per-prompt scores of three evaluated models (naive, argmax,
argmax + exact reward). Same output format as select_rows.py; only the geneval2 bench is written.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path


def per_prompt(eval_dir):
    f = glob.glob(f"{eval_dir}/geneval2_scores/*/scores.json")[0]
    return {int(r["idx"]): r for r in json.load(open(f))["per_prompt"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", required=True); ap.add_argument("--argmax", required=True); ap.add_argument("--reward", required=True)
    ap.add_argument("--atoms", type=int, default=9); ap.add_argument("--n", type=int, default=8); ap.add_argument("--n_neg", type=int, default=4)
    ap.add_argument("--out_root", default="phaseW/qual_rewx_9atom")
    args = ap.parse_args()
    R, A, X = per_prompt(args.random), per_prompt(args.argmax), per_prompt(args.reward)
    ids = [i for i in R if int(R[i]["atom_count"]) == args.atoms and i in A and i in X]
    rows = [{"idx": i, "prompt": R[i]["prompt"], "category": f"geneval2 · {args.atoms} atoms", "margin": X[i]["score"] - R[i]["score"],
             "random": R[i]["score"], "argmax": A[i]["score"], "reward": X[i]["score"], "skills": R[i]["skills"]} for i in ids]
    gain = sorted(rows, key=lambda r: -r["margin"])[:args.n]
    loss = [r for r in sorted(rows, key=lambda r: r["margin"]) if r["margin"] < 0][:args.n_neg]
    sel = [dict(r, panel="gain") for r in gain] + [dict(r, panel="loss") for r in loss]
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps({"geneval2": sel}, indent=1))
    (root / "geneval2_subset.json").write_text(json.dumps([{"idx": r["idx"], "category": "geneval2", "prompt": r["prompt"]} for r in sel], indent=1))
    (root / "compbench_subset.json").write_text("[]")
    n_pos = sum(r["margin"] > 0 for r in rows); n_neg = sum(r["margin"] < 0 for r in rows)
    print(f"{args.atoms}-atom prompts: {len(rows)}; reward beats naive on {n_pos}, loses on {n_neg}, ties {len(rows) - n_pos - n_neg}")
    for r in sel:
        print(f"  p{r['idx']:05d} {r['panel']:4s} rnd {r['random']:.2f} arg {r['argmax']:.2f} rew {r['reward']:.2f}  {r['prompt'][:90]}")


if __name__ == "__main__":
    main()
