#!/usr/bin/env python3
"""Randomly selected rows for the exact-reward qualitative sheet (the counterpart of the
cherry-picked sheets): N prompts per benchmark drawn uniformly with a fixed seed from the prompts
all three evaluated models were scored on. Same output format as select_rows.py.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_rows import per_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", required=True); ap.add_argument("--argmax", required=True); ap.add_argument("--reward", required=True)
    ap.add_argument("--pool", default="phaseFP/eval_pool_101203")
    ap.add_argument("--out_root", default="phaseW/qual_rewx_random")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    cb_pool = {(r["category"], r["prompt"]): int(r["idx"]) for r in json.load(open(f"{args.pool}/compbench_prompts.json"))}
    rng = np.random.default_rng(args.seed); sel = {}
    for bench in ("compbench", "geneval2"):
        R, A, X = (per_prompt(d, bench, cb_pool) for d in (args.random, args.argmax, args.reward))
        ids = sorted(set(R) & set(A) & set(X))
        pick = sorted(int(i) for i in rng.choice(ids, size=args.n, replace=False))
        rows = [{"idx": i, "prompt": A[i][2], "category": A[i][1], "margin": X[i][0] - A[i][0], "random": R[i][0], "argmax": A[i][0], "reward": X[i][0], "panel": "random"} for i in pick]
        sel[bench] = rows
        (root / f"{bench}_subset.json").write_text(json.dumps([{"idx": r["idx"], "category": r["category"], "prompt": r["prompt"]} for r in rows], indent=1))
        print(f"{bench}: " + "; ".join(f"p{r['idx']:05d} [{r['category']}] rnd {r['random']:.2f} arg {r['argmax']:.2f} rew {r['reward']:.2f}" for r in rows))
    (root / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"wrote {root}/selection.json and subset files")


if __name__ == "__main__":
    main()
