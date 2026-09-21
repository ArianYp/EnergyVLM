#!/usr/bin/env python3
"""Merge the per-category official scores of the benchmark candidates (scripts/score_bench_candidates.lsf)
into a selection cache the trainer reads (selection_rank{0..3}.jsonl + cache_meta.json), docs/bench/:

    idx, category, prompt, seed_base = seed + idx*1000, N, bench_score [N] (the category's official
    evaluator on each candidate), bench_argmax_idx, random_idx, reference "" (no photograph: the DINO
    reward has no target here).

`--selector bench` trains on argmax bench_score, `--selector random` on random_idx. Also prints the
selection headroom per category (mean of the argmax vs the mean candidate): the number that says how
much best-of-N by the evaluator can move a student trained on the picks.

    python data/build_bench_selection.py --scores cache/bench/scores --label bench_teacher_k10 --K 10 --N 16 --out cache/bench_k10_n16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pools/bench_train/compbench_prompts.json")
    ap.add_argument("--scores", default="cache/bench/scores")
    ap.add_argument("--label", default="bench_teacher_k10")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--N", type=int, default=16); ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--out", default="cache/bench_k10_n16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pool = {int(r["idx"]): r for r in json.load(open(args.pool))}
    scores = {}
    for c in CATS:
        f = f"{args.scores}/{args.label}_s{args.K}_{c}/scores.json"
        assert os.path.exists(f), f"missing {f}"
        for r in json.load(open(f))["per_prompt"]:
            s = [float(x) for x in r["image_scores"]]
            assert len(s) == args.N, (c, r["idx"], len(s))
            scores[int(r["idx"])] = s
    missing = [i for i in pool if i not in scores]
    print(f"scored {len(scores)}/{len(pool)} prompts; missing {len(missing)}")
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in sorted(scores):
        s = scores[i]
        rows.append({"idx": i, "category": pool[i]["category"], "prompt": pool[i]["prompt"], "reference": "",
                     "seed_base": args.seed + i * 1000, "N": args.N, "j_start": 0,
                     "bench_score": s, "bench_argmax_idx": int(np.argmax(s)), "random_idx": int(rng.integers(args.N))})
    for k in range(4):
        with open(out / f"selection_rank{k}.jsonl", "w") as fh:
            for n, r in enumerate(rows):
                if n % 4 == k:
                    fh.write(json.dumps(r) + "\n")
    verdict = {}
    print(f"{'category':12s} {'n':>4s} {'mean cand':>9s} {'random':>7s} {'argmax':>7s} {'headroom':>8s}")
    for c in CATS:
        S = np.array([r["bench_score"] for r in rows if r["category"] == c])
        if not len(S):
            continue
        rnd = np.array([r["bench_score"][r["random_idx"]] for r in rows if r["category"] == c])
        verdict[c] = {"n": int(len(S)), "mean": float(S.mean()), "random": float(rnd.mean()), "argmax": float(S.max(1).mean())}
        print(f"{c:12s} {len(S):4d} {S.mean():9.4f} {rnd.mean():7.4f} {S.max(1).mean():7.4f} {S.max(1).mean() - S.mean():+8.4f}")
    json.dump({"K": args.K, "cfg": args.cfg, "N": args.N, "n_prompts": len(rows), "pool": args.pool, "scores": args.scores,
               "scorer": "official T2I-CompBench++ evaluator of each prompt's category (eval/compbench.py)", "headroom": verdict},
              open(out / "cache_meta.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
