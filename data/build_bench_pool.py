#!/usr/bin/env python3
"""Benchmark-prompt training pool (the GORS / CTCal data protocol, docs/bench/): the T2I-CompBench++
TRAIN prompts of all 8 categories (~700 each), minus any text that appears in the held-out evaluation
pool (pools/eval/compbench_prompts.json, the val split) and minus duplicates.

Writes <out_root>/compbench_prompts.json as [{idx, category, prompt}] with a global idx -- the shape
eval/generate.py, eval/compbench.py and data/build_bench_candidates.py consume -- plus manifest.json.
The idx numbering is this pool's own: it does NOT coincide with the evaluation pool's, so never score
one pool's images against the other's prompt file.

    python data/build_bench_pool.py --compbench_dir third_party/T2I-CompBench/examples/dataset \
        --exclude pools/eval/compbench_prompts.json --out_root pools/bench_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compbench_dir", default="third_party/T2I-CompBench/examples/dataset")
    ap.add_argument("--exclude", default="pools/eval/compbench_prompts.json",
                    help="prompt pool whose texts must not appear (the held-out evaluation pool)")
    ap.add_argument("--out_root", default="pools/bench_train")
    args = ap.parse_args()
    ds = Path(args.compbench_dir)
    held = {r["prompt"].strip().lower() for r in json.load(open(args.exclude))}
    pool, dropped, seen = [], {}, set()
    for c in CATS:
        n0 = 0
        for line in open(ds / f"{c}_train.txt"):
            p = line.strip()
            if not p:
                continue
            n0 += 1
            key = p.lower()
            if key in held or key in seen:
                dropped[c] = dropped.get(c, 0) + 1
                continue
            seen.add(key)
            pool.append({"idx": len(pool), "category": c, "prompt": p})
        print(f"{c:12s} train {n0:4d} -> kept {sum(r['category'] == c for r in pool):4d} "
              f"(dropped {dropped.get(c, 0)}: in the eval pool or duplicate)")
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(pool, open(out / "compbench_prompts.json", "w"), indent=1)
    json.dump({"n": len(pool), "per_category": {c: sum(r["category"] == c for r in pool) for c in CATS}, "dropped": dropped,
               "source": f"{ds}/<cat>_train.txt", "excluded": str(args.exclude)},
              open(out / "manifest.json", "w"), indent=1)
    print(f"wrote {len(pool)} prompts -> {out}/compbench_prompts.json")


if __name__ == "__main__":
    main()
