#!/usr/bin/env python3
"""Manifest for the latent-scorer study: train / val captions from one candidate cache (never the
captions of the cache the scorer will be used on) and test = the target cache itself.

    python3 data/build_latent_manifest.py --train_cache cache/train --test_cache cache/train_3k \
        --out cache/latents/manifest.jsonl --n_train 22000 --n_val 2000
"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--train_cache", required=True)
ap.add_argument("--test_cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, default=22000)
ap.add_argument("--n_val", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20260907)
args = ap.parse_args()
excl, test = set(), []
for f in sorted(Path(args.test_cache).glob("selection_rank*.jsonl")):
    for ln in f.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln); excl.add(r["prompt"]); r["split"] = "test"; test.append(r)
pool = []
for f in sorted(Path(args.train_cache).glob("selection_rank*.jsonl")):
    for ln in f.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["prompt"] not in excl and "dino_patch_cos" in r:
                pool.append(r)
pool.sort(key=lambda r: r["idx"])
random.Random(args.seed).shuffle(pool)
sel = pool[:args.n_train + args.n_val]
for i, r in enumerate(sel):
    r["split"] = "train" if i < args.n_train else "val"
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as fh:
    for r in sel + test:
        fh.write(json.dumps(r) + "\n")
print(f"train {min(args.n_train, len(sel))} val {max(0, len(sel) - args.n_train)} test {len(test)}")
