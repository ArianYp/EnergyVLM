#!/usr/bin/env python3
"""Training caption pool: COCO captions paired with the photograph each was written for.

One caption per image (the first in annotation order), every reference file checked on disk, and
duplicate caption text dropped so a caption maps to exactly one reference. `idx` is the position
in the sorted image-id list and stays fixed for the life of the pool: the candidate cache derives
each caption's noise seeds from it.

    python data/build_pool.py --coco_root /path/to/COCO --split train2017 \
        --out_prompts pools/train/prompts.json --out_manifest pools/train/pool_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True, help="directory holding annotations/ and <split>/")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--n", type=int, default=0, help="subsample to this many captions; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="another pool_manifest.json whose captions and image ids must not reappear")
    ap.add_argument("--out_prompts", required=True)
    ap.add_argument("--out_manifest", required=True)
    args = ap.parse_args()

    root = Path(args.coco_root)
    blob = json.loads((root / "annotations" / f"captions_{args.split}.json").read_text())
    first: dict[int, str] = {}
    for record in blob["annotations"]:
        first.setdefault(int(record["image_id"]), record["caption"].strip())
    rows = [{"idx": i, "prompt": caption, "coco_image_id": image_id}
            for i, (image_id, caption) in enumerate(sorted(first.items()))]

    if args.exclude:
        blocked = json.loads(Path(args.exclude).read_text())
        bad_text = {r["prompt"] for r in blocked}
        bad_ids = {int(r["coco_image_id"]) for r in blocked}
        before = len(rows)
        rows = [r for r in rows if r["prompt"] not in bad_text and int(r["coco_image_id"]) not in bad_ids]
        print(f"excluded {before - len(rows)} rows present in {args.exclude}")

    images = root / args.split
    usable, missing, seen, dup = [], 0, set(), 0
    for row in rows:
        ref = images / f"{int(row['coco_image_id']):012d}.jpg"
        if not ref.exists():
            missing += 1
            continue
        if row["prompt"] in seen:
            dup += 1
            continue
        seen.add(row["prompt"])
        usable.append({**row, "reference": str(ref.resolve())})

    if args.n and args.n < len(usable):
        order = np.random.default_rng(args.seed).permutation(len(usable))[: args.n]
        usable = [usable[i] for i in sorted(order)]

    prompts = [{"idx": r["idx"], "category": "coco", "prompt": r["prompt"]} for r in usable]
    manifest = [{"idx": r["idx"], "prompt": r["prompt"], "coco_image_id": r["coco_image_id"],
                 "reference": r["reference"]} for r in usable]
    for p in (args.out_prompts, args.out_manifest):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_prompts).write_text(json.dumps(prompts, indent=1))
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"selected": len(usable), "missing_reference_files": missing,
                      "dropped_duplicate_captions": dup}, indent=2))


if __name__ == "__main__":
    main()
