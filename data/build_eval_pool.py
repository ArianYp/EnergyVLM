#!/usr/bin/env python3
"""Evaluation prompt pools, written once and hashed.

    compbench   T2I-CompBench prompts, stratified over its 8 categories from the requested split
                (default: the held-out val split, with any prompt that also appears in train
                removed so val and train are disjoint).
    geneval2    GenEval2/geneval2_data.jsonl in file order. The order matters: GenEval2's
                evaluation.py emits one score list per line with no key, so idx == line number
                is what joins per-prompt scores back.
    fidelity    COCO val2017 captions, one per image, for FID / CMMD against the real val2017
                photographs.

Every pool is `[{idx, category, prompt}, ...]`, the shape eval/generate.py consumes, and
manifest.json records a sha256 per file so two evaluation jobs can assert they generated from the
same prompts.

    python data/build_eval_pool.py --out_root pools/eval \
        --compbench_dir third_party/T2I-CompBench/examples/dataset \
        --geneval2_data third_party/GenEval2/geneval2_data.jsonl \
        --coco_captions /path/to/COCO/annotations/captions_val2017.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

CATEGORIES = ["color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "complex", "numeracy"]


def compbench_pool(dataset_dir: Path, per_category: int, seed: int, split: str) -> list[dict]:
    exclude: set[str] = set()
    if split != "train":
        for cat in CATEGORIES:
            f = dataset_dir / f"{cat}_train.txt"
            exclude.update(ln.strip() for ln in f.read_text().splitlines() if ln.strip())
    rng = random.Random(seed)
    pool: list[dict] = []
    for cat in CATEGORIES:
        lines = [ln.strip() for ln in (dataset_dir / f"{cat}_{split}.txt").read_text().splitlines() if ln.strip()]
        lines = [p for p in dict.fromkeys(lines) if p not in exclude]
        for p in rng.sample(lines, min(per_category, len(lines))):
            pool.append({"idx": len(pool), "category": cat, "prompt": p})
    return pool


def geneval2_pool(data: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in data.read_text().splitlines() if ln.strip()]
    pool = [{"idx": i, "category": "geneval2", "prompt": r["prompt"],
             "skills": sorted(set(r.get("skills", []))), "atom_count": r.get("atom_count")}
            for i, r in enumerate(rows)]
    if len(pool) != len({p["prompt"] for p in pool}):
        raise SystemExit("duplicate GenEval2 prompts; evaluation.py keys images by prompt text")
    return pool


def coco_pool(captions: Path, n: int, seed: int) -> list[dict]:
    ann = json.loads(captions.read_text())
    first: dict[int, tuple[int, str]] = {}
    for a in ann["annotations"]:
        cur = first.get(a["image_id"])
        if cur is None or a["id"] < cur[0]:
            first[a["image_id"]] = (a["id"], a["caption"].strip().replace("\n", " "))
    items = [(img_id, cap) for img_id, (_, cap) in sorted(first.items())]
    if n and n < len(items):
        items = sorted(random.Random(seed).sample(items, n))
    return [{"idx": i, "category": "coco", "prompt": cap, "coco_image_id": img_id}
            for i, (img_id, cap) in enumerate(items)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--compbench_dir", required=True)
    ap.add_argument("--geneval2_data", required=True)
    ap.add_argument("--coco_captions", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--per_category", type=int, default=1000)
    ap.add_argument("--fidelity_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_root)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    pools = {
        "compbench_prompts.json": compbench_pool(Path(args.compbench_dir), args.per_category, args.seed, args.split),
        "geneval2_prompts.json": geneval2_pool(Path(args.geneval2_data)),
        "fidelity_prompts.json": coco_pool(Path(args.coco_captions), args.fidelity_n, args.seed),
    }
    hashes = {}
    for name, pool in pools.items():
        (out / name).write_text(json.dumps(pool, indent=1))
        hashes[name] = sha256(out / name)
    manifest = {"configuration": vars(args), "sha256": hashes,
                "counts": {k: len(v) for k, v in pools.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
