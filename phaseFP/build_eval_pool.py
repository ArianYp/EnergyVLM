#!/usr/bin/env python3
"""Freeze every Phase-FP evaluation prompt manifest once, and hash it.

The Phase-I fidelity confusion happened because two evaluation jobs silently generated from
different prompt pools. Building all four manifests here, hashing them, and requiring each
per-model job to assert the hash makes "identical evaluation conditions" a checked precondition
rather than a convention.

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp0"))

from generate_candidates import build_dev_pool, train_prompt_set  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(payload, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--compbench_dir", default="T2I-CompBench/examples/dataset")
    parser.add_argument("--per_category", type=int, default=1000)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fidelity_n", type=int, default=2398)
    parser.add_argument("--diversity_per_category", type=int, default=50)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite {out_root}")
    out_root.mkdir(parents=True)

    dataset = Path(args.compbench_dir)
    exclude = train_prompt_set(dataset) if args.split != "train" else None
    compbench = build_dev_pool(
        dataset, args.per_category, args.seed, split=args.split, exclude=exclude
    )
    hashes = {"compbench_prompts.json": write_json(compbench, out_root / "compbench_prompts.json")}

    # Reuse the audited pool builders for the other three so there is exactly one code path.
    from phaseC.build_eval_pool import main as pool_main  # noqa: F401  (import proves availability)
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "phaseC" / "build_eval_pool.py"), "geneval2",
         "--out", str(out_root / "geneval2_prompts.json")],
        check=True, cwd=ROOT,
    )
    hashes["geneval2_prompts.json"] = sha256(out_root / "geneval2_prompts.json")

    subprocess.run(
        [sys.executable, str(ROOT / "phaseC" / "build_eval_pool.py"), "coco",
         "--n", str(args.fidelity_n), "--seed", str(args.seed),
         "--out", str(out_root / "fidelity_prompts.json")],
        check=True, cwd=ROOT,
    )
    hashes["fidelity_prompts.json"] = sha256(out_root / "fidelity_prompts.json")

    subprocess.run(
        [sys.executable, str(ROOT / "phaseC" / "build_eval_pool.py"), "subset",
         "--pool", str(out_root / "compbench_prompts.json"),
         "--per_category", str(args.diversity_per_category), "--seed", str(args.seed),
         "--out", str(out_root / "diversity_prompts.json")],
        check=True, cwd=ROOT,
    )
    hashes["diversity_prompts.json"] = sha256(out_root / "diversity_prompts.json")

    counts = {
        name: len(json.loads((out_root / name).read_text()))
        for name in hashes
    }
    manifest = {
        "out_root": str(out_root.resolve()),
        "configuration": vars(args),
        "sha256": hashes,
        "counts": counts,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
