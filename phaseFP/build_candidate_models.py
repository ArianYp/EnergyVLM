#!/usr/bin/env python3
"""Lay out each teacher candidate slot as its own pseudo-model, for exact policy expectations.

Scoring a *single random draw* per policy makes `Uniform` and `NonOracle` noisy estimates, and the
mixture identity `Uniform = (1/N)·Best + (1-1/N)·NonOracle` then fails by roughly one standard
error. That is what inflated the reported control bias to ~42% when the exact value is N/(N-1) =
33.3%.

Scoring every candidate slot instead lets both controls be computed as exact per-prompt
expectations, so the identity holds by construction and the inflation factor stops being an
empirical quantity at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True,
                        help="<gen>/images/Teacher holding p{idx:05d}/s{steps}/cand{j}.png")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    args = parser.parse_args()

    prompts = {int(p["idx"]): p for p in json.loads(Path(args.prompts).read_text())}
    images_root = Path(args.images_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept, made = [], {j: 0 for j in range(args.N)}
    for idx in sorted(prompts):
        srcs = [images_root / f"p{idx:05d}" / f"s{args.steps}" / f"cand{j}.png"
                for j in range(args.N)]
        if not all(s.exists() for s in srcs):
            continue
        for j, src in enumerate(srcs):
            # Each slot becomes a model whose single image is named cand0.png, which is the layout
            # the CompBench harness expects.
            dst_dir = out / "images" / f"Cand{j}" / f"p{idx:05d}" / f"s{args.steps}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "cand0.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
            made[j] += 1
        kept.append(idx)

    (out / "prompts.json").write_text(json.dumps(
        [prompts[i] for i in kept], indent=1))
    summary = {"prompts": len(kept), "N": args.N, "steps": args.steps,
               "symlinks_per_slot": made}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
