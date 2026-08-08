#!/usr/bin/env python3
"""Task T, stage 2 — score frozen-teacher candidates on the HELD-OUT val prompts with VQAScore.

The existing teacher oracle (+0.0717) was measured on the exp0 pool, which is the **train** split
and shares zero prompts with the split the students are evaluated on. That mismatch is why the
amortization ratio has no matched denominator and why the oracle diversity price is not a matched
contrast. This re-measures the ceiling on exactly the prompts the students are scored on.

Sharded like `exp0/score_candidates.py`: rank r takes prompts[r::world], appends to its own JSONL,
and skips already-scored keys, so the stage is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True,
                        help="<gen_root>/images/<LABEL>, holding p{idx:05d}/s{steps}/cand{j}.png")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument("--vqa_model", default="clip-flant5-xxl")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    import torch

    torch.cuda.set_device(local_rank)

    prompts = json.loads(Path(args.prompts).read_text())
    mine = [p for i, p in enumerate(prompts) if i % world == rank]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"scores_rank{rank}.jsonl"
    done = set()
    if shard.exists():
        for line in shard.read_text().splitlines():
            if line.strip():
                try:
                    done.add(int(json.loads(line)["idx"]))
                except Exception:
                    pass

    # t2v_metrics needs the vendored path and the compat stub, in this order.
    sys.path.insert(0, str(ROOT / "t2v_metrics"))
    sys.path.insert(0, str(ROOT / "exp0"))
    import _t2v_compat  # noqa: F401
    import t2v_metrics

    # t2v_metrics' own HF_CACHE_DIR constant is a local "./hf_cache/", so the real hub cache must be
    # passed explicitly or it re-downloads clip-flant5-xxl instead of using the prefetched snapshot.
    # It wants the HUB subdirectory, not the HF_HOME root — matching exp0/score_candidates.py.
    hub_cache = os.path.join(
        os.environ.get("HF_HOME", str(ROOT / "cache/huggingface")), "hub"
    )
    print(f"[r{rank}] loading VQAScore {args.vqa_model} (cache: {hub_cache})", flush=True)
    scorer = t2v_metrics.VQAScore(
        model=args.vqa_model, device=f"cuda:{local_rank}", cache_dir=hub_cache
    )

    images_root = Path(args.images_root)
    written = 0
    with open(shard, "a") as handle:
        for record in mine:
            idx = int(record["idx"])
            if idx in done:
                continue
            paths = [
                images_root / f"p{idx:05d}" / f"s{args.steps}" / f"cand{j}.png"
                for j in range(args.pool_size)
            ]
            if not all(p.exists() for p in paths):
                continue
            # Same extraction as exp0/score_candidates.py: the scorer returns [n_images, n_texts],
            # so squeeze the single text dimension. no_grad keeps it from building a graph.
            with torch.no_grad():
                vqa = (
                    scorer(images=[str(p) for p in paths], texts=[record["prompt"]])
                    .squeeze(1).float().cpu().tolist()
                )
            handle.write(json.dumps({
                "idx": idx,
                "category": record.get("category"),
                "prompt": record["prompt"],
                "steps": args.steps,
                # `config` doubles as the image subdirectory name, so audit/oracle_diversity_price.py
                # can read this bank unmodified: <images_root>/p{idx:05d}/{config}/cand{j}.png.
                "config": f"s{args.steps}",
                "vqa": vqa,
            }) + "\n")
            written += 1
            if written % 50 == 0:
                handle.flush()
                print(f"[r{rank}] {written}/{len(mine)} scored", flush=True)
    print(f"[r{rank}] done: {written} newly scored, {len(done)} already present", flush=True)


if __name__ == "__main__":
    main()
