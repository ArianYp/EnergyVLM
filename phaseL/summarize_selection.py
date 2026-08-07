#!/usr/bin/env python3
"""Validate a Phase-L selection cache and publish its immutable W&B record."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseI.train_preference import load_selection


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected_records", type=int, required=True)
    parser.add_argument("--expected_repeats", type=int, required=True)
    parser.add_argument("--project", default="sd-pref-repa-publication-v1")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    records = load_selection(root)
    if len(records) != args.expected_records:
        raise AssertionError((len(records), args.expected_records))
    if {record.get("label_source") for record in records} != {"correct_prompt"}:
        raise AssertionError("all publication labels must come from the correct prompt")

    repeats = Counter(int(record["seed_repeat"]) for record in records)
    expected_repeat_ids = set(range(args.expected_repeats))
    if set(repeats) != expected_repeat_ids:
        raise AssertionError((sorted(repeats), sorted(expected_repeat_ids)))
    by_source: dict[int, set[int]] = defaultdict(set)
    for record in records:
        by_source[int(record["source_idx"])].add(int(record["seed_repeat"]))
    incomplete = [source for source, values in by_source.items() if values != expected_repeat_ids]
    if incomplete:
        raise AssertionError(f"{len(incomplete)} source prompts lack complete repeat coverage")

    candidate_counts = {len(record["endpoint_vqa"]) for record in records}
    if len(candidate_counts) != 1:
        raise AssertionError(f"candidate-count mismatch: {sorted(candidate_counts)}")
    seed_bases = [int(record["seed_base"]) for record in records]
    if len(seed_bases) != len(set(seed_bases)):
        raise AssertionError("seed bases are not unique")
    margins = [
        max(map(float, record["endpoint_vqa"])) - min(map(float, record["endpoint_vqa"]))
        for record in records
    ]
    if statistics.fmean(margins) <= 0.20:
        raise AssertionError("mean best-to-worst VQAScore headroom is below 0.20")

    shard_paths = sorted(root.glob("selection_rank*.jsonl"))
    summary = {
        "pass": True,
        "records": len(records),
        "source_prompts": len(by_source),
        "repeat_count": args.expected_repeats,
        "records_per_repeat": {str(key): value for key, value in sorted(repeats.items())},
        "candidate_count": next(iter(candidate_counts)),
        "category_records": dict(sorted(Counter(
            str(record.get("category", "unknown")) for record in records
        ).items())),
        "mean_top_bottom_margin": statistics.fmean(margins),
        "median_top_bottom_margin": statistics.median(margins),
        "positive_margin_fraction": sum(value > 0 for value in margins) / len(margins),
        "selection_files_sha256": {path.name: sha256(path) for path in shard_paths},
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.no_wandb:
        return
    import wandb

    run = wandb.init(
        project=args.project, name=args.run_name, id=args.run_name,
        group=args.group, job_type="publication-selection-cache",
        tags=["phaseL", "publication", "selection", "preference", "correct-prompt"],
        config={"root": str(root), **summary}, dir=str(root), resume="never",
    )
    wandb.log({
        "selection/records": summary["records"],
        "selection/source_prompts": summary["source_prompts"],
        "selection/repeats": summary["repeat_count"],
        "selection/candidate_count": summary["candidate_count"],
        "selection/mean_top_bottom_margin": summary["mean_top_bottom_margin"],
        "selection/median_top_bottom_margin": summary["median_top_bottom_margin"],
        "selection/positive_margin_fraction": summary["positive_margin_fraction"],
        "selection/margin_histogram": wandb.Histogram(margins),
    }, step=0)

    table = wandb.Table(columns=[
        "repeat", "category", "prompt", "candidate", "score", "preferred", "image"
    ])
    for metadata_path in sorted((root / "samples").glob("repeat_*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        repeat = int(metadata_path.parent.parent.name.split("_")[-1])
        for candidate_idx, score in enumerate(metadata["scores"]):
            image_path = metadata_path.parent / f"cand{candidate_idx}.png"
            table.add_data(
                repeat, metadata["category"], metadata["prompt"], candidate_idx,
                float(score), candidate_idx == int(metadata["oracle_idx"]),
                wandb.Image(str(image_path), caption=(
                    f"{metadata['category']} | candidate {candidate_idx} | score {float(score):.4f}"
                )),
            )
    wandb.log({"samples/scored_teacher_candidates": table}, step=0)

    artifact = wandb.Artifact(f"{args.run_name}-manifest", type="selection-cache")
    artifact.add_file(str(summary_path), name="summary.json")
    artifact.add_file(str(root / "prompts.json"), name="prompts.json")
    for path in shard_paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    for key, value in summary.items():
        if isinstance(value, (bool, int, float, str)):
            run.summary[key] = value
    run.summary["selection_root"] = str(root)
    wandb.finish()


if __name__ == "__main__":
    main()
