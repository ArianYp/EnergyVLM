#!/usr/bin/env python3
"""Log publication fidelity or diversity JSON without duplicating large per-prompt payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key == "per_prompt":
                continue
            flatten(f"{prefix}/{key}" if prefix else str(key), child, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--kind", choices=["fidelity", "diversity"], required=True)
    parser.add_argument("--project", default="sd-pref-repa-publication-v1")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--train_jobs", required=True)
    args = parser.parse_args()

    report = Path(args.report).resolve()
    payload = json.loads(report.read_text())
    metrics: dict[str, float] = {}
    flatten(args.kind, payload, metrics)

    import wandb

    run = wandb.init(
        project=args.project, name=args.run_name, id=args.run_name,
        group=args.group, job_type=f"publication-{args.kind}-evaluation",
        tags=["phaseL", "publication", "evaluation", args.kind, "three-seed"],
        config={"report": str(report), "train_jobs": args.train_jobs},
        dir=str(report.parent), resume="never",
    )
    wandb.log(metrics, step=0)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f"{args.run_name}-report", type="evaluation")
    artifact.add_file(str(report), name=report.name)
    markdown = report.with_suffix(".md")
    if markdown.is_file():
        artifact.add_file(str(markdown), name=markdown.name)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
