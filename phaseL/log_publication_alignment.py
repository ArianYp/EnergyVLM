#!/usr/bin/env python3
"""Log full held-out alignment results and paired sample panels to W&B."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CATEGORIES = [
    "color", "shape", "texture", "spatial", "3d_spatial", "numeracy",
    "non_spatial", "complex",
]
PRIMARY_CATEGORIES = [
    "color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex",
]


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--treatments", required=True)
    parser.add_argument("--project", default="sd-pref-repa-publication-v1")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--train_jobs", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    models = args.models.split(",")
    treatments = args.treatments.split(",")
    metrics: dict[str, float] = {}
    for model in models:
        category_means = []
        for category in CATEGORIES:
            score = read_json(root / "compbench_scores" / f"{model}_s4_{category}" / "scores.json")
            value = float(score["mean"])
            metrics[f"compbench/{model}/{category}"] = value
            if category in PRIMARY_CATEGORIES:
                category_means.append(value)
        metrics[f"compbench/{model}/primary_equal_category"] = statistics.fmean(category_means)
        geneval = read_json(root / "geneval2_scores" / f"{model}_s4" / "scores.json")
        metrics[f"geneval2/{model}/mean"] = float(geneval["mean"])
        for skill, values in geneval.get("per_skill", {}).items():
            metrics[f"geneval2/{model}/skill/{skill}"] = float(values["mean"])

    for treatment in treatments:
        for control in ("PilotPreferenceREPA", "M1"):
            contrast = read_json(root / f"{treatment}_vs_{control}.json")
            prefix = f"contrast/{treatment}_vs_{control}"
            metrics[f"{prefix}/compbench_delta"] = float(contrast["primary"]["delta"])
            metrics[f"{prefix}/compbench_ci_low"] = float(contrast["primary"]["ci"][0])
            metrics[f"{prefix}/compbench_ci_high"] = float(contrast["primary"]["ci"][1])
            metrics[f"{prefix}/geneval2_delta"] = float(contrast["geneval2"]["delta"])
            metrics[f"{prefix}/geneval2_ci_low"] = float(contrast["geneval2"]["ci"][0])
            metrics[f"{prefix}/geneval2_ci_high"] = float(contrast["geneval2"]["ci"][1])

    compbench_deltas = [
        metrics[f"contrast/{model}_vs_PilotPreferenceREPA/compbench_delta"]
        for model in treatments
    ]
    geneval_deltas = [
        metrics[f"contrast/{model}_vs_PilotPreferenceREPA/geneval2_delta"]
        for model in treatments
    ]
    metrics["replication/compbench_delta_mean"] = statistics.fmean(compbench_deltas)
    metrics["replication/compbench_delta_min"] = min(compbench_deltas)
    metrics["replication/compbench_delta_max"] = max(compbench_deltas)
    metrics["replication/compbench_delta_std"] = statistics.stdev(compbench_deltas)
    metrics["replication/geneval2_delta_mean"] = statistics.fmean(geneval_deltas)
    metrics["replication/geneval2_delta_min"] = min(geneval_deltas)
    metrics["replication/geneval2_delta_max"] = max(geneval_deltas)
    metrics["replication/geneval2_delta_std"] = statistics.stdev(geneval_deltas)

    import wandb

    run = wandb.init(
        project=args.project, name=args.run_name, id=args.run_name,
        group=args.group, job_type="publication-alignment-evaluation",
        tags=["phaseL", "publication", "evaluation", "compbench", "geneval2", "three-seed"],
        config={
            "root": str(root), "models": models, "treatments": treatments,
            "train_jobs": args.train_jobs, "steps": 4,
        }, dir=str(root), resume="never",
    )
    wandb.log(metrics, step=0)
    for key, value in metrics.items():
        run.summary[key] = value

    prompts = read_json(root / "compbench" / "prompts.json")
    selected = []
    seen = set()
    for item in prompts:
        if item["category"] in seen:
            continue
        seen.add(item["category"])
        selected.append(item)
    table = wandb.Table(columns=["idx", "category", "prompt", *models])
    for item in selected:
        row = [int(item["idx"]), item["category"], item["prompt"]]
        for model in models:
            image = (
                root / "compbench" / "images" / model
                / f"p{int(item['idx']):05d}" / "s4" / "cand0.png"
            )
            if not image.is_file():
                raise FileNotFoundError(image)
            row.append(wandb.Image(str(image), caption=f"{model} | {item['prompt']}"))
        table.add_data(*row)
    wandb.log({"samples/paired_benchmark_panel": table}, step=0)

    artifact = wandb.Artifact(f"{args.run_name}-reports", type="evaluation")
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.md")):
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.summary["evaluation_root"] = str(root)
    wandb.finish()


if __name__ == "__main__":
    main()
