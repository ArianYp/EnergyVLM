#!/usr/bin/env python3
"""Combine three-seed alignment, fidelity, and diversity into one publication verdict."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def describe(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_root", required=True)
    parser.add_argument("--fidelity_report", required=True)
    parser.add_argument("--diversity_report", required=True)
    parser.add_argument("--treatments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project", default="sd-pref-repa-publication-v1")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--train_jobs", required=True)
    args = parser.parse_args()

    alignment_root = Path(args.alignment_root).resolve()
    treatments = args.treatments.split(",")
    if len(treatments) != 3:
        raise ValueError("publication verdict requires exactly three independently trained seeds")
    fidelity = read_json(Path(args.fidelity_report).resolve())
    diversity = read_json(Path(args.diversity_report).resolve())

    per_seed = {}
    compbench_deltas = []
    geneval_deltas = []
    pilot_cmmd = float(fidelity["results"]["PilotPreferenceREPA@4"]["cmmd"])
    pilot_dino = float(diversity["levels"]["PilotPreferenceREPA@4"]["dino"])
    pilot_lpips = float(diversity["levels"]["PilotPreferenceREPA@4"]["lpips"])
    for model in treatments:
        contrast = read_json(alignment_root / f"{model}_vs_PilotPreferenceREPA.json")
        compbench_delta = float(contrast["primary"]["delta"])
        geneval_delta = float(contrast["geneval2"]["delta"])
        compbench_deltas.append(compbench_delta)
        geneval_deltas.append(geneval_delta)
        fidelity_level = fidelity["results"][f"{model}@4"]
        diversity_level = diversity["levels"][f"{model}@4"]
        per_seed[model] = {
            "compbench_delta_vs_pilot": compbench_delta,
            "compbench_prompt_bootstrap_ci": list(map(float, contrast["primary"]["ci"])),
            "geneval2_delta_vs_pilot": geneval_delta,
            "geneval2_prompt_bootstrap_ci": list(map(float, contrast["geneval2"]["ci"])),
            "fid": float(fidelity_level["fid"]),
            "cmmd": float(fidelity_level["cmmd"]),
            "cmmd_delta_vs_pilot": float(fidelity_level["cmmd"]) - pilot_cmmd,
            "precision": float(fidelity_level["precision"]),
            "recall": float(fidelity_level["recall"]),
            "dino_diversity": float(diversity_level["dino"]),
            "dino_delta_vs_pilot": float(diversity_level["dino"]) - pilot_dino,
            "lpips_diversity": float(diversity_level["lpips"]),
            "lpips_delta_vs_pilot": float(diversity_level["lpips"]) - pilot_lpips,
        }

    compbench = describe(compbench_deltas)
    geneval = describe(geneval_deltas)
    all_alignment_positive = min(compbench_deltas) > 0 and min(geneval_deltas) > 0
    fidelity_noninferior = all(
        values["cmmd_delta_vs_pilot"] <= 2.0 for values in per_seed.values()
    )
    diversity_noninferior = all(
        values["dino_delta_vs_pilot"] >= -0.02
        and values["lpips_delta_vs_pilot"] >= -0.02
        for values in per_seed.values()
    )
    publication_pass = all((all_alignment_positive, fidelity_noninferior, diversity_noninferior))

    summary = {
        "pass": publication_pass,
        "decision_rule": (
            "all three seeds improve both held-out alignment benchmarks over the pilot; "
            "each seed has CMMD <= pilot + 2 and DINO/LPIPS diversity >= pilot - 0.02"
        ),
        "treatments": treatments,
        "per_seed": per_seed,
        "across_training_seeds": {
            "compbench_delta_vs_pilot": compbench,
            "geneval2_delta_vs_pilot": geneval,
        },
        "gates": {
            "all_alignment_positive": all_alignment_positive,
            "fidelity_noninferior": fidelity_noninferior,
            "diversity_noninferior": diversity_noninferior,
        },
        "caveat": (
            "Prompt-bootstrap intervals quantify held-out prompt uncertainty within each seed; "
            "the across-seed standard deviation is reported separately and n=3 is not treated "
            "as a precise confidence interval over training randomness."
        ),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase L publication-scale verdict\n\n",
        f"**Overall gate: {'PASS' if publication_pass else 'FAIL'}**\n\n",
        "| seed model | CompBench delta vs pilot | 95% prompt CI | GenEval2 delta vs pilot | 95% prompt CI | CMMD delta | DINO diversity delta | LPIPS diversity delta |\n",
        "|---|--:|---:|--:|---:|--:|--:|--:|\n",
    ]
    for model in treatments:
        values = per_seed[model]
        lines.append(
            f"| {model} | {values['compbench_delta_vs_pilot']:+.5f} | "
            f"[{values['compbench_prompt_bootstrap_ci'][0]:+.5f}, {values['compbench_prompt_bootstrap_ci'][1]:+.5f}] | "
            f"{values['geneval2_delta_vs_pilot']:+.5f} | "
            f"[{values['geneval2_prompt_bootstrap_ci'][0]:+.5f}, {values['geneval2_prompt_bootstrap_ci'][1]:+.5f}] | "
            f"{values['cmmd_delta_vs_pilot']:+.3f} | {values['dino_delta_vs_pilot']:+.4f} | "
            f"{values['lpips_delta_vs_pilot']:+.4f} |\n"
        )
    lines.extend([
        "\n## Across training seeds\n\n",
        f"- CompBench delta: {compbench['mean']:+.5f} ± {compbench['std']:.5f}; "
        f"range [{compbench['min']:+.5f}, {compbench['max']:+.5f}].\n",
        f"- GenEval2 delta: {geneval['mean']:+.5f} ± {geneval['std']:.5f}; "
        f"range [{geneval['min']:+.5f}, {geneval['max']:+.5f}].\n",
        f"- Fidelity non-inferiority: {'PASS' if fidelity_noninferior else 'FAIL'}.\n",
        f"- Diversity non-inferiority: {'PASS' if diversity_noninferior else 'FAIL'}.\n",
        f"\n{summary['caveat']}\n",
    ])
    output.write_text("".join(lines))
    print("".join(lines))

    import wandb

    run = wandb.init(
        project=args.project, name=args.run_name, id=args.run_name,
        group=args.group, job_type="publication-final-verdict",
        tags=["phaseL", "publication", "verdict", "three-seed"],
        config={
            "alignment_root": str(alignment_root), "treatments": treatments,
            "train_jobs": args.train_jobs, "decision_rule": summary["decision_rule"],
        }, dir=str(output.parent), resume="never",
    )
    metrics = {
        "verdict/pass": float(publication_pass),
        "verdict/all_alignment_positive": float(all_alignment_positive),
        "verdict/fidelity_noninferior": float(fidelity_noninferior),
        "verdict/diversity_noninferior": float(diversity_noninferior),
        "replication/compbench_delta_mean": compbench["mean"],
        "replication/compbench_delta_std": compbench["std"],
        "replication/geneval2_delta_mean": geneval["mean"],
        "replication/geneval2_delta_std": geneval["std"],
    }
    for model, values in per_seed.items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                metrics[f"seed/{model}/{key}"] = float(value)
    wandb.log(metrics, step=0)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f"{args.run_name}-report", type="evaluation")
    artifact.add_file(str(output), name=output.name)
    artifact.add_file(str(output.with_suffix(".json")), name=output.with_suffix(".json").name)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
