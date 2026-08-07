#!/usr/bin/env python3
"""Evaluate the predeclared gate before spending publication-scale training compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project", default="sd-pref-repa-publication-v1")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    root = Path(args.eval_root).resolve()
    contrast = read_json(root / "correct_vs_counterfactual.json")
    fidelity = read_json(root / "fidelity.json")
    diversity = read_json(root / "diversity.json")

    compbench = contrast["primary"]
    geneval = contrast["geneval2"]
    fidelity_gate = fidelity["gate"]["4"]
    diversity_gate = diversity["gate"]["4"]

    compbench_specific = float(compbench["ci"][0]) > 0
    geneval_direction = float(geneval["delta"]) > 0
    fidelity_noninferior = bool(fidelity_gate["pass"])
    diversity_noninferior = all(
        "FAIL" not in str(values["verdict"])
        for values in diversity_gate.values()
    )
    promotion = all((
        compbench_specific,
        geneval_direction,
        fidelity_noninferior,
        diversity_noninferior,
    ))

    summary = {
        "pass": promotion,
        "rule": (
            "CompBench paired CI lower bound > 0; GenEval2 delta > 0; "
            "matched fidelity and both diversity gates non-inferior"
        ),
        "eval_root": str(root),
        "compbench": {
            "delta": float(compbench["delta"]),
            "ci": list(map(float, compbench["ci"])),
            "pass": compbench_specific,
        },
        "geneval2": {
            "delta": float(geneval["delta"]),
            "ci": list(map(float, geneval["ci"])),
            "positive_direction": geneval_direction,
        },
        "fidelity": {
            "pass": fidelity_noninferior,
            "d_cmmd": float(fidelity_gate["d_cmmd"]),
            "d_fid": float(fidelity_gate["d_fid"]),
        },
        "diversity": {
            metric: {
                "delta": float(values["delta"]),
                "ci": list(map(float, values["ci"])),
                "verdict": str(values["verdict"]),
            }
            for metric, values in diversity_gate.items()
        },
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.no_wandb:
        return
    import wandb

    run = wandb.init(
        project=args.project, name=args.run_name, id=args.run_name,
        group=args.group, job_type="publication-promotion-gate",
        tags=["phaseL", "publication", "gate", "counterfactual-control"],
        config={"eval_root": str(root), "rule": summary["rule"]},
        dir=str(output.parent), resume="never",
    )
    metrics = {
        "gate/pass": float(promotion),
        "gate/compbench_delta": summary["compbench"]["delta"],
        "gate/compbench_ci_low": summary["compbench"]["ci"][0],
        "gate/compbench_ci_high": summary["compbench"]["ci"][1],
        "gate/geneval2_delta": summary["geneval2"]["delta"],
        "gate/geneval2_ci_low": summary["geneval2"]["ci"][0],
        "gate/geneval2_ci_high": summary["geneval2"]["ci"][1],
        "gate/fidelity_pass": float(fidelity_noninferior),
        "gate/d_cmmd": summary["fidelity"]["d_cmmd"],
        "gate/d_fid": summary["fidelity"]["d_fid"],
        "gate/diversity_pass": float(diversity_noninferior),
    }
    wandb.log(metrics, step=0)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f"{args.run_name}-decision", type="promotion-gate")
    artifact.add_file(str(output), name="summary.json")
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
