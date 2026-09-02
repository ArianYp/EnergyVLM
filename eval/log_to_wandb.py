#!/usr/bin/env python3
"""Push finished evaluations to Weights & Biases as tables, summary metrics and image grids.

One wandb run per evaluated model (named by its label), holding:
    eval/compbench_<category>, eval/compbench_mean, eval/geneval2      summary metrics
    eval/per_category                                                  table
    eval/per_prompt                                                    table (every prompt, both benchmarks)
    fidelity/fid, cmmd, precision, recall (when a fidelity report contains the label)
    samples/<benchmark>                                                a grid of the first --n_images images

    python eval/log_to_wandb.py --project my-project --eval_dir out/eval/eval_dino_patch_s0 \
        --fidelity out/fidelity_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--eval_dir", required=True, help="output of scripts/eval_alignment.lsf")
    ap.add_argument("--fidelity", default=None, help="json written by eval/fidelity.py")
    ap.add_argument("--n_images", type=int, default=16)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()
    import wandb

    d = Path(args.eval_dir)
    align = json.loads((d / "alignment.json").read_text())
    label = align["label"]
    run = wandb.init(project=args.project, name=args.run_name or f"eval_{label}", job_type="eval",
                     config={k: v for k, v in align.items() if k != "compbench"})

    summary = {f"eval/compbench_{c}": v for c, v in align["compbench"].items()}
    summary["eval/compbench_mean"] = align["compbench_mean"]
    summary["eval/geneval2"] = align["geneval2"]
    per_cat = wandb.Table(columns=["category", "mean"],
                          data=[[c, v] for c, v in sorted(align["compbench"].items())] + [["geneval2", align["geneval2"]]])

    rows = []
    for p in sorted(glob.glob(str(d / "compbench_scores" / "*" / "scores.json"))):
        for r in json.loads(Path(p).read_text())["per_prompt"]:
            rows.append(["compbench", r["category"], r["idx"], r["prompt"], float(r["score"])])
    for p in sorted(glob.glob(str(d / "geneval2_scores" / "*" / "scores.json"))):
        for r in json.loads(Path(p).read_text())["per_prompt"]:
            rows.append(["geneval2", "geneval2", r["idx"], r["prompt"], float(r["score"])])
    per_prompt = wandb.Table(columns=["benchmark", "category", "idx", "prompt", "score"], data=rows)

    if args.fidelity and Path(args.fidelity).exists():
        res = json.loads(Path(args.fidelity).read_text())["results"]
        for key, v in res.items():
            if v["model"].startswith(label):
                summary.update({f"fidelity/fid@{v['steps']}": v["fid"], f"fidelity/cmmd@{v['steps']}": v["cmmd"],
                                f"fidelity/precision@{v['steps']}": v["precision"],
                                f"fidelity/recall@{v['steps']}": v["recall"]})

    media = {}
    for bench in ("compbench", "geneval2"):
        imgs = sorted(glob.glob(str(d / bench / "images" / label / "p*" / "s*" / "cand0.png")))[: args.n_images]
        if imgs:
            media[f"samples/{bench}"] = [wandb.Image(p, caption=Path(p).parts[-3]) for p in imgs]

    wandb.log({"eval/per_category": per_cat, "eval/per_prompt": per_prompt, **media})
    for k, v in summary.items():
        run.summary[k] = v
    print(json.dumps(summary, indent=1))
    run.finish()


if __name__ == "__main__":
    main()
