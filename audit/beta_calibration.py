#!/usr/bin/env python3
"""Task K precondition — choose beta from the measured logit distribution, not by guesswork.

beta = 100 was a pilot value with no derivation. The preference loss is
-log sigmoid(-beta * (Delta_theta - Delta_0)); what matters is where beta puts the argument of the
sigmoid. If |beta * gap| is routinely above ~4 the loss is saturated and its gradient is nearly
independent of the margin; if it is routinely below ~0.1 the term is nearly linear and contributes
almost nothing beyond a constant.

Also separates the two ways the loss can achieve its objective: improving the winner's flow error
versus degrading the loser's. The aggregate margin cannot distinguish them, and they have different
consequences for fidelity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RECOMMENDED_ABS_LOGIT = (0.5, 4.0)


def load(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = {k for row in rows for k in row}
    out = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        out[key] = np.asarray(values, dtype=np.float64)
    return out


def describe(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "min": float(values.min()),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def analyze_run(name: str, path: Path, beta: float) -> dict:
    history = load(path)
    logit = history.get("train/preference_logit")
    if logit is None:
        raise SystemExit(f"{path} has no train/preference_logit")

    # The scale-free quantity. logit = -beta * (Delta_theta - Delta_0), so gap = -logit / beta.
    gap = -logit / beta
    abs_gap = np.abs(gap)
    saturated = np.abs(logit) > RECOMMENDED_ABS_LOGIT[1]
    dead = np.abs(logit) < RECOMMENDED_ABS_LOGIT[0]

    entry = {
        "run": name,
        "beta_used": beta,
        "logged_steps": int(logit.size),
        "preference_logit": describe(logit),
        "gap_delta_theta_minus_delta_0": describe(gap),
        "abs_gap": describe(abs_gap),
        "saturation": {
            "fraction_abs_logit_gt_4": float(saturated.mean()),
            "fraction_abs_logit_lt_0_5": float(dead.mean()),
            "fraction_in_responsive_band": float((~saturated & ~dead).mean()),
            "mean_sigmoid_derivative": float(
                np.mean(1.0 / (2.0 + np.exp(logit) + np.exp(-logit)))
            ),
        },
        "preference_accuracy": (
            describe(history["train/preference_accuracy"])
            if "train/preference_accuracy" in history
            else None
        ),
        "loss_preference": describe(history["train/loss_preference"]),
    }

    # beta values that would place a target quantile of |gap| at a target |logit|.
    median_abs_gap = float(np.median(abs_gap))
    q75_abs_gap = float(np.percentile(abs_gap, 75))
    entry["beta_recommendations"] = {
        "median_abs_gap": median_abs_gap,
        "q75_abs_gap": q75_abs_gap,
        "beta_for_median_logit_1": 1.0 / median_abs_gap if median_abs_gap > 0 else None,
        "beta_for_median_logit_2": 2.0 / median_abs_gap if median_abs_gap > 0 else None,
        "beta_for_q75_logit_4": 4.0 / q75_abs_gap if q75_abs_gap > 0 else None,
    }

    # Which branch moves? Both are logged relative to the frozen M1 reference.
    winner = history.get("train/positive_error_improvement_vs_reference")
    loser = history.get("train/negative_error_increase_vs_reference")
    if winner is not None and loser is not None:
        total = np.abs(winner) + np.abs(loser)
        share = np.divide(
            np.abs(winner), total, out=np.full_like(total, np.nan), where=total > 0
        )
        entry["mechanism"] = {
            "winner_error_improvement_vs_reference": describe(winner),
            "loser_error_increase_vs_reference": describe(loser),
            "mean_share_of_gap_change_from_winner": float(np.nanmean(share)),
            "final_winner_improvement": float(winner[-1]),
            "final_loser_increase": float(loser[-1]),
            "interpretation": (
                "winner-driven"
                if np.nanmean(share) > 0.6
                else "loser-driven"
                if np.nanmean(share) < 0.4
                else "balanced"
            ),
        }
    for key, label in (
        ("train/student_grad_norm_pre_clip", "grad_norm_pre_clip"),
        ("train/student_clip_ratio", "clip_ratio"),
        ("train/param_l2_from_m1", "param_l2_from_m1"),
        ("train/loss_anchor", "loss_anchor"),
    ):
        if key in history:
            entry[label] = describe(history[key])
    if "train/student_clip_ratio" in history:
        entry["clipping_rate"] = float((history["train/student_clip_ratio"] < 0.999).mean())
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=HISTORY.jsonl",
        help="Repeat per run.",
    )
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    entries = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        entries.append(analyze_run(name, Path(path), args.beta))

    report = {"beta_used": args.beta, "runs": entries}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        lines = [
            "# Preference-logit distribution and beta calibration",
            "",
            f"All completed Phase-I runs used beta = {args.beta:g}. "
            "The loss is `-log sigmoid(-beta * (Delta_theta - Delta_0))`, so the quantity that "
            "actually sets the gradient regime is `beta * |Delta_theta - Delta_0|`.",
            "",
            "## Where beta = 100 puts the sigmoid",
            "",
            "| run | median \\|logit\\| | frac \\|logit\\|>4 (saturated) | frac \\|logit\\|<0.5 (inert) | "
            "responsive | mean sigmoid' |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for entry in entries:
            sat = entry["saturation"]
            lines.append(
                f"| `{entry['run']}` | "
                f"{np.median(np.abs([entry['preference_logit']['median']])):.3f} | "
                f"{sat['fraction_abs_logit_gt_4']:.3f} | {sat['fraction_abs_logit_lt_0_5']:.3f} | "
                f"{sat['fraction_in_responsive_band']:.3f} | {sat['mean_sigmoid_derivative']:.4f} |"
            )
        lines += [
            "",
            "## Scale-free margin and implied beta",
            "",
            "| run | median \\|gap\\| | q75 \\|gap\\| | beta for median \\|logit\\|=1 | "
            "beta for median \\|logit\\|=2 | beta for q75 \\|logit\\|=4 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for entry in entries:
            rec = entry["beta_recommendations"]
            lines.append(
                f"| `{entry['run']}` | {rec['median_abs_gap']:.5f} | {rec['q75_abs_gap']:.5f} | "
                f"{rec['beta_for_median_logit_1']:.1f} | {rec['beta_for_median_logit_2']:.1f} | "
                f"{rec['beta_for_q75_logit_4']:.1f} |"
            )
        lines += [
            "",
            "## Does the loss improve winners or degrade losers?",
            "",
            "Both quantities are measured against the frozen M1 reference, so positive values mean "
            "the student moved away from M1 in the direction the loss requests.",
            "",
            "| run | winner error improvement | loser error increase | winner share | verdict |",
            "|---|---:|---:|---:|---|",
        ]
        for entry in entries:
            mech = entry.get("mechanism")
            if not mech:
                continue
            lines.append(
                f"| `{entry['run']}` | {mech['winner_error_improvement_vs_reference']['mean']:+.5f} | "
                f"{mech['loser_error_increase_vs_reference']['mean']:+.5f} | "
                f"{mech['mean_share_of_gap_change_from_winner']:.3f} | {mech['interpretation']} |"
            )
        lines += [
            "",
            "## Gradient and drift",
            "",
            "| run | grad norm (median) | clipping rate | final L2 drift from M1 | anchor loss (median) |",
            "|---|---:|---:|---:|---:|",
        ]
        for entry in entries:
            lines.append(
                f"| `{entry['run']}` | {entry['grad_norm_pre_clip']['median']:.4f} | "
                f"{entry.get('clipping_rate', float('nan')):.3f} | "
                f"{entry['param_l2_from_m1']['max']:.4f} | {entry['loss_anchor']['median']:.5f} |"
            )
        Path(args.md).write_text("\n".join(lines) + "\n")

    for entry in entries:
        print(json.dumps({
            "run": entry["run"],
            "median_logit": entry["preference_logit"]["median"],
            "q05_q95_logit": [entry["preference_logit"]["q05"], entry["preference_logit"]["q95"]],
            "saturation": entry["saturation"],
            "beta_recommendations": entry["beta_recommendations"],
            "mechanism": entry.get("mechanism", {}).get("interpretation"),
        }, indent=2))


if __name__ == "__main__":
    main()
