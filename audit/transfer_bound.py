#!/usr/bin/env python3
"""Amortization-ratio bound for the Phase-C1 selected-trajectory null.

The handoff quotes 0.0097 / 0.0717 ~ 13.5% as a shortcut and correctly refuses to call it a 95%
upper bound, asking for a "matched or joint bootstrap".

This script first establishes which of those is even possible. It keys both evaluations by PROMPT
TEXT (their integer `idx` fields are per-job and are NOT comparable: idx 0 of the C1 colour pool is
"a white piano and a black bench" while idx 0 of the headroom colour pool is "a blue backpack and a
green bottle"), then reports the overlap.

- If the populations overlap, a joint bootstrap resamples shared prompts once and recomputes both
  numerator and denominator, capturing their correlation.
- If they are disjoint, no joint bootstrap exists. Independence is then a property of the design
  rather than an assumption, so the ratio interval is formed from independent bootstraps. The
  resulting interval is honest about sampling error but the two effects still describe different
  prompt populations, which is a separate, non-statistical limitation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex"]


def load(scores_dir: Path, model: str, step: int, category: str) -> dict[str, float]:
    path = scores_dir / f"{model}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        row["prompt"]: float(row["score"])
        for row in payload["per_prompt"]
        if row.get("prompt") is not None
    }


def paired_by_prompt(
    scores_dir: Path, treatment: str, control: str, step: int
) -> dict[str, dict[str, float]]:
    """-> {category: {prompt: treatment - control}}"""
    out = {}
    for category in PRIMARY:
        a = load(scores_dir, treatment, step, category)
        b = load(scores_dir, control, step, category)
        shared = sorted(set(a) & set(b))
        if shared:
            out[category] = {prompt: a[prompt] - b[prompt] for prompt in shared}
    return out


def category_weighted_draws(
    differences: dict[str, dict[str, float]], draws: int, rng: np.random.Generator,
    restrict: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Equal-category-weighted mean, prompts resampled within category."""
    per_category = []
    for category, mapping in sorted(differences.items()):
        keys = restrict[category] if restrict else sorted(mapping)
        values = np.array([mapping[k] for k in keys], dtype=np.float64)
        n = values.shape[0]
        per_category.append(values[rng.integers(0, n, size=(draws, n))].mean(axis=1))
    return np.mean(per_category, axis=0)


def point_estimate(
    differences: dict[str, dict[str, float]], restrict: dict[str, list[str]] | None = None
) -> float:
    means = []
    for category, mapping in sorted(differences.items()):
        keys = restrict[category] if restrict else sorted(mapping)
        means.append(float(np.mean([mapping[k] for k in keys])))
    return float(np.mean(means))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer_scores_dir", default="phaseC/eval_scores")
    parser.add_argument("--transfer_treatment", default="B4")
    parser.add_argument("--transfer_control", default="B2")
    parser.add_argument("--transfer_step", type=int, default=4)
    parser.add_argument("--oracle_scores_dir", default="exp0/primary_headroom/scores")
    parser.add_argument("--oracle_treatment", default="oracle")
    parser.add_argument("--oracle_control", default="random")
    parser.add_argument("--oracle_step", type=int, default=8)
    parser.add_argument("--draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    transfer = paired_by_prompt(
        Path(args.transfer_scores_dir),
        args.transfer_treatment,
        args.transfer_control,
        args.transfer_step,
    )
    oracle = paired_by_prompt(
        Path(args.oracle_scores_dir),
        args.oracle_treatment,
        args.oracle_control,
        args.oracle_step,
    )
    if not transfer or not oracle:
        raise SystemExit("missing scored prompts for one of the two contrasts")

    overlap = {
        category: sorted(set(transfer.get(category, {})) & set(oracle.get(category, {})))
        for category in PRIMARY
    }
    total_overlap = sum(len(v) for v in overlap.values())
    joint_possible = total_overlap > 0

    rng = np.random.default_rng(args.seed)
    if joint_possible:
        restrict = {c: v for c, v in overlap.items() if v}
        transfer_draws = category_weighted_draws(
            {c: transfer[c] for c in restrict}, args.draws, rng, restrict
        )
        # Reuse the same prompt resampling for the denominator by re-seeding identically.
        rng_joint = np.random.default_rng(args.seed)
        oracle_draws = category_weighted_draws(
            {c: oracle[c] for c in restrict}, args.draws, rng_joint, restrict
        )
        transfer_point = point_estimate(transfer, restrict)
        oracle_point = point_estimate(oracle, restrict)
        method = "joint bootstrap on shared prompts"
    else:
        transfer_draws = category_weighted_draws(transfer, args.draws, rng)
        oracle_draws = category_weighted_draws(oracle, args.draws, np.random.default_rng(args.seed + 1))
        transfer_point = point_estimate(transfer)
        oracle_point = point_estimate(oracle)
        method = (
            "independent bootstraps; the two contrasts share no prompts, so no joint "
            "resampling exists and independence is a design property, not an assumption"
        )

    # Ratio distribution. Denominator draws are strictly positive here; guard anyway.
    positive = oracle_draws > 0
    ratio = np.full(args.draws, np.nan)
    ratio[positive] = transfer_draws[positive] / oracle_draws[positive]
    finite = ratio[np.isfinite(ratio)]

    report = {
        "method": method,
        "joint_bootstrap_possible": joint_possible,
        "prompt_key": "prompt text (integer idx is per-job and not comparable across jobs)",
        "overlap_prompts_by_category": {c: len(v) for c, v in overlap.items()},
        "total_overlap_prompts": total_overlap,
        "transfer": {
            "contrast": f"{args.transfer_treatment} minus {args.transfer_control} @ step {args.transfer_step}",
            "scores_dir": str(Path(args.transfer_scores_dir).resolve()),
            "prompts_by_category": {c: len(v) for c, v in transfer.items()},
            "delta": transfer_point,
            "ci95": [
                float(np.percentile(transfer_draws, 2.5)),
                float(np.percentile(transfer_draws, 97.5)),
            ],
            "upper_97_5": float(np.percentile(transfer_draws, 97.5)),
        },
        "oracle": {
            "contrast": f"{args.oracle_treatment} minus {args.oracle_control} @ step {args.oracle_step}",
            "scores_dir": str(Path(args.oracle_scores_dir).resolve()),
            "prompts_by_category": {c: len(v) for c, v in oracle.items()},
            "delta": oracle_point,
            "ci95": [
                float(np.percentile(oracle_draws, 2.5)),
                float(np.percentile(oracle_draws, 97.5)),
            ],
        },
        "ratio": {
            "point": transfer_point / oracle_point,
            "ci95": [
                float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)),
            ],
            "upper_95_one_sided": float(np.percentile(finite, 95.0)),
            "upper_97_5": float(np.percentile(finite, 97.5)),
            "p_ratio_gt_0_25": float((finite > 0.25).mean()),
        },
        "shortcut_for_comparison": {
            "upper_transfer_over_point_oracle": float(
                np.percentile(transfer_draws, 97.5) / oracle_point
            ),
            "upper_transfer_over_lower_oracle": float(
                np.percentile(transfer_draws, 97.5) / np.percentile(oracle_draws, 2.5)
            ),
        },
        "draws": args.draws,
        "seed": args.seed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        t, o, r = report["transfer"], report["oracle"], report["ratio"]
        lines = [
            "# Amortization ratio for the selected-trajectory null",
            "",
            f"Method: {method}.",
            "",
            "## Can the two contrasts be bootstrapped jointly?",
            "",
            f"Prompts are matched on **text**, not on the integer `idx` field, which is assigned "
            f"per job and is not comparable across jobs.",
            "",
            f"Shared prompts across the two evaluations: **{total_overlap}**.",
            "",
        ]
        if joint_possible:
            lines.append(
                "The populations overlap, so numerator and denominator are resampled together and "
                "their correlation is captured."
            )
        else:
            lines += [
                "The populations are **disjoint**. No joint or matched bootstrap exists in the "
                "current artifacts, so the ratio below combines two independent bootstraps. That "
                "handles sampling error correctly, but the numerator and denominator still "
                "describe different prompt populations. Closing that gap requires re-estimating "
                "the teacher oracle on the same held-out split as the transfer contrast "
                "(roadmap task T), not a different statistical procedure.",
            ]
        lines += [
            "",
            "## Components",
            "",
            "| quantity | contrast | prompts | Δ | 95% CI |",
            "|---|---|---:|---:|---|",
            f"| numerator (transfer) | {t['contrast']} | {sum(t['prompts_by_category'].values())} | "
            f"{t['delta']:+.5f} | [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}] |",
            f"| denominator (oracle) | {o['contrast']} | {sum(o['prompts_by_category'].values())} | "
            f"{o['delta']:+.5f} | [{o['ci95'][0]:+.5f}, {o['ci95'][1]:+.5f}] |",
            "",
            "## Ratio",
            "",
            f"- Point estimate: **{r['point']:.4f}**",
            f"- 95% interval: **[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]**",
            f"- One-sided 95% upper bound: **{r['upper_95_one_sided']:.4f}**",
            "",
            "### Comparison with the shortcut quoted in the handoff",
            "",
            f"- upper transfer bound / oracle point: "
            f"{report['shortcut_for_comparison']['upper_transfer_over_point_oracle']:.4f}",
            f"- upper transfer bound / oracle lower bound: "
            f"{report['shortcut_for_comparison']['upper_transfer_over_lower_oracle']:.4f}",
            "",
            "The shortcut divides an interval endpoint by a point estimate and is therefore neither "
            "a bound on the ratio nor a confidence statement about it. The ratio interval above is "
            "the quantity that can be cited.",
        ]
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "joint_bootstrap_possible": joint_possible,
        "total_overlap_prompts": total_overlap,
        "transfer": report["transfer"]["delta"],
        "transfer_ci": report["transfer"]["ci95"],
        "oracle": report["oracle"]["delta"],
        "oracle_ci": report["oracle"]["ci95"],
        "ratio": report["ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
