#!/usr/bin/env python3
"""Task D — dose audit of the fixed-pair cache, before any GPU time is spent.

The point of the audit is to know, in advance, how much label disagreement each arm actually
carries and where it lives. If the counterfactual arm barely disagrees with the correct arm, or the
disagreement is concentrated in one category, the pilot cannot answer the orientation question no
matter how it turns out.

Reports both preregistered populations: all records, and the reversal-only subset where the same
two images receive opposite labels.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseFP.build_fixed_pair_selection import ARMS
from phaseI.train_preference import load_selection


def quantiles(values: list[float], points=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {f"q{int(p * 100):02d}": ordered[min(n - 1, int(p * n))] for p in points}


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        **quantiles(values),
    }


def bootstrap_mean_ci(values: list[float], draws: int, seed: int) -> tuple[float, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = array.shape[0]
    means = array[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    root = Path(args.cache_root)
    arms = {arm: load_selection(root / arm) for arm in ARMS}
    correct = arms["correct"]
    n = len(correct)

    reversal_idx = {int(r["idx"]) for r in correct if r["pair_reverses_under_counterfactual"]}
    by_idx = {arm: {int(r["idx"]): r for r in records} for arm, records in arms.items()}

    # Effective label disagreement: fraction of records where the arm's positive image differs from
    # the correct arm's positive image. This is the true "dose" each arm delivers.
    disagreement = {}
    for arm in ARMS:
        differing = sum(
            1
            for idx in by_idx["correct"]
            if by_idx[arm][idx]["positive_idx"] != by_idx["correct"][idx]["positive_idx"]
        )
        disagreement[arm] = differing / n

    # Assigned margin, i.e. the signed original-prompt score gap the loss actually asks for.
    assigned = {}
    for arm in ARMS:
        assigned[arm] = [
            float(r["original_endpoint_vqa"][int(r["positive_idx"])])
            - float(r["original_endpoint_vqa"][int(r["negative_idx"])])
            for r in arms[arm]
        ]

    populations = {
        "all": [int(r["idx"]) for r in correct],
        "reversal_only": sorted(reversal_idx),
        "non_reversal": sorted({int(r["idx"]) for r in correct} - reversal_idx),
    }

    by_population = {}
    for name, idx_list in populations.items():
        keep = set(idx_list)
        entry = {"n": len(idx_list)}
        for arm in ARMS:
            values = [
                assigned[arm][i] for i, r in enumerate(arms[arm]) if int(r["idx"]) in keep
            ]
            lo, hi = bootstrap_mean_ci(values, args.bootstrap, args.seed)
            entry[arm] = {**summarize(values), "mean_ci95": [lo, hi]}
        # Contrast that the pilot is powered to detect: correct minus counterfactual assigned margin.
        contrast = [
            assigned["correct"][i] - assigned["counterfactual"][i]
            for i, r in enumerate(correct)
            if int(r["idx"]) in keep
        ]
        lo, hi = bootstrap_mean_ci(contrast, args.bootstrap, args.seed + 1)
        entry["correct_minus_counterfactual_assigned_margin"] = {
            **summarize(contrast),
            "mean_ci95": [lo, hi],
        }
        by_population[name] = entry

    # Where does the disagreement live? A dose concentrated in one category cannot support a general
    # claim, and the evaluator families are category-defined.
    category_counts = Counter(str(r.get("category", "unknown")) for r in correct)
    edit_counts = Counter(str(r.get("edit_family", "unknown")) for r in correct)
    reversal_by_category = Counter(
        str(by_idx["correct"][idx].get("category", "unknown")) for idx in reversal_idx
    )
    reversal_by_edit = Counter(
        str(by_idx["correct"][idx].get("edit_family", "unknown")) for idx in reversal_idx
    )

    # Reversals concentrated on small original margins would mean the counterfactual arm mostly
    # flips near-ties, which is a weak intervention regardless of its rate.
    reversal_original_margin = [
        float(by_idx["correct"][idx]["original_pair_margin"]) for idx in reversal_idx
    ]
    non_reversal_original_margin = [
        float(r["original_pair_margin"])
        for r in correct
        if int(r["idx"]) not in reversal_idx
    ]
    margin_gap = statistics.mean(reversal_original_margin) - statistics.mean(
        non_reversal_original_margin
    )

    # Counterfactual score margin on the fixed pair: how confidently the edited text reorders it.
    cf_margin_reversal = [
        abs(float(by_idx["correct"][idx]["counterfactual_pair_margin"])) for idx in reversal_idx
    ]

    report = {
        "cache_root": str(root.resolve()),
        "records": n,
        "arms": list(ARMS),
        "effective_label_disagreement_vs_correct": disagreement,
        "reversal_records": len(reversal_idx),
        "reversal_fraction": len(reversal_idx) / n,
        "populations": by_population,
        "reversal_rate_by_category": {
            category: reversal_by_category[category] / count
            for category, count in sorted(category_counts.items())
        },
        "reversal_rate_by_edit_family": {
            family: reversal_by_edit[family] / count
            for family, count in sorted(edit_counts.items())
        },
        "records_by_category": dict(sorted(category_counts.items())),
        "records_by_edit_family": dict(sorted(edit_counts.items())),
        "original_margin_reversal": summarize(reversal_original_margin),
        "original_margin_non_reversal": summarize(non_reversal_original_margin),
        "original_margin_reversal_minus_non_reversal": margin_gap,
        "counterfactual_margin_on_reversed_pairs": summarize(cf_margin_reversal),
        "categories_with_reversal_rate_below_0_10": [
            category
            for category, count in category_counts.items()
            if reversal_by_category[category] / count < 0.10
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        lines = [
            "# Fixed-pair dose audit",
            "",
            f"Cache: `{root}`  ",
            f"Records: **{n}**; reversal subset: **{len(reversal_idx)}** "
            f"({len(reversal_idx) / n:.2%})",
            "",
            "## Effective label disagreement versus the correct arm",
            "",
            "| arm | fraction of records with a different preferred image |",
            "|---|---:|",
        ]
        for arm in ARMS:
            lines.append(f"| `{arm}` | {disagreement[arm]:.4f} |")
        lines += [
            "",
            "## Assigned original-prompt score margin",
            "",
            "| population | arm | mean | 95% CI | median |",
            "|---|---|---:|---|---:|",
        ]
        for name, entry in by_population.items():
            for arm in ARMS:
                stats = entry[arm]
                lines.append(
                    f"| {name} (n={entry['n']}) | `{arm}` | {stats['mean']:+.4f} | "
                    f"[{stats['mean_ci95'][0]:+.4f}, {stats['mean_ci95'][1]:+.4f}] | "
                    f"{stats['q50']:+.4f} |"
                )
        lines += [
            "",
            "## Reversal rate by prompt category",
            "",
            "| category | records | reversal rate |",
            "|---|---:|---:|",
        ]
        for category, count in sorted(category_counts.items()):
            lines.append(
                f"| {category} | {count} | {reversal_by_category[category] / count:.4f} |"
            )
        lines += [
            "",
            "## Reversal rate by edit family",
            "",
            "| edit family | records | reversal rate |",
            "|---|---:|---:|",
        ]
        for family, count in sorted(edit_counts.items()):
            lines.append(f"| {family} | {count} | {reversal_by_edit[family] / count:.4f} |")
        lines += [
            "",
            "## Are reversals concentrated on near-ties?",
            "",
            f"- Mean original-prompt pair margin, reversed records: "
            f"{statistics.mean(reversal_original_margin):.4f}",
            f"- Mean original-prompt pair margin, non-reversed records: "
            f"{statistics.mean(non_reversal_original_margin):.4f}",
            f"- Difference: {margin_gap:+.4f}",
            f"- Mean |counterfactual margin| on reversed pairs: "
            f"{statistics.mean(cf_margin_reversal):.4f}",
        ]
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "records": n,
        "reversal_fraction": len(reversal_idx) / n,
        "effective_label_disagreement_vs_correct": disagreement,
        "original_margin_reversal_minus_non_reversal": margin_gap,
    }, indent=2))


if __name__ == "__main__":
    main()
