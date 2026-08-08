#!/usr/bin/env python3
"""Paired prompt bootstrap of a CompBench contrast, split by EVALUATOR ARCHITECTURE.

T2I-CompBench is not one VQA metric. Its categories are scored by three architecturally distinct
instruments, and the preference labels come from a fourth (VQAScore/CLIP-FlanT5). A contrast that
only moves the BLIP-VQA family is far weaker evidence than one that also moves the UniDet detection
family, because the latter shares no architecture with the selector.

The handoff records unbootstrapped family means. This produces the paired intervals they need
before they can be cited.

Resampling matches the reported estimand: prompts are resampled within category (paired across the
two models), category means are recomputed, and the family effect is the equal-weighted mean of its
category means. Equal category weighting is the preregistered primary, so the bootstrap must not
pool prompts across categories of unequal size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = {
    "BLIP-VQA": ["color", "shape", "texture"],
    "UniDet": ["spatial", "3d_spatial", "numeracy"],
    "3-in-1": ["complex"],
    "CLIPScore": ["non_spatial"],
}
PRIMARY_FAMILIES = ["BLIP-VQA", "UniDet", "3-in-1"]
# CLIPScore's scale is ~10x compressed relative to BLIP-VQA, so it is reported and never pooled.
SECONDARY_FAMILIES = ["CLIPScore"]


def load_scores(scores_dir: Path, model: str, step: int, category: str) -> dict[int, float]:
    path = scores_dir / f"{model}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(row["idx"]): float(row["score"]) for row in payload["per_prompt"]}


def paired_differences(
    scores_dir: Path, treatment: str, control: str, step: int
) -> dict[str, np.ndarray]:
    out = {}
    for category in sum(FAMILIES.values(), []):
        a = load_scores(scores_dir, treatment, step, category)
        b = load_scores(scores_dir, control, step, category)
        shared = sorted(set(a) & set(b))
        if shared:
            out[category] = np.array([a[k] - b[k] for k in shared], dtype=np.float64)
    return out


def bootstrap_families(
    differences: dict[str, np.ndarray], draws: int, seed: int
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    # One resampling of prompts drives every reported aggregate, so family and pooled intervals
    # stay mutually consistent rather than coming from independent bootstraps.
    category_draws = {}
    for category, values in differences.items():
        n = values.shape[0]
        picks = rng.integers(0, n, size=(draws, n))
        category_draws[category] = values[picks].mean(axis=1)

    family_stats = {}
    for family, categories in FAMILIES.items():
        present = [c for c in categories if c in category_draws]
        if not present:
            continue
        point = float(np.mean([differences[c].mean() for c in present]))
        samples = np.mean([category_draws[c] for c in present], axis=0)
        family_stats[family] = {
            "categories": present,
            "n_prompts": int(sum(differences[c].shape[0] for c in present)),
            "delta": point,
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "p_gt_zero": float((samples > 0).mean()),
        }

    pooled_present = [c for f in PRIMARY_FAMILIES for c in FAMILIES[f] if c in category_draws]
    pooled = {}
    if pooled_present:
        point = float(np.mean([differences[c].mean() for c in pooled_present]))
        samples = np.mean([category_draws[c] for c in pooled_present], axis=0)
        pooled = {
            "categories": pooled_present,
            "delta": point,
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }

    # Do the two architecturally disjoint primary families disagree? A large interval on the
    # difference means the split is uninformative, not that the families agree.
    contrast = {}
    if "BLIP-VQA" in family_stats and "UniDet" in family_stats:
        blip = np.mean([category_draws[c] for c in family_stats["BLIP-VQA"]["categories"]], axis=0)
        unidet = np.mean([category_draws[c] for c in family_stats["UniDet"]["categories"]], axis=0)
        gap = blip - unidet
        contrast = {
            "delta": family_stats["BLIP-VQA"]["delta"] - family_stats["UniDet"]["delta"],
            "ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
            "families_disagree_in_sign": bool(
                family_stats["BLIP-VQA"]["delta"] * family_stats["UniDet"]["delta"] < 0
            ),
        }

    category_stats = {
        category: {
            "n": int(values.shape[0]),
            "delta": float(values.mean()),
            "ci95": [
                float(np.percentile(category_draws[category], 2.5)),
                float(np.percentile(category_draws[category], 97.5)),
            ],
        }
        for category, values in differences.items()
    }
    return (
        {
            "families": family_stats,
            "pooled_primary": pooled,
            "blip_minus_unidet": contrast,
        },
        category_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    differences = paired_differences(scores_dir, args.treatment, args.control, args.step)
    if not differences:
        raise SystemExit(f"no shared scored prompts for {args.treatment} vs {args.control}")
    aggregates, categories = bootstrap_families(differences, args.draws, args.seed)

    report = {
        "title": args.title or f"{args.treatment} minus {args.control}",
        "scores_dir": str(scores_dir.resolve()),
        "treatment": args.treatment,
        "control": args.control,
        "step": args.step,
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "estimand": (
            "equal-category-weighted mean paired difference; prompts resampled within category"
        ),
        **aggregates,
        "categories": categories,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        lines = [
            f"# {report['title']} — CompBench by evaluator family",
            "",
            f"Paired prompt bootstrap, {args.draws} draws, seed {args.seed}. "
            "Prompts are resampled within category; the family effect is the equal-weighted mean "
            "of its category means.",
            "",
            "| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |",
            "|---|---|---:|---:|---|---:|",
        ]
        for family in PRIMARY_FAMILIES + SECONDARY_FAMILIES:
            stats = aggregates["families"].get(family)
            if not stats:
                continue
            secondary = " *(secondary)*" if family in SECONDARY_FAMILIES else ""
            lines.append(
                f"| **{family}**{secondary} | {', '.join(stats['categories'])} | "
                f"{stats['n_prompts']} | {stats['delta']:+.4f} | "
                f"[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}] | {stats['p_gt_zero']:.4f} |"
            )
        if aggregates["pooled_primary"]:
            pooled = aggregates["pooled_primary"]
            lines.append(
                f"| pooled primary | {len(pooled['categories'])} categories | — | "
                f"{pooled['delta']:+.4f} | "
                f"[{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}] | — |"
            )
        if aggregates["blip_minus_unidet"]:
            contrast = aggregates["blip_minus_unidet"]
            lines += [
                "",
                "## Architecture disagreement",
                "",
                f"BLIP-VQA minus UniDet: **{contrast['delta']:+.4f}** "
                f"[{contrast['ci95'][0]:+.4f}, {contrast['ci95'][1]:+.4f}]. "
                + (
                    "The two disjoint families disagree in sign."
                    if contrast["families_disagree_in_sign"]
                    else "Both disjoint families point the same way."
                ),
                "",
                "The UniDet family shares no architecture with the VQAScore selector that produced "
                "the training labels, so a same-signed UniDet effect is the part of this contrast "
                "that is not explainable by scorer circularity.",
            ]
        lines += [
            "",
            "## Per category",
            "",
            "| category | family | n | Δ | 95% CI |",
            "|---|---|---:|---:|---|",
        ]
        family_of = {c: f for f, cs in FAMILIES.items() for c in cs}
        for category, stats in sorted(categories.items()):
            lines.append(
                f"| {category} | {family_of[category]} | {stats['n']} | {stats['delta']:+.4f} | "
                f"[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}] |"
            )
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({"families": aggregates["families"], "pooled": aggregates["pooled_primary"]}, indent=2))


if __name__ == "__main__":
    main()
