#!/usr/bin/env python3
"""Phase-FP primary analysis — the preregistered monotone dose response across four arms.

A single pairwise contrast is underpowered here by construction: the correct-versus-counterfactual
orientation dose is only 0.108 in assigned-margin units against ~0.009 CI half-widths. The four arms
lie on a known dose ladder, so the trend across all of them is the statistic that uses the available
information.

One prompt resampling drives every reported number, so the slope, the pairwise contrasts, the
ordering probability and the per-family results are mutually consistent rather than coming from
independent bootstraps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex"]
FAMILIES = {
    "BLIP-VQA": ["color", "shape", "texture"],
    "UniDet": ["spatial", "3d_spatial", "numeracy"],
    "3-in-1": ["complex"],
    "CLIPScore": ["non_spatial"],
}
# Doses are a property of the cache, not of any result. Recomputed by phaseFP/dose_audit.py as
# (correct mean assigned margin) - (arm mean assigned margin).
DEFAULT_DOSE = {
    "correct": 0.0,
    "counterfactual": 0.10816,
    "random": 0.28663,
    "inverted": 0.57291,
}


def load_scores(scores_dir: Path, label: str, step: int, category: str) -> dict[str, float]:
    path = scores_dir / f"{label}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        row["prompt"]: float(row["score"])
        for row in payload["per_prompt"]
        if row.get("prompt") is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=LABEL:SCORES_DIR",
        help="e.g. correct=CorrectFixed:phaseFP/eval_CorrectFixed_123/compbench_scores",
    )
    parser.add_argument("--dose", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--reference_arm", default="correct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        label, _, scores_dir = rest.partition(":")
        arms[name] = (label, Path(scores_dir))
    doses = dict(DEFAULT_DOSE)
    for spec in args.dose:
        name, _, value = spec.partition("=")
        doses[name] = float(value)
    missing = [name for name in arms if name not in doses]
    if missing:
        raise SystemExit(f"no dose given for arms {missing}")

    # Prompts shared by every arm within each category; the analysis is fully paired.
    per_category: dict[str, tuple[list[str], dict[str, np.ndarray]]] = {}
    all_categories = PRIMARY + FAMILIES["CLIPScore"]
    for category in all_categories:
        loaded = {name: load_scores(path, label, args.step, category)
                  for name, (label, path) in arms.items()}
        if any(not v for v in loaded.values()):
            continue
        shared = sorted(set.intersection(*(set(v) for v in loaded.values())))
        if not shared:
            continue
        per_category[category] = (
            shared,
            {name: np.array([loaded[name][p] for p in shared]) for name in arms},
        )
    present_primary = [c for c in PRIMARY if c in per_category]
    if len(present_primary) != len(PRIMARY):
        raise SystemExit(
            f"missing primary categories: {sorted(set(PRIMARY) - set(present_primary))}"
        )

    rng = np.random.default_rng(args.seed)
    # picks[category] -> (draws, n) resampled prompt indices, shared by all arms (paired).
    picks = {c: rng.integers(0, len(v[0]), size=(args.draws, len(v[0])))
             for c, v in per_category.items()}

    def category_draws(name: str, category: str) -> np.ndarray:
        values = per_category[category][1][name]
        return values[picks[category]].mean(axis=1)

    def weighted(name: str, categories: list[str]) -> tuple[float, np.ndarray]:
        point = float(np.mean([per_category[c][1][name].mean() for c in categories]))
        samples = np.mean([category_draws(name, c) for c in categories], axis=0)
        return point, samples

    names = list(arms)
    primary_point = {}
    primary_draws = {}
    for name in names:
        primary_point[name], primary_draws[name] = weighted(name, present_primary)

    d = np.array([doses[name] for name in names], dtype=np.float64)
    d_centered = d - d.mean()
    denominator = float((d_centered ** 2).sum())

    stacked_point = np.array([primary_point[name] for name in names])
    slope_point = float((d_centered * stacked_point).sum() / denominator)
    stacked_draws = np.stack([primary_draws[name] for name in names], axis=0)  # (arms, draws)
    slope_draws = (d_centered[:, None] * stacked_draws).sum(axis=0) / denominator

    order = [n for n in ("correct", "counterfactual", "random", "inverted") if n in arms]
    ordering_ok = np.ones(args.draws, dtype=bool)
    for lhs, rhs in zip(order, order[1:]):
        ordering_ok &= primary_draws[lhs] > primary_draws[rhs]

    contrasts = {}
    reference = args.reference_arm
    for name in names:
        if name == reference:
            continue
        diff_point = primary_point[reference] - primary_point[name]
        diff_draws = primary_draws[reference] - primary_draws[name]
        contrasts[f"{reference}_minus_{name}"] = {
            "delta": diff_point,
            "ci95": [float(np.percentile(diff_draws, 2.5)),
                     float(np.percentile(diff_draws, 97.5))],
            "p_gt_zero": float((diff_draws > 0).mean()),
            "dose_gap": doses[name] - doses[reference],
        }

    families = {}
    for family, categories in FAMILIES.items():
        present = [c for c in categories if c in per_category]
        if not present:
            continue
        entry = {"categories": present, "arms": {}, "contrasts": {}}
        fam_draws = {}
        for name in names:
            point, draws_ = weighted(name, present)
            fam_draws[name] = draws_
            entry["arms"][name] = {
                "mean": point,
                "ci95": [float(np.percentile(draws_, 2.5)), float(np.percentile(draws_, 97.5))],
            }
        fam_stack = np.stack([fam_draws[name] for name in names], axis=0)
        fam_slope = (d_centered[:, None] * fam_stack).sum(axis=0) / denominator
        entry["dose_slope"] = {
            "point": float(
                (d_centered * np.array([entry["arms"][n]["mean"] for n in names])).sum()
                / denominator
            ),
            "ci95": [float(np.percentile(fam_slope, 2.5)), float(np.percentile(fam_slope, 97.5))],
            "p_lt_zero": float((fam_slope < 0).mean()),
        }
        for name in names:
            if name == reference:
                continue
            diff = fam_draws[reference] - fam_draws[name]
            entry["contrasts"][f"{reference}_minus_{name}"] = {
                "delta": entry["arms"][reference]["mean"] - entry["arms"][name]["mean"],
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "p_gt_zero": float((diff > 0).mean()),
            }
        families[family] = entry

    categories_out = {}
    for category, (shared, values) in per_category.items():
        categories_out[category] = {
            "n": len(shared),
            "arms": {name: float(values[name].mean()) for name in names},
        }

    passed = float(np.percentile(slope_draws, 97.5)) < 0
    report = {
        "step": args.step,
        "arms": {name: {"label": label, "scores_dir": str(path), "dose": doses[name]}
                 for name, (label, path) in arms.items()},
        "prompts_by_category": {c: len(v[0]) for c, v in per_category.items()},
        "bootstrap": {"draws": args.draws, "seed": args.seed,
                      "scheme": "prompts resampled within category, paired across arms"},
        "absolute_primary": {
            name: {
                "mean": primary_point[name],
                "ci95": [float(np.percentile(primary_draws[name], 2.5)),
                         float(np.percentile(primary_draws[name], 97.5))],
            }
            for name in names
        },
        "primary_dose_slope": {
            "point": slope_point,
            "ci95": [float(np.percentile(slope_draws, 2.5)),
                     float(np.percentile(slope_draws, 97.5))],
            "p_lt_zero": float((slope_draws < 0).mean()),
            "preregistered_gate": "95% interval entirely below zero",
            "pass": bool(passed),
        },
        "ordering_probability": {
            "order": order,
            "p_full_ordering": float(ordering_ok.mean()),
        },
        "contrasts": contrasts,
        "families": families,
        "categories": categories_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        s = report["primary_dose_slope"]
        lines = [
            "# Phase-FP fixed-pair dose response",
            "",
            f"Paired prompt bootstrap, {args.draws} draws, seed {args.seed}. "
            "Equal-category-weighted over the seven primary CompBench categories.",
            "",
            "## Preregistered primary: dose slope",
            "",
            f"**slope = {s['point']:+.5f}**, 95% CI [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}], "
            f"P(slope < 0) = {s['p_lt_zero']:.4f}",
            "",
            f"Gate ({s['preregistered_gate']}): **{'PASS' if s['pass'] else 'FAIL'}**",
            "",
            f"Probability of the full ordering {' > '.join(order)}: "
            f"**{report['ordering_probability']['p_full_ordering']:.4f}**",
            "",
            "## Absolute score by arm",
            "",
            "| arm | dose | CompBench primary | 95% CI |",
            "|---|---:|---:|---|",
        ]
        for name in names:
            a = report["absolute_primary"][name]
            lines.append(
                f"| `{name}` | {doses[name]:.4f} | {a['mean']:.5f} | "
                f"[{a['ci95'][0]:.5f}, {a['ci95'][1]:.5f}] |"
            )
        lines += [
            "",
            f"## Contrasts against `{reference}`",
            "",
            "| contrast | dose gap | Δ | 95% CI | P(Δ>0) |",
            "|---|---:|---:|---|---:|",
        ]
        for key, c in contrasts.items():
            lines.append(
                f"| `{key}` | {c['dose_gap']:.4f} | {c['delta']:+.5f} | "
                f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] | {c['p_gt_zero']:.4f} |"
            )
        lines += [
            "",
            "## By evaluator family",
            "",
            "UniDet shares no architecture with the VQAScore selector that produced the training "
            "labels, so its column is the part of any effect not explainable by scorer circularity.",
            "",
            "| family | dose slope | 95% CI | P(<0) | correct − counterfactual | 95% CI |",
            "|---|---:|---|---:|---:|---|",
        ]
        for family, entry in families.items():
            slope = entry["dose_slope"]
            key = f"{reference}_minus_counterfactual"
            c = entry["contrasts"].get(key)
            cf = (f"{c['delta']:+.5f} | [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]"
                  if c else "— | —")
            lines.append(
                f"| **{family}** | {slope['point']:+.5f} | "
                f"[{slope['ci95'][0]:+.5f}, {slope['ci95'][1]:+.5f}] | "
                f"{slope['p_lt_zero']:.4f} | {cf} |"
            )
        lines += ["", "## Per category (absolute)", "",
                  "| category | n | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|---:|" + "---:|" * len(names)]
        for category, entry in categories_out.items():
            row = " | ".join(f"{entry['arms'][n]:.5f}" for n in names)
            lines.append(f"| {category} | {entry['n']} | {row} |")
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "primary_dose_slope": report["primary_dose_slope"],
        "ordering_probability": report["ordering_probability"],
        "absolute_primary": report["absolute_primary"],
        "contrasts": contrasts,
    }, indent=2))


if __name__ == "__main__":
    main()
