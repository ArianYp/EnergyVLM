#!/usr/bin/env python3
"""Across-seed analysis of the fixed-pair dose response (promotion gate 3).

The roadmap requires confidence intervals that reflect **both** prompt sampling and training
randomness, and explicitly asks for them to be reported separately rather than conflated. With three
training seeds a random-effects model is not credible, so this reports:

  * per seed: the dose slope and each contrast, with a paired PROMPT bootstrap;
  * across seeds: the mean and the full range, plus the sign-consistency of every effect.

It deliberately does NOT pool prompts across seeds into one narrow interval. Doing so would treat
three draws from the training-randomness distribution as if they were independent prompt samples and
would understate the uncertainty that gate 3 exists to expose.
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
}
DEFAULT_DOSE = {"correct": 0.0, "counterfactual": 0.10816, "random": 0.28663, "inverted": 0.57291}
ARMS = list(DEFAULT_DOSE)


def load(scores_dir: Path, label: str, category: str, step: int = 4) -> dict[str, float]:
    path = scores_dir / f"{label}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        r["prompt"]: float(r["score"]) for r in payload["per_prompt"] if r.get("prompt") is not None
    }


def seed_stats(labels: dict[str, tuple[str, Path]], draws: int, seed: int) -> dict | None:
    """labels: arm -> (label, scores_dir). Returns slope/contrasts with a prompt bootstrap."""
    per = {}
    for category in PRIMARY:
        loaded = {a: load(d, lab, category) for a, (lab, d) in labels.items()}
        if any(not v for v in loaded.values()):
            return None
        shared = sorted(set.intersection(*(set(v) for v in loaded.values())))
        per[category] = (shared, {a: np.array([loaded[a][p] for p in shared]) for a in labels})

    rng = np.random.default_rng(seed)
    picks = {c: rng.integers(0, len(v[0]), size=(draws, len(v[0]))) for c, v in per.items()}

    def weighted(arm, categories):
        point = float(np.mean([per[c][1][arm].mean() for c in categories]))
        samples = np.mean([per[c][1][arm][picks[c]].mean(axis=1) for c in categories], axis=0)
        return point, samples

    point, sample = {}, {}
    for arm in labels:
        point[arm], sample[arm] = weighted(arm, PRIMARY)

    names = list(labels)
    d = np.array([DEFAULT_DOSE[a] for a in names])
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    slope_point = float((dc * np.array([point[a] for a in names])).sum() / den)
    slope_draws = (dc[:, None] * np.stack([sample[a] for a in names])).sum(axis=0) / den

    ordering = np.ones(draws, dtype=bool)
    ordered = [a for a in ("correct", "counterfactual", "random", "inverted") if a in labels]
    for lhs, rhs in zip(ordered, ordered[1:]):
        ordering &= sample[lhs] > sample[rhs]

    out = {
        "absolute": {a: point[a] for a in names},
        "dose_slope": {
            "point": slope_point,
            "ci95": [float(np.percentile(slope_draws, 2.5)),
                     float(np.percentile(slope_draws, 97.5))],
        },
        "ordering_probability": float(ordering.mean()),
        "contrasts": {},
        "families": {},
    }
    for arm in names:
        if arm == "correct":
            continue
        diff = sample["correct"] - sample[arm]
        out["contrasts"][f"correct_minus_{arm}"] = {
            "delta": point["correct"] - point[arm],
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        }
    for family, categories in FAMILIES.items():
        fp, fs = {}, {}
        for arm in names:
            fp[arm], fs[arm] = weighted(arm, categories)
        fslope = (dc[:, None] * np.stack([fs[a] for a in names])).sum(axis=0) / den
        out["families"][family] = {
            "dose_slope": float((dc * np.array([fp[a] for a in names])).sum() / den),
            "dose_slope_ci95": [float(np.percentile(fslope, 2.5)),
                                float(np.percentile(fslope, 97.5))],
            "correct_minus_counterfactual": (
                fp["correct"] - fp["counterfactual"] if "counterfactual" in fp else None
            ),
        }
    return out


def summarize_across(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        # sd with n=3 is reported for completeness but is a poor estimate; range is more honest.
        "sd": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "all_same_sign": bool(np.all(array > 0) or np.all(array < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_set", action="append", required=True,
        metavar="NAME=arm:LABEL:DIR,arm:LABEL:DIR,...",
        help="One per training seed.",
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260808)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    seeds = {}
    for spec in args.seed_set:
        name, _, rest = spec.partition("=")
        labels = {}
        for part in rest.split(","):
            arm, label, directory = part.split(":")
            labels[arm] = (label, Path(directory))
        seeds[name] = labels

    results, skipped = {}, []
    for name, labels in seeds.items():
        stats = seed_stats(labels, args.draws, args.bootstrap_seed)
        if stats is None:
            skipped.append(name)
            continue
        results[name] = stats
    if not results:
        raise SystemExit(f"no complete seed sets (skipped: {skipped})")

    across = {
        "dose_slope": summarize_across([r["dose_slope"]["point"] for r in results.values()]),
        "ordering_probability": summarize_across(
            [r["ordering_probability"] for r in results.values()]),
    }
    for key in results[next(iter(results))]["contrasts"]:
        across[key] = summarize_across([r["contrasts"][key]["delta"] for r in results.values()])
    for family in FAMILIES:
        across[f"{family}_dose_slope"] = summarize_across(
            [r["families"][family]["dose_slope"] for r in results.values()])

    report = {
        "seeds_analyzed": sorted(results),
        "seeds_skipped_incomplete": skipped,
        "per_seed": results,
        "across_seeds": across,
        "estimand_note": (
            "Per-seed intervals are prompt bootstraps and describe prompt sampling only. The "
            "across-seed mean/range describes training randomness. They are reported separately "
            "and must not be combined into a single interval with only three seeds."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        lines = [
            "# Fixed-pair dose response across training seeds",
            "",
            f"Seeds analyzed: {', '.join(sorted(results))}"
            + (f" (skipped, incomplete: {', '.join(skipped)})" if skipped else ""),
            "",
            "Per-seed intervals are **prompt** bootstraps. The across-seed row describes **training**",
            "randomness. They are different sources of variation and are not combined.",
            "",
            "## Dose slope (preregistered primary)",
            "",
            "| seed | slope | 95% CI (prompts) | ordering probability |",
            "|---|---:|---|---:|",
        ]
        for name in sorted(results):
            r = results[name]
            s = r["dose_slope"]
            lines.append(
                f"| {name} | {s['point']:+.5f} | [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}] | "
                f"{r['ordering_probability']:.4f} |"
            )
        a = across["dose_slope"]
        lines += [
            f"| **across seeds** | **{a['mean']:+.5f}** | range [{a['min']:+.5f}, {a['max']:+.5f}] | "
            f"{across['ordering_probability']['mean']:.4f} |",
            "",
            f"All seeds same sign: **{a['all_same_sign']}**. Across-seed range {a['range']:.5f}.",
            "",
            "## Absolute score by arm and seed",
            "",
            "| seed | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
            "|---|" + "---:|" * len(ARMS),
        ]
        for name in sorted(results):
            row = results[name]["absolute"]
            lines.append(f"| {name} | " + " | ".join(
                f"{row[a]:.5f}" if a in row else "—" for a in ARMS) + " |")
        lines += ["", "## Contrasts against `correct`", "",
                  "| contrast | " + " | ".join(sorted(results)) + " | mean | range | same sign |",
                  "|---|" + "---:|" * (len(results) + 2) + ":-:|"]
        for key in sorted(results[next(iter(results))]["contrasts"]):
            vals = [results[s]["contrasts"][key]["delta"] for s in sorted(results)]
            summ = across[key]
            lines.append(
                f"| `{key}` | " + " | ".join(f"{v:+.5f}" for v in vals)
                + f" | {summ['mean']:+.5f} | {summ['range']:.5f} | "
                + ("yes" if summ["all_same_sign"] else "**no**") + " |"
            )
        lines += ["", "## Dose slope by evaluator family", "",
                  "| family | " + " | ".join(sorted(results)) + " | mean | same sign |",
                  "|---|" + "---:|" * (len(results) + 1) + ":-:|"]
        for family in FAMILIES:
            vals = [results[s]["families"][family]["dose_slope"] for s in sorted(results)]
            summ = across[f"{family}_dose_slope"]
            lines.append(
                f"| {family} | " + " | ".join(f"{v:+.5f}" for v in vals)
                + f" | {summ['mean']:+.5f} | " + ("yes" if summ["all_same_sign"] else "**no**") + " |"
            )
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({"seeds": sorted(results), "skipped": skipped,
                      "across_seeds": across["dose_slope"]}, indent=2))


if __name__ == "__main__":
    main()
