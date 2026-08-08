#!/usr/bin/env python3
"""Task N mechanism probe — does the preference signal actually live at high sigma?

The shared-noise coalescence hypothesis says competition should be strongest where the two branches'
inputs collapse together:

    z+ - z- = (1 - sigma)(x+ - x-)   ->  0   as sigma -> 1,
    u+ - u- = x- - x+                       stays separated.

Training loss alone cannot test this, because the loss averages over the sampled timesteps. The
per-example telemetry keeps sigma alongside each branch's flow error, so the sigma profile of the
error gap is directly measurable — and it is measurable from the MAIN arm, before spending compute
on sigma-band-restricted training runs.

Also compares noise-coupling conditions. Because the independent-noise draw happens after all other
RNG consumption, z+ is bit-identical between conditions and only z- differs, so the two runs are a
matched pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load(path: Path) -> dict[str, np.ndarray]:
    sigma, e_pos, e_neg, r_pos, r_neg, step = [], [], [], [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n = len(row["sigma"])
        sigma.extend(row["sigma"])
        e_pos.extend(row["e_theta_pos"])
        e_neg.extend(row["e_theta_neg"])
        r_pos.extend(row["e_ref_pos"])
        r_neg.extend(row["e_ref_neg"])
        step.extend([row["step"]] * n)
    return {
        "sigma": np.asarray(sigma),
        "e_pos": np.asarray(e_pos),
        "e_neg": np.asarray(e_neg),
        "r_pos": np.asarray(r_pos),
        "r_neg": np.asarray(r_neg),
        "step": np.asarray(step),
    }


def profile(data: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    sigma = data["sigma"]
    # Per-example reference-relative gap: the quantity the loss drives negative.
    gap = (data["e_pos"] - data["e_neg"]) - (data["r_pos"] - data["r_neg"])
    winner = data["r_pos"] - data["e_pos"]   # >0 means the student improved on the winner
    loser = data["e_neg"] - data["r_neg"]    # >0 means the student degraded the loser
    rng = np.random.default_rng(seed)

    bands = []
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        mask = (sigma >= lo) & (sigma < hi if hi < 1.0 else sigma <= hi)
        n = int(mask.sum())
        if n < 8:
            bands.append({"sigma_range": [lo, hi], "n": n})
            continue
        values = gap[mask]
        picks = rng.integers(0, n, size=(draws, n))
        samples = values[picks].mean(axis=1)
        bands.append({
            "sigma_range": [lo, hi],
            "n": n,
            "mean_sigma": float(sigma[mask].mean()),
            "gap_mean": float(values.mean()),
            "gap_ci95": [float(np.percentile(samples, 2.5)),
                         float(np.percentile(samples, 97.5))],
            "abs_gap_mean": float(np.abs(values).mean()),
            "winner_improvement_mean": float(winner[mask].mean()),
            "loser_degradation_mean": float(loser[mask].mean()),
            "winner_share": float(
                np.abs(winner[mask]).mean()
                / max(np.abs(winner[mask]).mean() + np.abs(loser[mask]).mean(), 1e-12)
            ),
            "error_scale_positive": float(data["e_pos"][mask].mean()),
            # The mechanism-relevant quantity: |gap| relative to how large the branch errors are in
            # this band. Raw |gap| is U-shaped simply because flow errors are large at both ends of
            # the sigma range, which would otherwise be mistaken for a signal at low sigma.
            "abs_gap_scale_normalized": float(
                np.abs(values).mean()
                / max((data["e_pos"][mask] + data["e_neg"][mask]).mean(), 1e-12)
            ),
        })

    # Is |gap| monotone in sigma? Correlate on the raw examples rather than on band means, and
    # normalise by the branch error scale so the answer is not just "errors grow with sigma".
    scale = np.maximum(data["e_pos"] + data["e_neg"], 1e-12)
    normalized = np.abs(gap) / scale
    finite = np.isfinite(normalized)
    corr_raw = float(np.corrcoef(sigma[finite], np.abs(gap)[finite])[0, 1])
    corr_norm = float(np.corrcoef(sigma[finite], normalized[finite])[0, 1])
    picks = rng.integers(0, finite.sum(), size=(draws, int(finite.sum())))
    s_f, n_f = sigma[finite], normalized[finite]
    corr_draws = np.array([
        np.corrcoef(s_f[p], n_f[p])[0, 1] for p in picks[: min(draws, 2000)]
    ])
    return {
        "examples": int(sigma.size),
        "bands": bands,
        "corr_sigma_abs_gap_raw": corr_raw,
        "corr_sigma_abs_gap_scale_normalized": corr_norm,
        "corr_normalized_ci95": [float(np.percentile(corr_draws, 2.5)),
                                 float(np.percentile(corr_draws, 97.5))],
        "overall_gap_mean": float(gap.mean()),
        "overall_winner_improvement": float(winner.mean()),
        "overall_loser_degradation": float(loser.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        path = Path(run_dir) / "telemetry.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        runs[name] = profile(load(path), args.draws, args.seed)

    report = {"runs": runs, "band_edges": BAND_EDGES, "draws": args.draws, "seed": args.seed}

    # Matched shared-vs-independent comparison, if both are present.
    if "shared" in runs and "independent" in runs:
        report["noise_coupling"] = {
            "shared_corr": runs["shared"]["corr_sigma_abs_gap_scale_normalized"],
            "independent_corr": runs["independent"]["corr_sigma_abs_gap_scale_normalized"],
            "shared_overall_gap": runs["shared"]["overall_gap_mean"],
            "independent_overall_gap": runs["independent"]["overall_gap_mean"],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        lines = [
            "# Preference signal versus noise level",
            "",
            "Per-example rows from `telemetry.jsonl`. The gap is the reference-relative quantity the "
            "loss drives negative: `(e_theta+ - e_theta-) - (e_0+ - e_0-)`.",
            "",
        ]
        for name, entry in runs.items():
            lines += [
                f"## `{name}` ({entry['examples']:,} examples)",
                "",
                f"- corr(sigma, |gap|), raw: **{entry['corr_sigma_abs_gap_raw']:+.4f}**",
                f"- corr(sigma, |gap| / branch error scale): "
                f"**{entry['corr_sigma_abs_gap_scale_normalized']:+.4f}** "
                f"[{entry['corr_normalized_ci95'][0]:+.4f}, "
                f"{entry['corr_normalized_ci95'][1]:+.4f}]",
                "",
                "The scale-normalised correlation is the one that tests coalescence: the raw "
                "correlation would be positive merely because flow errors grow with sigma.",
                "",
                "| sigma band | n | mean gap | 95% CI | mean \\|gap\\| | **\\|gap\\| / error scale** | winner share | branch error scale |",
                "|---|---:|---:|---|---:|---:|---:|---:|",
            ]
            for band in entry["bands"]:
                if "gap_mean" not in band:
                    lines.append(
                        f"| [{band['sigma_range'][0]:.1f}, {band['sigma_range'][1]:.1f}) | "
                        f"{band['n']} | — | — | — | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| [{band['sigma_range'][0]:.1f}, {band['sigma_range'][1]:.1f}) | "
                    f"{band['n']} | {band['gap_mean']:+.6f} | "
                    f"[{band['gap_ci95'][0]:+.6f}, {band['gap_ci95'][1]:+.6f}] | "
                    f"{band['abs_gap_mean']:.6f} | **{band['abs_gap_scale_normalized']:.4f}** | "
                    f"{band['winner_share']:.3f} | "
                    f"{band['error_scale_positive']:.4f} |"
                )
            lines += [
                "",
                "Read the **bold** column, not raw |gap|. Raw |gap| is U-shaped because flow errors "
                "are large at both ends of the sigma range; dividing by the branch error scale "
                "removes that and leaves the quantity the coalescence hypothesis predicts.",
                "",
            ]
        if "noise_coupling" in report:
            nc = report["noise_coupling"]
            lines += [
                "## Shared versus independent corruption noise",
                "",
                f"- scale-normalised corr(sigma, |gap|): shared {nc['shared_corr']:+.4f} vs "
                f"independent {nc['independent_corr']:+.4f}",
                f"- overall mean gap: shared {nc['shared_overall_gap']:+.6f} vs "
                f"independent {nc['independent_overall_gap']:+.6f}",
                "",
                "Coalescence is supported only if the shared condition shows a materially stronger "
                "sigma dependence AND wins on the primary endpoint. A flat profile in both retires "
                "the mechanism.",
            ]
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({name: {
        "examples": e["examples"],
        "corr_normalized": e["corr_sigma_abs_gap_scale_normalized"],
        "corr_ci": e["corr_normalized_ci95"],
        "overall_gap": e["overall_gap_mean"],
    } for name, e in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
