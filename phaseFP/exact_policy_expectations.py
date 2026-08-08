#!/usr/bin/env python3
"""Exact best-of-K headroom from per-candidate CompBench scores.

With every candidate slot scored, each selection policy becomes a per-prompt expectation rather
than a draw:

    Best_i       = CompBench(candidate chosen by VQAScore argmax)      [selection is by VQAScore]
    Uniform_i    = mean_j CompBench(cand_j)
    NonOracle_i  = mean_{j != oracle} CompBench(cand_j)

The mixture identity Uniform = (1/N)Best + (1-1/N)NonOracle then holds by construction, and the
bias from excluding the oracle from the control pool is exactly N/(N-1) — not something to estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex"]


def load(scores_dir: Path, label: str, step: int, category: str) -> dict[str, float]:
    path = scores_dir / f"{label}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {r["prompt"]: float(r["score"]) for r in payload["per_prompt"] if r.get("prompt")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand_scores_dir", required=True)
    parser.add_argument("--selection", required=True,
                        help="selection.json from build_teacher_ceiling.py (has oracle_j per idx)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    selection = {int(r["idx"]): r for r in json.loads(Path(args.selection).read_text())}
    scores_dir = Path(args.cand_scores_dir)

    per_cat = {}
    for category in PRIMARY:
        cand = [load(scores_dir, f"Cand{j}", args.steps, category) for j in range(args.N)]
        if any(not c for c in cand):
            continue
        shared = sorted(set.intersection(*(set(c) for c in cand)))
        # Map prompt -> oracle slot, from the VQAScore-based selection already recorded.
        by_prompt = {r["prompt"]: r for r in selection.values() if "prompt" in r}
        rows = []
        for prompt in shared:
            rec = by_prompt.get(prompt)
            if rec is None:
                continue
            vals = np.array([c[prompt] for c in cand], dtype=np.float64)
            oracle = int(rec["oracle_j"])
            best = vals[oracle]
            uniform = float(vals.mean())
            non_oracle = float(np.delete(vals, oracle).mean())
            rows.append((best, uniform, non_oracle))
        if rows:
            per_cat[category] = np.asarray(rows)  # (n, 3)

    if len(per_cat) != len(PRIMARY):
        raise SystemExit(f"missing categories: {sorted(set(PRIMARY) - set(per_cat))}")

    rng = np.random.default_rng(args.seed)
    picks = {c: rng.integers(0, v.shape[0], size=(args.draws, v.shape[0]))
             for c, v in per_cat.items()}

    def agg(col: int) -> tuple[float, np.ndarray]:
        point = float(np.mean([per_cat[c][:, col].mean() for c in PRIMARY]))
        draws = np.mean([per_cat[c][:, col][picks[c]].mean(axis=1) for c in PRIMARY], axis=0)
        return point, draws

    (b, bd), (u, ud), (n, nd) = agg(0), agg(1), agg(2)
    headroom_u, headroom_u_d = b - u, bd - ud
    headroom_n, headroom_n_d = b - n, bd - nd

    identity_gap = u - ((1.0 / args.N) * b + (1 - 1.0 / args.N) * n)
    exact_bias = args.N / (args.N - 1)

    report = {
        "prompts": int(sum(v.shape[0] for v in per_cat.values())),
        "N": args.N,
        "absolute": {"best": b, "uniform": u, "non_oracle": n},
        "headroom_vs_uniform": {
            "delta": headroom_u,
            "ci95": [float(np.percentile(headroom_u_d, 2.5)),
                     float(np.percentile(headroom_u_d, 97.5))],
        },
        "headroom_vs_non_oracle": {
            "delta": headroom_n,
            "ci95": [float(np.percentile(headroom_n_d, 2.5)),
                     float(np.percentile(headroom_n_d, 97.5))],
        },
        "mixture_identity_residual": identity_gap,
        "control_bias": {
            "exact_N_over_N_minus_1": exact_bias,
            "observed_ratio": headroom_n / headroom_u if headroom_u else None,
        },
        "note": (
            "With exact per-prompt expectations the identity residual should be ~0 (float error "
            "only) and the observed ratio should equal N/(N-1) exactly. Any residual indicates a "
            "prompt-set mismatch between the candidate slots, not sampling noise."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        hu, hn = report["headroom_vs_uniform"], report["headroom_vs_non_oracle"]
        Path(args.md).write_text("\n".join([
            f"# Exact best-of-{args.N} headroom (per-candidate CompBench expectations)",
            "",
            f"{report['prompts']} prompt-category rows; every candidate slot scored, so each policy "
            "is an exact per-prompt expectation rather than a single draw.",
            "",
            "| policy | CompBench primary |",
            "|---|---:|",
            f"| best-of-{args.N} (VQAScore argmax) | {b:.5f} |",
            f"| uniform over all {args.N} | {u:.5f} |",
            f"| uniform over the {args.N - 1} non-oracle | {n:.5f} |",
            "",
            f"- **Headroom vs uniform (unbiased): {hu['delta']:+.5f} "
            f"[{hu['ci95'][0]:+.5f}, {hu['ci95'][1]:+.5f}]**",
            f"- Headroom vs non-oracle (biased): {hn['delta']:+.5f} "
            f"[{hn['ci95'][0]:+.5f}, {hn['ci95'][1]:+.5f}]",
            "",
            f"Mixture-identity residual: {identity_gap:+.2e} (should be ~0).",
            f"Control bias: exact N/(N-1) = {exact_bias:.4f}; "
            f"observed {report['control_bias']['observed_ratio']:.4f}.",
            "",
            "Excluding the oracle from the control pool inflates measured headroom by exactly "
            f"{100 * (exact_bias - 1):.1f}% — this is algebra, not an empirical finding.",
        ]) + "\n")

    print(json.dumps({k: report[k] for k in
                      ("absolute", "headroom_vs_uniform", "headroom_vs_non_oracle",
                       "mixture_identity_residual", "control_bias")}, indent=2))


if __name__ == "__main__":
    main()
