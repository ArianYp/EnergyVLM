#!/usr/bin/env python3
"""Task F gate — prove the four fixed-pair arms are a valid label-only intervention.

A cache that merely *claims* identical pairs is not enough: the trainer rolls the teacher out at
run time, so the images it actually trains on must be checked after the fact. Each arm hashes its
teacher endpoints in canonical pair order; if the four hashes agree step-for-step, the arms
provably differ only in the sign of the label.

Also checks that the orientation actually reaches the loss, by requiring the arms to disagree on
the assigned margin exactly where the cache says they should.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_telemetry(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="Repeat once per arm, e.g. --arm correct=checkpoints/phaseFP/phaseFP_correct_123",
    )
    parser.add_argument("--min_steps", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, run_dir = spec.partition("=")
        arms[name] = load_telemetry(Path(run_dir) / "telemetry.jsonl")

    failures = []
    if len(arms) < 2:
        raise SystemExit("need at least two arms to compare")
    for name, rows in arms.items():
        if len(rows) < args.min_steps:
            failures.append(f"{name}: only {len(rows)} telemetry rows (< {args.min_steps})")

    # 1. Same records, same order.
    reference_name = "correct" if "correct" in arms else sorted(arms)[0]
    reference = arms[reference_name]
    common = min(len(rows) for rows in arms.values())
    for name, rows in arms.items():
        for i in range(common):
            if rows[i]["idx"] != reference[i]["idx"] or rows[i]["step"] != reference[i]["step"]:
                failures.append(
                    f"{name}: data order diverges at row {i} "
                    f"({rows[i]['step']},{rows[i]['idx']}) vs "
                    f"({reference[i]['step']},{reference[i]['idx']})"
                )
                break

    # 2. Bit-identical teacher endpoints — the core claim.
    endpoint_mismatches = []
    for i in range(common):
        hashes = {name: rows[i]["endpoints_sha256"] for name, rows in arms.items()}
        if len(set(hashes.values())) != 1:
            endpoint_mismatches.append({"row": i, "step": reference[i]["step"],
                                        "idx": reference[i]["idx"], "hashes": hashes})
    if endpoint_mismatches:
        failures.append(
            f"teacher endpoints differ across arms on {len(endpoint_mismatches)}/{common} rows"
        )

    # 3. Same unordered pair key everywhere.
    for i in range(common):
        keys = {rows[i].get("pair_key") for rows in arms.values()}
        if len(keys) != 1:
            failures.append(f"pair_key differs across arms at row {i}")
            break

    # 4. The orientation reaches the loss: arms must disagree where the cache says they do.
    orientation_rows = defaultdict(dict)
    for name, rows in arms.items():
        for i in range(common):
            orientation_rows[i][name] = rows[i]

    checks = {}
    if "correct" in arms and "inverted" in arms:
        opposed = sum(
            1
            for i in range(common)
            if orientation_rows[i]["correct"]["orientation"]
            == -orientation_rows[i]["inverted"]["orientation"]
        )
        checks["inverted_always_opposed"] = opposed == common
        if opposed != common:
            failures.append(f"inverted arm opposed on only {opposed}/{common} rows")
    if "correct" in arms and "counterfactual" in arms:
        disagree = [
            i
            for i in range(common)
            if orientation_rows[i]["correct"]["orientation"]
            != orientation_rows[i]["counterfactual"]["orientation"]
        ]
        # Every disagreement must coincide with the cache's reversal flag, and vice versa.
        flagged = {i for i in range(common) if orientation_rows[i]["correct"].get("reverses")}
        checks["counterfactual_disagrees_exactly_on_reversals"] = set(disagree) == flagged
        checks["counterfactual_disagreement_rows"] = len(disagree)
        if set(disagree) != flagged:
            failures.append(
                "counterfactual orientation disagreement does not match the reversal flags"
            )

    # 5. The loss is finite and non-degenerate everywhere.
    for name, rows in arms.items():
        bad = [
            r["step"]
            for r in rows
            if not all(
                isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
                for v in [r["logit"], r["delta_theta"], r["delta_ref"], r["grad_norm_pre_clip"]]
            )
        ]
        if bad:
            failures.append(f"{name}: non-finite telemetry at steps {bad[:5]}")
        if any(r["grad_norm_pre_clip"] <= 0 for r in rows):
            failures.append(f"{name}: non-positive gradient norm")

    # 6. Step-zero sanity: the student starts at the frozen reference, so the first logit is ~0.
    for name, rows in arms.items():
        first = rows[0]
        checks[f"{name}_first_logit"] = first["logit"]
        checks[f"{name}_first_sigma_mean"] = sum(first["sigma"]) / len(first["sigma"])

    report = {
        "arms": sorted(arms),
        "rows_compared": common,
        "rows_per_arm": {name: len(rows) for name, rows in arms.items()},
        "endpoint_hash_identical_rows": common - len(endpoint_mismatches),
        "endpoint_mismatch_examples": endpoint_mismatches[:3],
        "checks": checks,
        "failures": failures,
        "pass": not failures,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
