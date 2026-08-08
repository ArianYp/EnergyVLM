#!/usr/bin/env python3
"""CPU-only integrity gate for the fixed-unordered-pair cache and its trainer path.

This runs before any GPU arm. It proves the property the whole design rests on: the four arms see
byte-identical unordered pairs and differ only in the label sign.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseFP.build_fixed_pair_selection import ARMS, pair_key
from phaseI.train_preference import (
    SIGMA_BANDS,
    fixed_pair,
    load_selection,
    pair_manifest,
    assert_pair_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--expected_records", type=int, default=5042)
    parser.add_argument("--expected_reversals", type=int, default=1373)
    parser.add_argument("--min_original_margin", type=float, default=0.20)
    args = parser.parse_args()

    root = Path(args.cache_root)
    arms = {arm: load_selection(root / arm) for arm in ARMS}

    for arm, records in arms.items():
        assert len(records) == args.expected_records, (arm, len(records))

    # 1. Cross-arm unordered-pair identity, checked from the raw indices rather than only the key.
    reference = arms["correct"]
    manifest = pair_manifest(reference)
    for arm, records in arms.items():
        assert pair_manifest(records) == manifest, f"{arm} pair manifest differs"
        for lhs, rhs in zip(reference, records):
            assert int(lhs["idx"]) == int(rhs["idx"])
            assert {int(lhs["pair_a"]), int(lhs["pair_b"])} == {
                int(rhs["pair_a"]), int(rhs["pair_b"])
            }, f"{arm} pair membership differs at idx {lhs['idx']}"
            assert lhs["prompt"] == rhs["prompt"]
            assert int(lhs["seed_base"]) == int(rhs["seed_base"])
            assert lhs["original_endpoint_vqa"] == rhs["original_endpoint_vqa"]
            assert rhs["pair_key"] == pair_key(
                int(rhs["idx"]), int(rhs["seed_base"]), int(rhs["pair_a"]), int(rhs["pair_b"])
            ), f"{arm} pair_key does not match its own indices at idx {rhs['idx']}"

    # 2. Serialized orientation is self-consistent and never re-derived from scores at train time.
    orientation = {}
    margins = {}
    for arm, records in arms.items():
        orientation[arm] = {}
        margins[arm] = []
        for record in records:
            positive, negative, margin, flipped = fixed_pair(record)
            assert {positive, negative} == {int(record["pair_a"]), int(record["pair_b"])}
            assert flipped == (int(record["orientation"]) < 0)
            orientation[arm][int(record["idx"])] = int(record["orientation"])
            margins[arm].append(margin)

    n = args.expected_records
    # correct: always the original-prompt sign, so every assigned margin is non-negative.
    assert all(margin >= 0 for margin in margins["correct"])
    assert sum(margins["correct"]) / n >= args.min_original_margin
    # inverted: exactly opposite, so its assigned margins are the negation of correct's.
    assert all(
        math.isclose(margins["inverted"][i], -margins["correct"][i], rel_tol=1e-12)
        for i in range(n)
    )
    assert all(orientation["inverted"][k] == -orientation["correct"][k] for k in orientation["correct"])
    # random: exactly balanced and near-zero mean assigned margin.
    positives = sum(1 for v in orientation["random"].values() if v > 0)
    assert abs(positives - n / 2) <= 0.5, positives
    assert abs(sum(margins["random"]) / n) < 0.02, sum(margins["random"]) / n
    # counterfactual: disagrees with correct on exactly the reversal subset.
    disagree = [k for k, v in orientation["counterfactual"].items() if v != orientation["correct"][k]]
    assert len(disagree) == args.expected_reversals, (len(disagree), args.expected_reversals)
    reversal_flags = {
        int(r["idx"]) for r in arms["correct"] if r["pair_reverses_under_counterfactual"]
    }
    assert set(disagree) == reversal_flags, "reversal flags disagree with counterfactual signs"

    # 3. On the reversal subset the two headline arms give the same two images opposite labels.
    #    This is the sharp orientation-only population.
    for idx in list(reversal_flags)[:50]:
        c = next(r for r in arms["correct"] if int(r["idx"]) == idx)
        f = next(r for r in arms["counterfactual"] if int(r["idx"]) == idx)
        assert c["positive_idx"] == f["negative_idx"] and c["negative_idx"] == f["positive_idx"]

    # 4. The manifest gate rejects a tampered cache.
    manifest_path = Path(args.cache_root) / "pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    for arm, records in arms.items():
        assert_pair_manifest(records, manifest_path)
    tampered = dict(manifest)
    first = next(iter(tampered))
    tampered[first] = "0" * 64
    tampered_path = Path(args.cache_root) / "_tampered_manifest.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True))
    try:
        assert_pair_manifest(arms["correct"], tampered_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("manifest gate accepted a tampered pair key")
    finally:
        tampered_path.unlink()

    # 5. Sigma bands partition the unit interval without overlap.
    assert SIGMA_BANDS["all"] == (0.0, 1.0)
    ordered = [SIGMA_BANDS[name] for name in ("low", "mid", "high")]
    assert ordered[0][0] == 0.0 and math.isclose(ordered[-1][1], 1.0)
    for (_, hi), (lo, _) in zip(ordered, ordered[1:]):
        assert math.isclose(hi, lo)

    # 6. Loss sign: at the frozen reference the logit is 0, and descent lowers the model gap.
    beta = 100.0
    model_gap = torch.tensor(0.03, requires_grad=True)
    logit = -beta * (model_gap - torch.tensor(0.03))
    loss = -torch.nn.functional.logsigmoid(logit)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6)
    loss.backward()
    assert model_gap.grad is not None and float(model_gap.grad) > 0

    report = {
        "records": n,
        "arms": list(ARMS),
        "cross_arm_pair_identity": True,
        "reversal_records": len(reversal_flags),
        "reversal_fraction": len(reversal_flags) / n,
        "correct_mean_assigned_margin": sum(margins["correct"]) / n,
        "random_mean_assigned_margin": sum(margins["random"]) / n,
        "inverted_mean_assigned_margin": sum(margins["inverted"]) / n,
        "counterfactual_mean_assigned_margin": sum(margins["counterfactual"]) / n,
        "random_positive_fraction": positives / n,
        "manifest_sha_path": str(manifest_path),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
