#!/usr/bin/env python3
"""CPU-only integrity checks for the Phase-I cache and preference loss sign."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phaseI.train_preference import (
    enable_trainable_parameters,
    load_selection,
    preferred_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_dir", default=str(Path(__file__).resolve().parent.parent / "phaseC"))
    parser.add_argument("--expected_records", type=int, default=5559)
    parser.add_argument("--min_oracle_margin", type=float, default=0.20)
    args = parser.parse_args()

    frozen = torch.nn.Linear(3, 2)
    frozen.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert enable_trainable_parameters(frozen) == 8
    assert all(parameter.requires_grad for parameter in frozen.parameters())

    records = load_selection(args.selection_dir)
    assert len(records) == args.expected_records, (len(records), args.expected_records)

    oracle_margins = []
    placebo_margins = []
    flips = 0
    for record in records:
        pos, neg, margin, flip = preferred_pair(record, "oracle", 20260806)
        scores = record["endpoint_vqa"]
        assert scores[pos] == max(scores)
        assert scores[neg] == min(scores)
        assert margin >= 0
        oracle_margins.append(margin)

        pos, neg, margin, flip = preferred_pair(record, "randomized", 20260806)
        assert {pos, neg} == {
            max(range(len(scores)), key=scores.__getitem__),
            min(range(len(scores)), key=scores.__getitem__),
        }
        placebo_margins.append(margin)
        flips += int(flip)

    flip_fraction = flips / len(records)
    assert 0.47 < flip_fraction < 0.53, flip_fraction
    assert sum(oracle_margins) / len(oracle_margins) > args.min_oracle_margin
    assert abs(sum(placebo_margins) / len(placebo_margins)) < 0.01

    # At the frozen-reference initialization the DPO logit is 0 and loss is log(2).
    beta = 100.0
    model_gap = torch.tensor(0.03, requires_grad=True)
    reference_gap = torch.tensor(0.03)
    logit = -beta * (model_gap - reference_gap)
    loss = -torch.nn.functional.logsigmoid(logit)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6)
    loss.backward()
    # Gradient descent lowers model_gap = error_positive - error_negative, as desired.
    assert model_gap.grad is not None and float(model_gap.grad) > 0

    print({
        "records": len(records),
        "oracle_top_bottom_margin": sum(oracle_margins) / len(oracle_margins),
        "placebo_assigned_margin": sum(placebo_margins) / len(placebo_margins),
        "placebo_flip_fraction": flip_fraction,
        "initial_preference_loss": float(loss),
        "d_loss_d_model_gap": float(model_gap.grad),
    })


if __name__ == "__main__":
    main()
