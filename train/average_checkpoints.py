#!/usr/bin/env python3
"""Uniform weight average of several checkpoints of one run.

Long runs at a fixed-size normalised step (the gradient clip is active on every update and there
is no EMA) oscillate on a plateau: consecutive checkpoints of one run can differ by more than the
effect being measured. Averaging the last few checkpoints removes that oscillation without
retraining. The output has the same {"model", "step", "variant"} layout as checkpoint_final.pt,
with "step" set to the latest member, so the evaluation launcher accepts it unchanged.

    python train/average_checkpoints.py --run checkpoints/dino_patch_s0 \
        --steps 40000,45000,50000,55000,final --out checkpoints/dino_patch_s0/checkpoint_avg_last5.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint directory of one training run")
    ap.add_argument("--steps", default="40000,45000,50000,55000,final",
                    help="comma-separated checkpoint steps to average; 'final' = checkpoint_final.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    files = [run / ("checkpoint_final.pt" if s == "final" else f"checkpoint_step{s}.pt") for s in args.steps.split(",")]
    for f in files:
        assert f.is_file(), f
    acc, n, meta = None, 0, None
    for f in files:
        ck = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        sd = ck["model"]
        if acc is None:
            acc = {k: v.detach().to(torch.float32).clone() for k, v in sd.items()}
            meta = {"step": int(ck["step"]), "variant": ck.get("variant")}
        else:
            assert set(sd) == set(acc), "state-dict keys differ"
            for k, v in sd.items():
                acc[k] += v.detach().to(torch.float32)
            meta["step"] = max(meta["step"], int(ck["step"]))
        n += 1
        print(f"[avg] {f.name} (step {ck['step']})", flush=True)
    for k in acc:
        acc[k] /= n
    torch.save({"model": acc, "step": meta["step"], "variant": meta["variant"],
                "averaged_from": [str(f) for f in files]}, args.out)
    print(f"[avg] wrote {args.out}: mean of {n} checkpoints, step field {meta['step']}")


if __name__ == "__main__":
    main()
