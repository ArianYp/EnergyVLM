#!/usr/bin/env python3
"""Uniform weight average of several checkpoints of one run (checkpoint averaging / LAWA-style).

The 118k runs save every 5k steps and their per-checkpoint CompBench swings by ~0.02, as large as
the selection effect. Averaging the last few checkpoints of a run is the standard post-hoc way to
remove that oscillation without retraining or an EMA. Output has the same {"model", "step",
"variant"} layout as checkpoint_final.pt so the eval launcher accepts it unchanged.

    python3 phaseW/average_checkpoints.py --run checkpoints/phaseW/phaseW_B2_118k_s0_128712 \
        --steps 40000,45000,50000,55000,final --out .../checkpoint_avg40k.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", default="40000,45000,50000,55000,final")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    files = [run / ("checkpoint_final.pt" if s == "final" else f"checkpoint_step{s}.pt") for s in args.steps.split(",")]
    for f in files:
        assert f.is_file(), f
    acc, n, meta = None, 0, None
    for f in files:
        # plain sequential read: mmap page-faulting 5 x 10 GB over Lustre stalled for >2 h under
        # contention (jobs 131149/53/57 hit their run limit after one file); a full read is minutes
        ck = torch.load(f, map_location="cpu", mmap=False, weights_only=False)
        sd = ck["model"]
        if acc is None:
            acc = {k: v.detach().to(torch.float32).clone() for k, v in sd.items()}
            meta = {"step": int(ck["step"]), "variant": ck.get("variant")}
        else:
            assert set(sd) == set(acc), "state-dict keys differ"
            for k, v in sd.items():
                acc[k] += v.detach().to(torch.float32)
            # the eval launcher asserts the recorded step is the run's final step, so the average
            # carries the LATEST step among its members, not the first one loaded
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
