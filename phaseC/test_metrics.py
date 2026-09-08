#!/usr/bin/env python3
"""
CPU checks of the fidelity/diversity maths, so a bug is caught before the GPU jobs
spend hours generating images for it.

Covers the parts that are easy to get silently wrong:
  * the cached-kernel CMMD bootstrap must equal the direct CMMD under an identity
    resample (it exists only because recomputing a 5k x 5k cdist per replicate is
    infeasible, so it must be provably the same quantity)
  * FID must be ~0 for a set against itself
  * improved precision/recall must separate MODE COLLAPSE (on-manifold, low coverage)
    from a healthy set — this is the gate's actual job
  * the diversity bootstrap must bracket its point estimate
  * square_crop must square portrait, landscape and square inputs

    python phaseC/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp0"))
from diversity_eval import boot_ci  # noqa: E402
from fidelity_cmmd import cmmd  # noqa: E402
from fidelity_eval import (cmmd_boot_ci, fid_from_feats,  # noqa: E402
                           precision_recall, square_crop)


class IdentityRng:
    """rng stub whose 'resample' is the original order, so the bootstrap must
    reproduce the point estimate exactly."""

    def integers(self, lo, hi, n):
        return np.arange(n)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    fails = []

    # 1. cached-kernel CMMD bootstrap == direct CMMD under identity resample
    g, r = torch.randn(200, 64), torch.randn(200, 64) + 0.3
    lo, hi = cmmd_boot_ci(g, r, IdentityRng(), reps=3)
    direct = cmmd(g, r)
    err = max(abs(lo - direct), abs(hi - direct))
    print(f"1. cached-kernel bootstrap vs direct CMMD: err={err:.2e}")
    if err > 1e-4:
        fails.append("CMMD bootstrap does not reproduce the direct CMMD")

    # 2. a real bootstrap brackets the point estimate
    lo, hi = cmmd_boot_ci(g, r, np.random.default_rng(0), reps=300)
    print(f"2. CMMD bootstrap CI [{lo:.3f}, {hi:.3f}] vs point {direct:.3f}")
    if not lo <= direct <= hi:
        fails.append("CMMD bootstrap CI does not bracket the point estimate")

    # 3. FID against itself is ~0, and grows with a shift
    f = rng.normal(size=(500, 128))
    self_fid, shift_fid = fid_from_feats(f, f), fid_from_feats(f, f + 1.0)
    print(f"3. FID(x,x)={self_fid:.2e}  FID(x,x+1)={shift_fid:.2f}")
    if abs(self_fid) > 1e-6 or shift_fid < 1.0:
        fails.append("FID is not ~0 against itself, or does not grow with a shift")

    # 4. precision/recall separates mode collapse from a healthy set.
    #    NOTE the cases are built to survive L2 normalisation: scaling features (x*0.15)
    #    or shifting them far (x+50) are no-ops / degenerate once normalised, so they
    #    would test nothing.
    real = torch.randn(400, 64)
    collapsed = real[:8].repeat_interleave(50, 0) + 0.01 * torch.randn(400, 64)
    p_id, r_id = precision_recall(real, real.clone())
    p_co, r_co = precision_recall(real, collapsed)
    print(f"4. identical: P={p_id:.3f} R={r_id:.3f} | collapsed to 8 modes: "
          f"P={p_co:.3f} R={r_co:.3f}")
    if not (r_id > 0.8 and r_co < 0.2 and p_co > 0.8):
        fails.append("precision/recall does not flag mode collapse as "
                     "high-precision / low-recall")

    # 5. diversity bootstrap brackets its mean
    vals = rng.normal(0.3, 0.05, 400)
    m, blo, bhi = boot_ci(vals, np.random.default_rng(0), reps=2000)
    print(f"5. diversity boot_ci mean={m:.4f} CI [{blo:.4f}, {bhi:.4f}]")
    if not blo <= m <= bhi:
        fails.append("diversity bootstrap CI does not bracket its mean")

    # 6. square_crop
    for shape in [(480, 640, 3), (640, 480, 3), (512, 512, 3)]:
        out = square_crop(np.zeros(shape, dtype=np.uint8))
        if not out.shape[0] == out.shape[1] == min(shape[:2]):
            fails.append(f"square_crop({shape}) -> {out.shape}")
    print("6. square_crop squares portrait / landscape / square inputs")

    if fails:
        print("\nFAIL:")
        for f_ in fails:
            print(f"  - {f_}")
        sys.exit(1)
    print("\nPASS — fidelity/diversity maths behave as intended")


if __name__ == "__main__":
    main()
