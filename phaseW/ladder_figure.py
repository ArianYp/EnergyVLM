#!/usr/bin/env python3
"""Figure for the one-factor ablations of the exact-reward recipe (S4j) and the factorial (S4i):
per-arm seed points and means on CompBench, GenEval2 and CMMD (averaged checkpoints), with the
recipe (argmax + exact reward, lambda 15.5, two least-noisy states) and argmax as reference lines.

    python3 phaseW/ladder_figure.py   -> reports/figs/reward_ablations.pdf/.png
"""
from __future__ import annotations

import os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s4_seed_stats import load  # noqa: E402

ARMS = [  # label, display, group
    ("S4_B2", "random pick, no reward", "reference"),
    ("S4_CD_dinop_hard", "argmax, no reward", "reference"),
    ("S4_B2-rewXi", "random pick + exact reward", "factorial"),
    ("S4_CD_dinop_hard-rewXi", "argmax + exact reward (recipe)", "factorial"),
    ("S4_CD_dinop_hard-rewXi-l7.75", "λ = 7.75", "reward weight"),
    ("S4_CD_dinop_hard-rewXi-l31", "λ = 31", "reward weight"),
    ("S4_CD_dinop_hard-rewXi-l62", "λ = 62", "reward weight"),
    ("S4_CD_dinop_hard-rewXi-R1", "1 rewarded state", "rewarded states"),
    ("S4_CD_dinop_hard-rewXi-R5", "5 rewarded states (all)", "rewarded states"),
    ("S4_CD_dinop_hard-rewXi-noisiest", "2 noisiest states", "rewarded states"),
    ("S4_CD_dinop_hard-rewXi-bilinear", "bilinear resize", "implementation"),
    ("S4_CD_dinop_hard-rewXi-bf16", "bf16 DINO", "implementation"),
    ("S4_CD_dinop_hard-rewRi-s16", "projector reward, 16 refresh steps", "projector"),
    ("S4_CD_dinop_hard-rewRi-e25", "projector reward, refresh every 25", "projector"),
]
COL = {"reference": "#333333", "factorial": "#c0392b", "reward weight": "#1f77b4", "rewarded states": "#2ca02c", "implementation": "#7f7f7f", "projector": "#b8860b"}


def fidelity(paths=("phaseW/fidelity_s4_report.md", "phaseW/fidelity_s4i_report.md", "phaseW/fidelity_s4j_report.md")):
    out = defaultdict(dict)
    for p in paths:
        if not os.path.isfile(p):
            continue
        for ln in open(p):
            m = re.match(r"\|\s*(S4_\S+?)-avglast3_s(\d)@4\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|", ln)
            if m:
                out[m.group(1)][int(m.group(2))] = float(m.group(4))     # CMMD
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load("-avglast3"); fid = fidelity()
    arms = [(k, n, g) for k, n, g in ARMS if k + "-avglast3" in rows]
    panels = [("CompBench", lambda k: [rows[k + "-avglast3"][s][0] for s in sorted(rows[k + "-avglast3"])]),
              ("GenEval2 (x100)", lambda k: [rows[k + "-avglast3"][s][1] for s in sorted(rows[k + "-avglast3"])]),
              ("CMMD (lower is better)", lambda k: [fid[k][s] for s in sorted(fid.get(k, {}))])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(arms) + 1.6), sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, (title, get) in zip(axes, panels):
        ref = {}
        for yi, (k, n, g) in zip(y, arms):
            v = np.array(get(k), float)
            if len(v) == 0:
                continue
            ax.scatter(v, np.full(len(v), yi), s=18, color=COL[g], alpha=0.45, zorder=2)
            ax.plot([v.mean()], [yi], marker="D", ms=7, color=COL[g], zorder=3)
            if len(v) > 1:
                ax.plot([v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1)], [yi, yi], color=COL[g], lw=1.2, zorder=1)
            if k in ("S4_CD_dinop_hard", "S4_CD_dinop_hard-rewXi"):
                ref[k] = v.mean()
        for k, ls in (("S4_CD_dinop_hard", ":"), ("S4_CD_dinop_hard-rewXi", "--")):
            if k in ref:
                ax.axvline(ref[k], color="#555555", ls=ls, lw=1)
        ax.set_title(title, fontsize=11); ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y); axes[0].set_yticklabels([n for _, n, _ in arms], fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="D", ls="", label=g) for g, c in COL.items()] + [Line2D([], [], color="#555555", ls=":", label="argmax mean"), Line2D([], [], color="#555555", ls="--", label="recipe mean")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Exact-reward recipe: factorial cells and one-factor ablations, averaged checkpoints (seed points, mean ± sd)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out = Path("reports/figs/reward_ablations"); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out.with_suffix(".pdf"), "arms:", len(arms), "with CMMD:", sum(1 for k, _, _ in arms if fid.get(k)))


if __name__ == "__main__":
    main()
