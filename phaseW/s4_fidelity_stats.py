#!/usr/bin/env python3
"""Seed-level fidelity statistics for the selection-rule arms from phaseW/fidelity_s4_report.md.

Parses the per-model rows (label@steps | n | FID | split-half | CMMD | CI | precision | recall),
groups by arm and protocol (raw final vs -avglast3), prints seed means and seed-paired CMMD /
precision / recall contrasts against argmax and random, and LaTeX rows for the report.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = [("S4_B2", "random (fixed draw)"), ("S4_CD_dinop_hard", "argmax"),
        ("S4_CD_dinop_full-T0.04", "Boltzmann T=0.04, exact"), ("S4_CD_dinop_cat-T0.04", "Boltzmann T=0.04, sampled"),
        ("S4_CD_dinop_catfreeze-T0.04", "Boltzmann T=0.04, frozen draw"),
        ("S4_CD_latent_hard", "latent scorer, argmax"), ("S4_CD_latent_full-T0.04", "latent scorer, Boltzmann T=0.04, exact")]
NAMES = dict(ARMS)
path = sys.argv[1] if len(sys.argv) > 1 else "phaseW/fidelity_s4_report.md"
rows = defaultdict(dict)   # (arm, suffix) -> seed -> (fid, cmmd, prec, rec)
for ln in open(path):
    m = re.match(r"\|\s*(S4_\S+?)(-avglast3)?_s(\d)@4\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
    if m:
        rows[(m.group(1), m.group(2) or "")][int(m.group(3))] = tuple(float(m.group(k)) for k in (4, 5, 6, 7))


def seedstat(v):
    v = np.asarray(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def contrast(a, b, col):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s][col] - b[s][col] for s in seeds])
    p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else float("nan")
    return d, p


for suffix, title in (("", "raw final checkpoints"), ("-avglast3", "average of the 2k/4k/6k checkpoints")):
    print(f"\n## Fidelity, {title} (5,000 COCO val captions, 4 steps, w=1; per seed s0 / s1 / s2 and seed mean)")
    print("| arm | FID | CMMD | precision | recall |\n|---|---|---|---|---|")
    tex = []
    for key, name in ARMS:
        r = rows.get((key, suffix))
        if not r:
            continue
        seeds = sorted(r)
        cells = []
        for col in range(4):
            v = [r[s][col] for s in seeds]; m, sd = seedstat(v)
            fmt = "{:.2f}" if col < 2 else "{:.3f}"
            cells.append(f"{' / '.join(fmt.format(x) for x in v)} ({fmt.format(m)})")
        print(f"| {name} | " + " | ".join(cells) + " |")
        tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
    print("\n%% LaTeX rows\n" + "\n".join(tex))
    print("\n| contrast | CMMD per seed | mean +- sd (p) | precision mean | recall mean |\n|---|---|---|---|---|")
    for a, b in (("S4_CD_dinop_hard", "S4_B2"), ("S4_CD_dinop_full-T0.04", "S4_CD_dinop_hard"), ("S4_CD_dinop_cat-T0.04", "S4_CD_dinop_hard"),
                 ("S4_CD_dinop_catfreeze-T0.04", "S4_CD_dinop_hard"), ("S4_CD_dinop_full-T0.04", "S4_CD_dinop_cat-T0.04"),
                 ("S4_CD_latent_hard", "S4_CD_dinop_hard"), ("S4_CD_latent_full-T0.04", "S4_CD_dinop_full-T0.04"), ("S4_CD_latent_hard", "S4_B2")):
        ra, rb = rows.get((a, suffix)), rows.get((b, suffix))
        if not ra or not rb:
            continue
        dc, pc = contrast(ra, rb, 1); dp, _ = contrast(ra, rb, 2); dr, _ = contrast(ra, rb, 3)
        print(f"| {NAMES[a]} vs {NAMES[b]} | {' / '.join(f'{x:+.2f}' for x in dc)} | {dc.mean():+.3f} +- {dc.std(ddof=1):.3f} ({pc:.2f}) | {dp.mean():+.3f} | {dr.mean():+.3f} |")
