#!/usr/bin/env python3
"""Aggregate the held-out DINO monitor (phaseW/latent_scorer/heldout/<arm>_s<seed>@<ckpt>.json) into
per-arm, per-checkpoint seed statistics and seed-paired contrasts (reviewer requirement 5 of the
exact-reward study). Two quantities per checkpoint: DINO similarity of the supervised clean estimates
(x0-hat at states 5 and 6 of the argmax teacher trajectory) and of complete 4-step samples, both
against the caption's reference photo, offline PIL scorer, 16 validation captions x 2 (x0) or 2 seeds.

    python3 phaseW/latent_scorer/heldout_compare.py [--tex]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ARMS = [("B2", "random pick"), ("CD_dinop_hard", "argmax"), ("CD_dinop_hard-rewX", "argmax + exact DINO reward")]
CKPTS = ["step2000", "step4000", "final", "avg_last5"]
CONTRASTS = [("CD_dinop_hard", "B2"), ("CD_dinop_hard-rewX", "CD_dinop_hard"), ("CD_dinop_hard-rewX", "B2")]


def load():
    d = defaultdict(dict)   # (arm, ckpt) -> seed -> {x0: per-caption mean, sam: per-caption mean, teacher}
    for p in sorted((HERE / "heldout").glob("*.json")):
        m = re.match(r"(.+)_s(\d+)@(.+)\.json$", p.name)
        if not m:
            continue
        arm, seed, ck = m.group(1), int(m.group(2)), m.group(3)
        j = json.load(open(p))
        caps = sorted(j["captions"], key=lambda c: c["idx"])
        d[(arm, ck)][seed] = {"x0": np.array([np.mean(c["x0_dino"]) for c in caps]),
                              "sam": np.array([np.mean(c["sample_dino"]) for c in caps]),
                              "teacher": np.array([c["teacher_argmax_dino"] for c in caps])}
    return d


def msd(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", action="store_true"); ap.add_argument("--out", default=str(HERE / "heldout_summary.md"))
    args = ap.parse_args()
    d = load()
    lines = ["# Held-out DINO monitor (16 validation captions, offline PIL scorer)", ""]
    teacher = None
    for (arm, ck), seeds in d.items():
        teacher = next(iter(seeds.values()))["teacher"].mean(); break
    lines.append(f"Teacher argmax candidate (8 steps, w=7), same captions: DINO {teacher:.4f}" if teacher is not None else "")
    lines.append(""); lines.append("| arm | checkpoint | seeds | x0-hat DINO (mean +- sd over seeds) | 4-step sample DINO (mean +- sd) |"); lines.append("|---|---|---|---|---|")
    tex = []
    for arm, name in ARMS:
        for ck in CKPTS:
            seeds = d.get((arm, ck))
            if not seeds:
                continue
            x0 = msd([s["x0"].mean() for s in seeds.values()]); sam = msd([s["sam"].mean() for s in seeds.values()])
            lines.append(f"| {name} | {ck} | {x0[2]} | {x0[0]:.4f} +- {x0[1]:.4f} | {sam[0]:.4f} +- {sam[1]:.4f} |")
            tex.append(f"{name} & {ck.replace('_', ' ')} & {x0[2]} & {x0[0]:.4f} $\\pm$ {x0[1]:.4f} & {sam[0]:.4f} $\\pm$ {sam[1]:.4f} \\\\")
    lines.append(""); lines.append("## Seed-paired contrasts (difference of per-seed means; t-test over seeds, n = common seeds)"); lines.append("")
    lines.append("| contrast | checkpoint | n | x0-hat: diff +- sd, p | 4-step sample: diff +- sd, p |"); lines.append("|---|---|---|---|---|")
    ctex = []
    names = dict(ARMS)
    for a, b in CONTRASTS:
        for ck in CKPTS:
            sa, sb = d.get((a, ck)), d.get((b, ck))
            if not sa or not sb:
                continue
            common = sorted(set(sa) & set(sb))
            if len(common) < 2:
                continue
            dx = np.array([sa[s]["x0"].mean() - sb[s]["x0"].mean() for s in common]); ds = np.array([sa[s]["sam"].mean() - sb[s]["sam"].mean() for s in common])
            px = stats.ttest_1samp(dx, 0).pvalue; ps = stats.ttest_1samp(ds, 0).pvalue
            lines.append(f"| {names[a]} - {names[b]} | {ck} | {len(common)} | {dx.mean():+.4f} +- {dx.std(ddof=1):.4f}, p={px:.3f} | {ds.mean():+.4f} +- {ds.std(ddof=1):.4f}, p={ps:.3f} |")
            ctex.append(f"{names[a]} $-$ {names[b]} & {ck.replace('_', ' ')} & {len(common)} & {dx.mean():+.4f} $\\pm$ {dx.std(ddof=1):.4f} ({px:.2f}) & {ds.mean():+.4f} $\\pm$ {ds.std(ddof=1):.4f} ({ps:.2f}) \\\\")
    # per-caption paired view for the averaged model (pool seeds x captions)
    lines.append(""); lines.append("## Caption-level paired contrast on the averaged model (seeds x captions pooled)"); lines.append("")
    for a, b in CONTRASTS:
        sa, sb = d.get((a, "avg_last5")), d.get((b, "avg_last5"))
        if not sa or not sb:
            continue
        common = sorted(set(sa) & set(sb))
        if not common:
            continue
        dx = np.concatenate([sa[s]["x0"] - sb[s]["x0"] for s in common]); ds = np.concatenate([sa[s]["sam"] - sb[s]["sam"] for s in common])
        lines.append(f"- {names[a]} - {names[b]}: x0-hat {dx.mean():+.4f} (paired t p={stats.ttest_1samp(dx, 0).pvalue:.3f}, n={len(dx)}); "
                     f"4-step sample {ds.mean():+.4f} (p={stats.ttest_1samp(ds, 0).pvalue:.3f}, n={len(ds)})")
    txt = "\n".join(lines) + "\n"
    Path(args.out).write_text(txt); print(txt)
    if args.tex:
        print("% --- arm rows"); print("\n".join(tex)); print("% --- contrast rows"); print("\n".join(ctex))


if __name__ == "__main__":
    main()
