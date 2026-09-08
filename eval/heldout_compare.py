#!/usr/bin/env python3
"""Aggregate eval/heldout_dino.py outputs (<dir>/<arm>_s<seed>@<ckpt>.json) into per-arm,
per-checkpoint seed statistics and seed-paired contrasts: DINO similarity of the supervised clean
estimates and of complete 4-step samples, both against the reference photograph.

    python3 eval/heldout_compare.py --dir out/heldout \
        --arm random="random pick" --arm dino_patch=argmax --arm dino_patch-rewX="argmax + exact DINO reward" \
        --contrast dino_patch:random --contrast dino_patch-rewX:dino_patch --contrast dino_patch-rewX:random
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CKPTS = ["step2000", "step4000", "final", "avg_last5"]


def load(d):
    out = defaultdict(dict)   # (arm, ckpt) -> seed -> per-caption means
    for p in sorted(Path(d).glob("*.json")):
        m = re.match(r"(.+)_s(\d+)@(.+)\.json$", p.name)
        if not m:
            continue
        j = json.load(open(p)); caps = sorted(j["captions"], key=lambda c: c["idx"])
        out[(m.group(1), m.group(3))][int(m.group(2))] = {"x0": np.array([np.mean(c["x0_dino"]) for c in caps]),
                                                          "sam": np.array([np.mean(c["sample_dino"]) for c in caps]),
                                                          "teacher": np.array([c["teacher_argmax_dino"] for c in caps])}
    return out


def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/heldout")
    ap.add_argument("--arm", action="append", default=None, help="label=display name, in table order")
    ap.add_argument("--contrast", action="append", default=None, help="labelA:labelB (A minus B)")
    ap.add_argument("--out", default=None, help="markdown output (default <dir>/summary.md)")
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    arms = [a.split("=", 1) for a in (args.arm or ["random=random pick", "dino_patch=argmax", "dino_patch-rewX=argmax + exact DINO reward"])]
    names = {a: b for a, b in arms}
    contrasts = [c.split(":") for c in (args.contrast or ["dino_patch:random", "dino_patch-rewX:dino_patch", "dino_patch-rewX:random"])]
    d = load(args.dir)
    lines = ["# Held-out DINO monitor (validation captions, offline scorer)", ""]
    teacher = next((s["teacher"].mean() for seeds in d.values() for s in seeds.values()), None)
    if teacher is not None:
        lines.append(f"Teacher argmax candidate, same captions: DINO {teacher:.4f}")
    lines += ["", "| arm | checkpoint | seeds | x0-hat DINO (mean +- sd over seeds) | 4-step sample DINO (mean +- sd) |", "|---|---|---|---|---|"]
    tex, ctex = [], []
    for a, name in arms:
        for ck in CKPTS:
            seeds = d.get((a, ck))
            if not seeds:
                continue
            x0 = msd([s["x0"].mean() for s in seeds.values()]); sam = msd([s["sam"].mean() for s in seeds.values()])
            lines.append(f"| {name} | {ck} | {x0[2]} | {x0[0]:.4f} +- {x0[1]:.4f} | {sam[0]:.4f} +- {sam[1]:.4f} |")
            tex.append(f"{name} & {ck.replace('_', ' ')} & {x0[2]} & {x0[0]:.4f} $\\pm$ {x0[1]:.4f} & {sam[0]:.4f} $\\pm$ {sam[1]:.4f} \\\\")
    lines += ["", "## Seed-paired contrasts (difference of per-seed means; t-test over seeds)", "",
              "| contrast | checkpoint | n | x0-hat: diff +- sd, p | 4-step sample: diff +- sd, p |", "|---|---|---|---|---|"]
    for a, b in contrasts:
        for ck in CKPTS:
            sa, sb = d.get((a, ck)), d.get((b, ck))
            if not sa or not sb:
                continue
            common = sorted(set(sa) & set(sb))
            if len(common) < 2:
                continue
            dx = np.array([sa[s]["x0"].mean() - sb[s]["x0"].mean() for s in common]); ds = np.array([sa[s]["sam"].mean() - sb[s]["sam"].mean() for s in common])
            px = stats.ttest_1samp(dx, 0).pvalue; ps = stats.ttest_1samp(ds, 0).pvalue
            lines.append(f"| {names.get(a, a)} - {names.get(b, b)} | {ck} | {len(common)} | {dx.mean():+.4f} +- {dx.std(ddof=1):.4f}, p={px:.3f} | {ds.mean():+.4f} +- {ds.std(ddof=1):.4f}, p={ps:.3f} |")
            ctex.append(f"{names.get(a, a)} $-$ {names.get(b, b)} & {ck.replace('_', ' ')} & {len(common)} & {dx.mean():+.4f} $\\pm$ {dx.std(ddof=1):.4f} ({px:.2f}) & {ds.mean():+.4f} $\\pm$ {ds.std(ddof=1):.4f} ({ps:.2f}) \\\\")
    lines += ["", "## Caption-level paired contrast on the averaged model (seeds x captions pooled)", ""]
    for a, b in contrasts:
        sa, sb = d.get((a, "avg_last5")), d.get((b, "avg_last5"))
        if not sa or not sb:
            continue
        common = sorted(set(sa) & set(sb))
        if not common:
            continue
        dx = np.concatenate([sa[s]["x0"] - sb[s]["x0"] for s in common]); ds = np.concatenate([sa[s]["sam"] - sb[s]["sam"] for s in common])
        lines.append(f"- {names.get(a, a)} - {names.get(b, b)}: x0-hat {dx.mean():+.4f} (paired t p={stats.ttest_1samp(dx, 0).pvalue:.3f}, n={len(dx)}); "
                     f"4-step sample {ds.mean():+.4f} (p={stats.ttest_1samp(ds, 0).pvalue:.3f}, n={len(ds)})")
    txt = "\n".join(lines) + "\n"
    Path(args.out or Path(args.dir) / "summary.md").write_text(txt); print(txt)
    if args.tex:
        print("% --- arm rows"); print("\n".join(tex)); print("% --- contrast rows"); print("\n".join(ctex))


if __name__ == "__main__":
    main()
