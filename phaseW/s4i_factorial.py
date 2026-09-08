#!/usr/bin/env python3
"""The selection x reward factorial (S4i). Cells, with M_sr = CompBench (and GenEval2) of the arm with
selection s (0 random pick, 1 argmax) and reward r (0 none, 1 exact DINO reward):

    M_00  S4_B2                     M_01  S4_B2-rewXi
    M_10  S4_CD_dinop_hard          M_11  S4_CD_dinop_hard-rewXi

Reports every cell (seed mean +- sd, n seeds), the four simple effects as seed-paired contrasts, the
selection benefit given the reward (M_11 - M_01), the interaction I = (M_11 - M_10) - (M_01 - M_00)
with its seed-level interval (per-seed interaction over the seeds all four cells share), and a
two-way ANOVA-style unpaired check. Raw final checkpoints and averaged checkpoints.

    python3 phaseW/s4i_factorial.py [--tex]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s4_seed_stats import contrast, load  # noqa: E402

CELLS = {"00": "S4_B2", "01": "S4_B2-rewXi", "10": "S4_CD_dinop_hard", "11": "S4_CD_dinop_hard-rewXi"}
NAMES = {"00": "random pick, no reward", "01": "random pick + exact reward", "10": "argmax, no reward", "11": "argmax + exact reward"}
COLS = {0: "CompBench", 1: "GenEval2"}


def cell_stats(rows, key, col):
    v = np.array([rows[key][s][col] for s in sorted(rows[key])])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def fmtc(c, col):
    if c is None:
        return "n/a"
    f = "{:+.4f}" if col == 0 else "{:+.2f}"
    return f"{f.format(c['mean'])} +- {f.format(c['sd']).lstrip('+')} [{f.format(c['ci95'][0])}, {f.format(c['ci95'][1])}] (p={c['t_p']:.3f}, n={len(c['seeds'])})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", action="store_true"); ap.add_argument("--out", default="phaseW/s4i_factorial.md")
    args = ap.parse_args()
    lines = ["# Selection x reward factorial (S4i)", ""]
    tex = []
    for suffix, title in (("", "raw final checkpoints"), ("-avglast3", "averaged checkpoints (2k/4k/6k)")):
        rows = load(suffix)
        keys = {k: v + suffix for k, v in CELLS.items()}
        if not all(k in rows for k in keys.values()):
            lines.append(f"## {title}: missing cells {[NAMES[c] for c, k in keys.items() if k not in rows]}"); continue
        lines += [f"## {title}", ""]
        for col in (0, 1):
            lines += [f"### {COLS[col]}", "", "| cell | seeds | mean +- sd | per seed |", "|---|---|---|---|"]
            f = "{:.4f}" if col == 0 else "{:.2f}"
            for c, k in keys.items():
                m, sd, n = cell_stats(rows, k, col)
                per = " / ".join(f.format(rows[k][s][col]) for s in sorted(rows[k]))
                lines.append(f"| M_{c} {NAMES[c]} | {n} | {f.format(m)} +- {f.format(sd)} | {per} |")
                tex.append(f"{NAMES[c]} & {suffix or 'raw'} & {COLS[col]} & {n} & {f.format(m)} $\\pm$ {f.format(sd)} \\\\")
            lines += ["", "| contrast | seed-paired mean +- sd [95% CI] (p, n) |", "|---|---|"]
            eff = {"reward | random (M_01 - M_00)": (keys["01"], keys["00"]), "reward | argmax (M_11 - M_10)": (keys["11"], keys["10"]),
                   "selection | no reward (M_10 - M_00)": (keys["10"], keys["00"]), "selection | reward (M_11 - M_01)": (keys["11"], keys["01"])}
            for name, (a, b) in eff.items():
                c = contrast(rows, a, b, col, name); lines.append(f"| {name} | {fmtc(c, col)} |")
            # interaction over the seeds all four cells share
            common = sorted(set.intersection(*[set(rows[k]) for k in keys.values()]))
            I = np.array([(rows[keys['11']][s][col] - rows[keys['10']][s][col]) - (rows[keys['01']][s][col] - rows[keys['00']][s][col]) for s in common])
            if len(I) > 1:
                t = stats.ttest_1samp(I, 0); ci = stats.t.interval(0.95, len(I) - 1, loc=I.mean(), scale=I.std(ddof=1) / np.sqrt(len(I)))
                lines.append(f"| interaction I = (M_11-M_10)-(M_01-M_00) | {f.format(I.mean())} +- {f.format(I.std(ddof=1))} [{f.format(ci[0])}, {f.format(ci[1])}] (p={t.pvalue:.3f}, n={len(I)}) |")
            # unpaired two-way check: additive model fit on all runs
            y, S, R = [], [], []
            for c, k in keys.items():
                for s in rows[k]:
                    y.append(rows[k][s][col]); S.append(int(c[0])); R.append(int(c[1]))
            y, S, R = map(np.array, (y, S, R)); X = np.c_[np.ones_like(y), S, R, S * R]
            beta, res, *_ = np.linalg.lstsq(X, y, rcond=None); dof = len(y) - 4
            sigma2 = float(((y - X @ beta) ** 2).sum() / dof); cov = sigma2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
            pI = 2 * stats.t.sf(abs(beta[3] / se[3]), dof)
            lines.append(f"| unpaired OLS: selection {f.format(beta[1])} (se {f.format(se[1])}), reward {f.format(beta[2])} (se {f.format(se[2])}), interaction {f.format(beta[3])} (se {f.format(se[3])}, p={pI:.3f}), n runs {len(y)} | |")
            lines.append("")
    txt = "\n".join(lines) + "\n"; Path(args.out).write_text(txt); print(txt)
    if args.tex:
        print("% --- cell rows"); print("\n".join(tex))


if __name__ == "__main__":
    main()
