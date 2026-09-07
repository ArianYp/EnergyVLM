#!/usr/bin/env python3
"""Seed-level statistics for the selection-rule arms (report Section 10, reviewer step 0).

The prompt-level paired tests in phaseW/verifier_swap_result.py average the seeds' per-prompt
scores and test across prompts; that is uncertainty over prompts for the mean model, not over
repeated training. This script reports, per arm and contrast:
  * per-seed scores and per-seed paired differences (seed s of arm A minus seed s of arm B; the
    arms share data order and Delta draws at equal seed),
  * mean, standard deviation, a 95% t-interval over seeds (n = 3, so wide) and the paired t p-value,
  * GPU-hours per arm from the LSF job logs (training + evaluation),
  * Kish ESS and per-visit switching probability of the Boltzmann weights on the 3k cache.

    python3 phaseW/s4_seed_stats.py --out phaseW/s4_seed_stats.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = [  # label prefix, display name, jobs file
    ("S4_B2", "random (fixed draw)", "phaseW/s4_jobs_done.txt"),
    ("S4_CD_dinop_hard", "argmax", "phaseW/s4_jobs_done.txt"),
    ("S4_CD_dinop_full-T0.04", "Boltzmann T=0.04, exact", "phaseW/s4b_jobs.txt"),
    ("S4_CD_dinop_full-T0.08", "Boltzmann T=0.08, exact", "phaseW/s4cd_jobs.txt"),
    ("S4_CD_dinop_full-T1e6", "uniform, exact", "phaseW/s4cd_jobs.txt"),
    ("S4_CD_dinop_cat-T0.04", "Boltzmann T=0.04, sampled", "phaseW/s4_jobs_done.txt"),
    ("S4_CD_dinop_cat-T0.08", "Boltzmann T=0.08, sampled", "phaseW/s4_jobs_done.txt"),
    ("S4_CD_uniform_visit", "uniform-visit, sampled", "phaseW/s4_jobs_done.txt"),
    ("S4_CD_dinop_catfreeze-T0.04", "Boltzmann T=0.04, frozen draw", "phaseW/s4e_jobs.txt"),
    ("S4_CD_latent_hard", "latent scorer, argmax", "phaseW/s4f_jobs.txt"),
    ("S4_CD_latent_full-T0.04", "latent scorer, Boltzmann T=0.04, exact", "phaseW/s4f_jobs.txt"),
    ("S4_CD_dinop_hard-rewF", "argmax + projector reward (frozen)", "phaseW/s4g_jobs.txt"),
    ("S4_CD_dinop_hard-rewR", "argmax + projector reward (refreshed)", "phaseW/s4g_jobs.txt"),
]
CONTRASTS = [
    ("S4_CD_dinop_hard", "S4_B2"), ("S4_CD_dinop_full-T0.04", "S4_B2"), ("S4_CD_dinop_full-T0.08", "S4_B2"),
    ("S4_CD_dinop_full-T1e6", "S4_B2"), ("S4_CD_dinop_cat-T0.04", "S4_B2"), ("S4_CD_dinop_cat-T0.08", "S4_B2"),
    ("S4_CD_uniform_visit", "S4_B2"), ("S4_CD_dinop_catfreeze-T0.04", "S4_B2"),
    ("S4_CD_dinop_full-T0.04", "S4_CD_dinop_hard"), ("S4_CD_dinop_full-T0.08", "S4_CD_dinop_hard"),
    ("S4_CD_dinop_full-T1e6", "S4_CD_dinop_hard"), ("S4_CD_dinop_cat-T0.04", "S4_CD_dinop_hard"),
    ("S4_CD_dinop_catfreeze-T0.04", "S4_CD_dinop_hard"),
    ("S4_CD_dinop_full-T0.04", "S4_CD_dinop_cat-T0.04"), ("S4_CD_dinop_catfreeze-T0.04", "S4_CD_dinop_cat-T0.04"),
    ("S4_CD_dinop_full-T1e6", "S4_CD_uniform_visit"),
    ("S4_CD_latent_hard", "S4_B2"), ("S4_CD_latent_full-T0.04", "S4_B2"),
    ("S4_CD_latent_hard", "S4_CD_dinop_hard"), ("S4_CD_latent_full-T0.04", "S4_CD_dinop_hard"),
    ("S4_CD_latent_full-T0.04", "S4_CD_dinop_full-T0.04"), ("S4_CD_latent_full-T0.04", "S4_CD_latent_hard"),
    ("S4_CD_dinop_hard-rewF", "S4_CD_dinop_hard"), ("S4_CD_dinop_hard-rewR", "S4_CD_dinop_hard"),
    ("S4_CD_dinop_hard-rewF", "S4_B2"), ("S4_CD_dinop_hard-rewR", "S4_B2"), ("S4_CD_dinop_hard-rewR", "S4_CD_dinop_hard-rewF"),
]


def load(suffix=""):
    rows = defaultdict(dict)
    for d in sorted(glob.glob("phaseN/eval_S4_*_s*_*")):
        f = os.path.join(d, "alignment.json")
        if not os.path.isfile(f):
            continue
        a = json.load(open(f))
        m = re.match(r"(.*)_s(\d)$", a["label"])
        if not m:
            continue
        rows[m.group(1)][int(m.group(2))] = (a["compbench_mean"], a["geneval2"] * 100)
    return rows


def runtime_seconds(job):
    for f in glob.glob(f"logs/*-{job}.out"):
        m = re.search(r"Run time :\s+(\d+)", open(f, errors="ignore").read())
        if m:
            return int(m.group(1))
    return None


def gpu_hours(jobs_file, prefix):
    tr, ev = 0.0, 0.0
    if not os.path.isfile(jobs_file):
        return None, None
    for ln in open(jobs_file):
        p = ln.split()
        if len(p) < 4 or not prefix.endswith(p[0]):
            continue
        t, e = runtime_seconds(p[2]), runtime_seconds(p[3])
        tr += (t or 0) / 3600; ev += (e or 0) / 3600
    return tr, ev


def fmt_seed(v):
    return " / ".join(f"{x:.4f}" for x in v)


def contrast(rows, a, b, col, name):
    seeds = sorted(set(rows[a]) & set(rows[b]))
    if len(seeds) < 2:
        return None
    d = np.array([rows[a][s][col] - rows[b][s][col] for s in seeds])
    t = stats.ttest_1samp(d, 0.0)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else (np.nan, np.nan)
    return {"seeds": seeds, "per_seed": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "ci95": [float(ci[0]), float(ci[1])], "t_p": float(t.pvalue), "name": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phaseW/s4_seed_stats.md")
    ap.add_argument("--suffix", default="", help="e.g. -avglast3 to report the averaged checkpoints")
    ap.add_argument("--tex", action="store_true", help="also print LaTeX table rows")
    args = ap.parse_args()
    rows = load()
    names = {k: n for k, n, _ in ARMS}
    lines = ["# Selection-rule arms: seed-level statistics (3k pool, S4 configuration)", ""]
    for suffix, title in (("", "raw final checkpoints"), ("-avglast3", "average of the checkpoints at 2k, 4k and 6k updates")):
        lines += [f"## Per-seed scores, {title}", "", "| arm | CompBench s0 / s1 / s2 | mean +- sd | GenEval2 s0 / s1 / s2 | mean +- sd | GPU-h train | GPU-h eval |", "|---|---|---|---|---|---|---|"]
        for key, name, jobs in ARMS:
            k = key + suffix
            if k not in rows:
                continue
            seeds = sorted(rows[k]); c = np.array([rows[k][s][0] for s in seeds]); g = np.array([rows[k][s][1] for s in seeds])
            tr, ev = gpu_hours(jobs, key) if not suffix else (None, None)
            lines.append(f"| {name} | {fmt_seed(c)} | {c.mean():.4f} +- {c.std(ddof=1):.4f} | {' / '.join(f'{x:.2f}' for x in g)} | {g.mean():.2f} +- {g.std(ddof=1):.2f} | "
                         f"{'' if tr is None else f'{tr:.1f}'} | {'' if ev is None else f'{ev:.1f}'} |")
        lines.append("")
        lines += [f"## Seed-paired contrasts, {title}", "", "| contrast | CompBench per seed | mean +- sd | 95% t-CI | p (seed t) | GenEval2 per seed | mean +- sd | p |", "|---|---|---|---|---|---|---|---|"]
        out_json = {}
        for a, b in CONTRASTS:
            ka, kb = a + suffix, b + suffix
            if ka not in rows or kb not in rows:
                continue
            cc = contrast(rows, ka, kb, 0, f"{names[a]} vs {names[b]}"); cg = contrast(rows, ka, kb, 1, "")
            if cc is None:
                continue
            out_json[f"{ka} vs {kb}"] = {"compbench": cc, "geneval2": cg}
            lines.append(f"| {names[a]} vs {names[b]} | {' / '.join(f'{x:+.4f}' for x in cc['per_seed'])} | {cc['mean']:+.4f} +- {cc['sd']:.4f} | [{cc['ci95'][0]:+.4f}, {cc['ci95'][1]:+.4f}] | {cc['t_p']:.3f} | "
                         f"{' / '.join(f'{x:+.2f}' for x in cg['per_seed'])} | {cg['mean']:+.2f} +- {cg['sd']:.2f} | {cg['t_p']:.3f} |")
        lines.append("")
        json.dump(out_json, open(args.out.replace(".md", f"{suffix or '-raw'}.json"), "w"), indent=1)
    # weights statistics on the 3k cache
    S = []
    for f in sorted(glob.glob("phaseN/coco_selection_dinopatch/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                S.append(json.loads(ln)["dino_patch_cos"])
    S = np.asarray(S, float)
    lines += ["## Boltzmann weights on the 3k cache (N = 4)", "", "| T | mean entropy ESS | median entropy ESS | mean Kish ESS | median Kish ESS | P(switch between visits) | E[distinct candidates in 2 visits] | mean max weight |", "|---|---|---|---|---|---|---|---|"]
    for T in (0.02, 0.04, 0.08, 0.16, 1e6):
        z = (S - S.max(1, keepdims=True)) / T; W = np.exp(z); W /= W.sum(1, keepdims=True)
        H = -(W * np.log(np.clip(W, 1e-300, None))).sum(1)
        kish = 1.0 / (W ** 2).sum(1); psw = 1.0 - (W ** 2).sum(1); u2 = (1 - (1 - W) ** 2).sum(1)
        lines.append(f"| {T:g} | {np.exp(H).mean():.2f} | {np.median(np.exp(H)):.2f} | {kish.mean():.2f} | {np.median(kish):.2f} | {psw.mean():.3f} | {u2.mean():.2f} | {W.max(1).mean():.3f} |")
    lines.append("")
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))
    if args.tex:
        tex = ["%% --- per-arm rows: name & raw s0/s1/s2 & raw mean+-sd & avg s0/s1/s2 & avg mean+-sd & GenEval2 raw mean & GenEval2 avg mean & GPU-h/run"]
        for key, name, jobs in ARMS:
            if key not in rows:
                continue
            r = rows[key]; a = rows.get(key + "-avglast3", {})
            sr = sorted(r); sa = sorted(a)
            cr = np.array([r[s][0] for s in sr]); gr = np.array([r[s][1] for s in sr])
            tr, _ = gpu_hours(jobs, key)
            row = f"{name} & {' / '.join(f'{x:.4f}' for x in cr)} & {cr.mean():.4f} $\\pm$ {cr.std(ddof=1):.4f}"
            if sa:
                ca = np.array([a[s][0] for s in sa]); ga = np.array([a[s][1] for s in sa])
                row += f" & {' / '.join(f'{x:.4f}' for x in ca)} & {ca.mean():.4f} $\\pm$ {ca.std(ddof=1):.4f} & {gr.mean():.2f} & {ga.mean():.2f}"
            else:
                row += f" & -- & -- & {gr.mean():.2f} & --"
            row += f" & {'' if tr is None else f'{tr/len(sr):.1f}'} \\\\"
            tex.append(row)
        tex.append("%% --- contrast rows: name & raw per-seed & raw mean+-sd [CI] p & avg per-seed & avg mean+-sd [CI] p")
        for a_, b_ in CONTRASTS:
            if a_ not in rows or b_ not in rows:
                continue
            cr = contrast(rows, a_, b_, 0, ""); ca = contrast(rows, a_ + "-avglast3", b_ + "-avglast3", 0, "") if (a_ + "-avglast3" in rows and b_ + "-avglast3" in rows) else None
            def cell(c):
                if c is None:
                    return "-- & --"
                return (f"{' / '.join(f'{x:+.4f}' for x in c['per_seed'])} & ${c['mean']:+.4f} \\pm {c['sd']:.4f}$ "
                        f"[${c['ci95'][0]:+.3f}$, ${c['ci95'][1]:+.3f}$] ($p={c['t_p']:.2f}$)")
            tex.append(f"{names[a_]} vs {names[b_]} & {cell(cr)} & {cell(ca)} \\\\")
        print("\n".join(tex))


if __name__ == "__main__":
    main()
