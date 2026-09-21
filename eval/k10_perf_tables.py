#!/usr/bin/env python3
"""Performance tables (docs/k10/): naive random-selection CD (old constant-LR schedule, raw final =
"unconverged", and its averaged checkpoint), naive CD on the converged schedule (K=8), and the best
branch (scored selection + refreshed projector reward, K=10 teacher grid, converged schedule,
averaged; + grid A under --eval10). Seeds are averaged where several exist. Absolute numbers and
deltas vs the best branch, per CompBench category, per GenEval2 skill and per prompt complexity.
Reads the experimental tree's phaseN/ records (common/artifacts.py).

    python eval/k10_perf_tables.py [--eval10] [--artifacts <tree>]   -> docs/k10/PERF_TABLES.md (PERF_TABLES10.md)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]


def models(eval10: bool):
    """(name, glob of eval dirs -- seeds averaged), the last entry being the BEST branch."""
    m = [
        ("naive CD, old schedule, raw final (unconverged)", "phaseN/eval_S4_B2_s[0-4]_*"),
        ("naive CD, old schedule, avg last 3", "phaseN/eval10_S4_B2-avglast3_s[0-4]_*" if eval10 else "phaseN/eval_S4_B2-avglast3_s[0-4]_*"),
        ("naive CD, converged schedule, K=8, avg", "phaseN/eval10_W_B2-hp1_3k-hp1-avg_s[0-2]_*" if eval10 else "phaseN/eval_W_B2-hp1_3k-hp1-avg_s0_145251"),
        ("scored + projector reward, K=8, avg (paper)", "phaseN/eval10_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg_s[0-2]_*" if eval10 else "phaseN/eval_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg_s0_145253"),
        ("scored + projector reward, K=10, avg", "phaseN/eval10_W_CD_dinop_hard-k10-hp1_3k-hp1-avg_s[0-2]_*" if eval10 else "phaseN/eval_W_CD_dinop_hard-k10-hp1_3k-hp1-avg_s0_153475"),
    ]
    if eval10:
        m.append(("BEST: scored + projector reward, K=10 + grid A, avg", "phaseN/eval10_W_CD_dinop_hard-k10-hp1_3k-hp1-avg-gridA_s[0-2]_*"))
    return m


def load_dir(d):
    al = json.load(open(f"{d}/alignment.json"))
    # categories from the per-category scores.json (older alignment.json files key them differently)
    cats = {}
    for c in CATS:
        # exact '<label>_s<steps>_<category>' match: '*_spatial' also catches '_3d_spatial' and '_non_spatial'
        fs = [f for f in glob.glob(f"{d}/compbench_scores/*_{c}/scores.json")
              if re.search(rf"_s\d+_{re.escape(c)}$", os.path.basename(os.path.dirname(f)))]
        assert len(fs) == 1, (d, c, fs)
        cats[c] = float(json.load(open(fs[0]))["mean"])
    al["compbench"] = cats; al["compbench_mean"] = float(np.mean(list(cats.values())))
    g = json.load(open(glob.glob(f"{d}/geneval2_scores/*/scores.json")[0]))
    skills = {}
    ps = g.get("per_skill") or {}
    for k, v in ps.items():
        skills[k] = float(v["mean"] if isinstance(v, dict) and "mean" in v else v)
    # complexity = number of atoms in the prompt
    comp = collections.defaultdict(list)
    for r in g.get("per_prompt", []):
        n = len(r.get("atom_scores") or r.get("skills") or [])
        comp[BINS(n)].append(float(r["score"]))
    n_comp = {k: len(v) for k, v in comp.items()}
    comp = {k: float(np.mean(v)) for k, v in comp.items()}
    return {"cb": al["compbench_mean"], "cats": {c: al["compbench"][c] for c in CATS}, "ge2": al["geneval2"], "skills": skills, "comp": comp, "n_comp": n_comp}


def BINS(n):
    """GenEval2 prompt complexity = number of scored atoms (3-11 in this pool)."""
    return "3-4 atoms" if n <= 4 else "5-6 atoms" if n <= 6 else "7-8 atoms" if n <= 8 else "9+ atoms"


def avg(dicts):
    out = {"cb": float(np.mean([d["cb"] for d in dicts])), "ge2": float(np.mean([d["ge2"] for d in dicts]))}
    out["cats"] = {c: float(np.mean([d["cats"][c] for d in dicts])) for c in CATS}
    sk = sorted({k for d in dicts for k in d["skills"]}); out["skills"] = {k: float(np.mean([d["skills"][k] for d in dicts if k in d["skills"]])) for k in sk}
    cp = sorted({k for d in dicts for k in d["comp"]}); out["comp"] = {k: float(np.mean([d["comp"][k] for d in dicts if k in d["comp"]])) for k in cp}
    out["n_comp"] = dicts[0].get("n_comp", {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--eval10", action="store_true", help="the official 10-images-per-prompt protocol (eval10_* dirs), 3 seeds per arm")
    ap.add_argument("--out", default=None, help="default docs/k10/PERF_TABLES.md (PERF_TABLES10.md with --eval10)")
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    outfile = args.out or str(ROOT / "docs" / "k10" / ("PERF_TABLES10.md" if args.eval10 else "PERF_TABLES.md"))
    res = []
    for name, pat in models(args.eval10):
        ds = sorted(glob.glob(pat)); ds = [d for d in ds if os.path.exists(f"{d}/alignment.json")]
        assert ds, (name, pat)
        res.append((name, len(ds), avg([load_dir(d) for d in ds])))
    best = res[-1][2]
    L = ["# Performance tables (2026-09-17)", "",
         "All 4-step students of SD3.5-Medium, 3k COCO captions, sampled at guidance 1 on the scheduler grid; CompBench = mean of 8 categories "
         "(300 held-out prompts each, one image per prompt); GenEval2 = Soft-TIFA geometric mean (800 prompts). Deltas are BEST minus the row. "
         "Old schedule = constant LR 2.8e-5, 6,000 updates; converged = cosine LR 1e-5, batch 16, 3,000 updates, window K={4..7} (K=8) or {6..9} (K=10).", "",
         "## Summary", "", "| model | seeds | CompBench | d vs BEST | GenEval2 | d vs BEST |", "|---|---|---|---|---|---|"]
    for name, n, r in res:
        L.append(f"| {name} | {n} | {r['cb']:.4f} | {best['cb'] - r['cb']:+.4f} | {r['ge2']:.4f} | {best['ge2'] - r['ge2']:+.4f} |")
    L += ["", "## T2I-CompBench per category (absolute)", "", "| model | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for name, n, r in res:
        L.append(f"| {name} | " + " | ".join(f"{r['cats'][c]:.4f}" for c in CATS) + " |")
    L += ["", "## T2I-CompBench per category (BEST minus row)", "", "| model | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for name, n, r in res[:-1]:
        L.append(f"| {name} | " + " | ".join(f"{best['cats'][c] - r['cats'][c]:+.4f}" for c in CATS) + " |")
    sk = list(best["skills"])
    if sk:
        L += ["", "## GenEval2 per skill (absolute; Soft-TIFA per-atom mean by skill)", "", "| model | " + " | ".join(sk) + " |", "|---|" + "---|" * len(sk)]
        for name, n, r in res:
            L.append(f"| {name} | " + " | ".join(f"{r['skills'].get(k, float('nan')):.4f}" for k in sk) + " |")
        L += ["", "## GenEval2 per skill (BEST minus row)", "", "| model | " + " | ".join(sk) + " |", "|---|" + "---|" * len(sk)]
        for name, n, r in res[:-1]:
            L.append(f"| {name} | " + " | ".join(f"{best['skills'][k] - r['skills'].get(k, float('nan')):+.4f}" for k in sk) + " |")
    cp = ["3-4 atoms", "5-6 atoms", "7-8 atoms", "9+ atoms"]
    L += ["", "## GenEval2 per prompt complexity (number of scored atoms; absolute, then BEST minus row)", "",
          "| model | " + " | ".join(cp) + " |", "|---|" + "---|" * len(cp)]
    for name, n, r in res:
        L.append(f"| {name} | " + " | ".join(f"{r['comp'].get(k, float('nan')):.4f}" for k in cp) + " |")
    L += ["| **delta rows** | " + " | ".join("" for _ in cp) + " |"]
    for name, n, r in res[:-1]:
        L.append(f"| {name} | " + " | ".join(f"{best['comp'][k] - r['comp'].get(k, float('nan')):+.4f}" for k in cp) + " |")
    if args.eval10:
        L[2] = L[2].replace("one image per prompt", "the official TEN images per prompt (the unconverged raw-final row stays at one image)")
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(outfile).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
