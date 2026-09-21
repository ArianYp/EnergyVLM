#!/usr/bin/env python3
"""Qualitative sheets for the best branch (docs/k10/figs/: scored selection + refreshed projector
reward, K=10 teacher grid, seed 0, averaged checkpoint) against the naive K=8 baseline and the paper's
K=8 arm, on the SAME prompts and seeds (the eval images already on disk). Sheets, in order:
  1-2  strengths: the prompts where the best branch beats the naive baseline by the most (CompBench
       per-prompt score gap), at most 2 per category, then GenEval2
  3    typical: a random draw of prompts (seeded)
  4    failures: the prompts where the best branch loses to the naive baseline by the most
Each tile carries the evaluator's per-prompt score. CPU only. Reads the experimental tree's
phaseN/eval_*_<job> records (common/artifacts.py).

    python eval/k10_qual_sheets.py [--artifacts <tree>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
MODELS = [  # column name, eval job
    ("teacher 28-step cfg7", None),                # eval_REF_teacher_s28cfg7_* (same pool and seeds)
    ("naive K=8 (avg)", 145251),
    ("scored+reward K=8 (avg, paper)", 145253),
    ("scored+reward K=10 (avg, BEST)", 153475),
]


def eval_dir(job):
    if job is None:
        d = sorted(glob.glob("phaseN/eval_REF_teacher_s28cfg7_*"))
        return d[-1] if d else None
    d = glob.glob(f"phaseN/eval_*_{job}"); assert len(d) == 1, (job, d); return d[0]


def load(job, bench):
    """{key: (score, image path, prompt)}; key = (category, idx) for compbench, idx for geneval2."""
    d = eval_dir(job)
    if d is None:
        return {}
    label = os.listdir(f"{d}/{bench}/images")[0]
    steps = sorted(os.listdir(f"{d}/{bench}/images/{label}/p00000"))[0]
    out = {}
    if bench == "compbench":
        for c in CATS:
            # exact category match: '*_spatial' would also catch '_3d_spatial'
            fs = [f for f in glob.glob(f"{d}/compbench_scores/*_{c}/scores.json")
                  if os.path.basename(os.path.dirname(f)).endswith(f"_{steps}_{c}")]
            if not fs:
                print(f"[warn] no scores for {c} in {d}")
                continue
            for r in json.load(open(fs[0])).get("per_prompt", []):
                out[(c, int(r["idx"]))] = (float(r["score"]), f"{d}/compbench/images/{label}/p{int(r['idx']):05d}/{steps}/cand0.png", r["prompt"])
    else:
        fs = glob.glob(f"{d}/geneval2_scores/*/scores.json")
        if fs:
            for r in json.load(open(fs[0])).get("per_prompt", []):
                out[int(r["idx"])] = (float(r["score"]), f"{d}/geneval2/images/{label}/p{int(r['idx']):05d}/{steps}/cand0.png", r["prompt"])
    return out


def sheet(rows, data, title, path, T=232, pad=6, cap_h=62):
    font = ImageFont.load_default()
    cols = [m for m in MODELS if data[m[0]]]
    W = pad + len(cols) * (T + pad); H = pad + 26 + len(rows) * (T + cap_h + pad)
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((pad, pad), title, fill="#000", font=font)
    for c, (name, _) in enumerate(cols):
        dr.text((pad + c * (T + pad), pad + 13), name, fill="#333", font=font)
    for r, (key, prompt, tag) in enumerate(rows):
        y = pad + 26 + r * (T + cap_h + pad)
        for c, (name, _) in enumerate(cols):
            x = pad + c * (T + pad)
            rec = data[name].get(key)
            if rec is None:
                dr.rectangle([x, y, x + T, y + T], outline="#ccc"); dr.text((x + 4, y + 4), "n/a", fill="#999", font=font); continue
            sc, p, _ = rec
            try:
                im.paste(Image.open(p).convert("RGB").resize((T, T)), (x, y))
            except Exception:
                dr.rectangle([x, y, x + T, y + T], outline="#c00")
            col = "#060" if name.startswith("scored+reward K=10") else "#000"
            dr.text((x + 2, y + T + 2), f"score {sc:.2f}", fill=col, font=font)
        dr.multiline_text((pad + 2, y + T + 14), "\n".join(textwrap.wrap(f"[{tag}] {prompt}", 60)[:3]), fill="#000", font=font)
    im.save(path, quality=90); print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--fig_dir", default=str(ROOT / "docs" / "k10" / "figs"))
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    fig = args.fig_dir; os.makedirs(fig, exist_ok=True)
    rng = np.random.default_rng(0)
    for bench in ("compbench", "geneval2"):
        data = {m[0]: load(m[1], bench) for m in MODELS}
        best, base = data[MODELS[3][0]], data[MODELS[1][0]]
        keys = sorted(set(best) & set(base))
        gap = {k: best[k][0] - base[k][0] for k in keys}
        # strengths: largest gaps, category-diverse
        order = sorted(keys, key=lambda k: -gap[k]); picked, per_cat = [], {}
        for k in order:
            c = k[0] if bench == "compbench" else "geneval2"
            if per_cat.get(c, 0) >= (2 if bench == "compbench" else 8):
                continue
            picked.append(k); per_cat[c] = per_cat.get(c, 0) + 1
            if len(picked) >= 8:
                break
        rows = [(k, best[k][2], f"{k[0] if bench == 'compbench' else 'geneval2'} gap {gap[k]:+.2f}") for k in picked]
        sheet(rows, data, f"{bench}: STRENGTHS of the best branch (largest per-prompt gain over naive K=8; scores under each image)", f"{fig}/{bench}_strengths.jpg")
        # typical
        draw = [keys[i] for i in rng.choice(len(keys), 6, replace=False)]
        rows = [(k, best[k][2], f"{k[0] if bench == 'compbench' else 'geneval2'} gap {gap[k]:+.2f}") for k in draw]
        sheet(rows, data, f"{bench}: TYPICAL prompts (random draw, not selected)", f"{fig}/{bench}_typical.jpg")
        # failures
        worst = sorted(keys, key=lambda k: gap[k])[:6]
        rows = [(k, best[k][2], f"{k[0] if bench == 'compbench' else 'geneval2'} gap {gap[k]:+.2f}") for k in worst]
        sheet(rows, data, f"{bench}: FAILURES of the best branch (largest per-prompt loss vs naive K=8)", f"{fig}/{bench}_failures.jpg")
        g = np.array(list(gap.values()))
        print(f"{bench}: n={len(g)} best-vs-naive mean gap {g.mean():+.4f}, wins {np.mean(g > 0):.3f} losses {np.mean(g < 0):.3f} ties {np.mean(g == 0):.3f}")


if __name__ == "__main__":
    main()
