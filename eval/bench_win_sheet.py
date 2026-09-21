#!/usr/bin/env python3
"""Where scored selection wins (docs/bench/): prompts with the largest per-prompt gain of the
evaluator-argmax arm over the random-pick control on the SAME held-out prompts, in the categories where
our delta beats CTCal's. Both images come from the same evaluation, so prompt and seed are identical;
only the training differs. Reads the experimental tree's phaseN/eval_<label>_<job> records. CPU only.

    python eval/bench_win_sheet.py --cats 3d_spatial,shape,complex,spatial --per_cat 3 [--artifacts <tree>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402

ARG = "W_CD_bench_hard-k10-hp1_bench-hp1-avg_s0"      # evaluator-argmax arm (--selector bench)
CTL = "W_B2-k10-hp1_bench-hp1-avg_s0"                 # random-pick control on the same cache


def eval_dir(label):
    ds = [d for d in sorted(glob.glob(f"phaseN/eval_{label}_*"))
          if os.path.exists(f"{d}/alignment.json") and re.fullmatch(rf"eval_{re.escape(label)}_\d+", os.path.basename(d))]
    assert ds, label
    return ds[-1]


def per_prompt(d, cat):
    fs = [f for f in glob.glob(f"{d}/compbench_scores/*_{cat}/scores.json")
          if re.search(rf"_s\d+_{re.escape(cat)}$", os.path.basename(os.path.dirname(f)))]
    assert fs, (d, cat)
    return {int(r["idx"]): (float(r["score"]), r["prompt"]) for r in json.load(open(fs[0]))["per_prompt"]}


def img(d, label, idx, steps=4):
    p = Path(d) / "compbench" / "images" / label / f"p{idx:05d}" / f"s{steps}" / "cand0.png"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--cats", default="3d_spatial,shape,complex,spatial")
    ap.add_argument("--per_cat", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs" / "bench" / "win_sheet.jpg"))
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    da, db = eval_dir(ARG), eval_dir(CTL)
    rows = []
    for c in args.cats.split(","):
        A, B = per_prompt(da, c), per_prompt(db, c)
        common = sorted(set(A) & set(B))
        gaps = sorted(common, key=lambda i: -(A[i][0] - B[i][0]))[:args.per_cat]
        for i in gaps:
            rows.append((c, i, A[i][1], A[i][0], B[i][0]))
    T, pad = 244, 6
    font = ImageFont.load_default()
    W = pad + 2 * (T + pad) + 400
    im = Image.new("RGB", (W, pad + 22 + len(rows) * (T + pad)), "white")
    dr = ImageDraw.Draw(im)
    dr.text((pad, pad), "Where scored selection wins: same prompt, same noise, same recipe — only the distilled trajectory differs", fill="#000", font=font)
    dr.text((pad, pad + 11), "left = random-pick control      right = scored selection (ours)", fill="#555", font=font)
    for r, (c, idx, prompt, sa, sb) in enumerate(rows):
        y = pad + 22 + r * (T + pad)
        for col, (d, lab, sc, tag, colour) in enumerate([(db, CTL, sb, "random", "#a00"), (da, ARG, sa, "OURS", "#0a0")]):
            x = pad + col * (T + pad)
            p = img(d, lab, idx)
            if p:
                im.paste(Image.open(p).convert("RGB").resize((T, T)), (x, y))
            else:
                dr.rectangle([x, y, x + T, y + T], outline="#ccc")
            dr.text((x + 3, y + 3), f"{tag}  {sc:.3f}", fill=colour, font=font)
        tx = pad + 2 * (T + pad) + 4
        dr.text((tx, y + 4), f"[{c}]  idx {idx}", fill="#555", font=font)
        dr.multiline_text((tx, y + 18), "\n".join(textwrap.wrap(prompt, 48)[:5]), fill="#000", font=font, spacing=3)
        dr.text((tx, y + T - 14), f"gain {sa - sb:+.3f}", fill="#000", font=font)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out, quality=90)
    print(f"wrote {args.out} ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
