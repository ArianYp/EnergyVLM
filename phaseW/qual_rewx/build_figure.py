#!/usr/bin/env python3
"""Contact sheet for the exact-reward qualitative comparison (same conventions as
phaseW/build_qual_figure.py). Every column of a row shares the SAME initial noise: the generation
job seeds with `--seed + pool_idx` for every model, so differences down a row are attributable to
the model / step count / guidance and not to the draw.

Rows are CHERRY-PICKED and the figure says so: the 8 largest (exact reward - argmax) per-prompt
margins (at most 2 per CompBench category), followed by the 4 most negative margins, so the sheet
shows both where the reward helps and where it hurts. Averaged checkpoints, seed 0, of every arm.
"""
from __future__ import annotations

import argparse, json, textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_W, HEAD_H, FOOT_H = 256, 6, 330, 68, 30
COLUMNS = [  # (header, label dir, steps, score key or None, framed)
    ("base, 4 step\nno CFG", "BASE_cfg1", 4, None, False),
    ("base, 4 step\nCFG 7", "BASE_cfg7", 4, None, False),
    ("random pick\n4 step", "RANDOM", 4, "random", False),
    ("argmax (DINO-patch)\n4 step", "ARGMAX", 4, "argmax", False),
    ("argmax + exact reward\n4 step", "REWARD", 4, "reward", True),
    ("base, 28 step\nno CFG", "BASE_cfg1", 28, None, False),
    ("base, 28 step\nCFG 7", "BASE_cfg7", 28, None, False),
]
SCORE_COL = {"random": (150, 60, 60), "argmax": (20, 90, 170), "reward": (20, 110, 40)}


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell(path: Path) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    im = Image.new("RGB", (CELL, CELL), (32, 32, 36))
    ImageDraw.Draw(im).text((10, CELL // 2 - 8), "missing", fill=(200, 90, 90), font=font(15))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["compbench", "geneval2"])
    ap.add_argument("--root", default="phaseW/qual_rewx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None, help="override the sheet title")
    ap.add_argument("--rule", default=None, help="override the selection-rule footer")
    args = ap.parse_args()
    root = Path(args.root)
    sel = json.loads((root / "selection.json").read_text())[args.bench]
    ncol, nrow = len(COLUMNS), len(sel)
    random_rows = all(r.get("panel") == "random" for r in sel)
    n_gain = nrow if random_rows else sum(r.get("panel", "gain") == "gain" for r in sel)
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = HEAD_H + nrow * (CELL + FOOT_H + PAD) + PAD + 34 + 26
    sheet = Image.new("RGB", (W, H), (250, 250, 250)); d = ImageDraw.Draw(sheet)
    f_hd, f_pr, f_sc, f_ti = font(15, True), font(14), font(14, True), font(15, True)
    if args.title:
        d.text((PAD, 8), args.title, fill=(150, 40, 40), font=f_ti)
    elif random_rows:
        d.text((PAD, 8), f"RANDOMLY SELECTED prompts (fixed draw, not chosen by any score)  |  {args.bench}  |  "
                         "averaged checkpoints, seed 0, identical initial noise across all columns", fill=(40, 90, 40), font=f_ti)
    else:
        d.text((PAD, 8), f"CHERRY-PICKED: {n_gain} largest (exact reward - argmax) margins, then {nrow - n_gain} most negative  |  {args.bench}  |  "
                         "averaged checkpoints, seed 0, identical initial noise across all columns", fill=(150, 40, 40), font=f_ti)
    for c, (name, *_rest) in enumerate(COLUMNS):
        x = LABEL_W + c * (CELL + PAD)
        col = (20, 110, 40) if "reward" in name else ((20, 90, 170) if "argmax" in name else (60, 60, 60))
        for k, line in enumerate(name.split("\n")):
            d.text((x + 4, 32 + k * 17), line, fill=col, font=f_hd)
    y_off = 0
    for r, item in enumerate(sel):
        if r == n_gain:
            y0 = HEAD_H + r * (CELL + FOOT_H + PAD) + y_off
            d.line([(PAD, y0 + 4), (W - PAD, y0 + 4)], fill=(150, 40, 40), width=2)
            d.text((PAD, y0 + 8), "where the reward HURTS (most negative margins)", fill=(150, 40, 40), font=f_sc); y_off = 26
        y = HEAD_H + r * (CELL + FOOT_H + PAD) + y_off
        i = item["idx"]
        wrapped = textwrap.wrap(item["prompt"], 34)[:5]
        d.text((PAD, y + 4), f"p{i:05d}  [{item['category']}]", fill=(120, 120, 120), font=font(12))
        for k, line in enumerate(wrapped):
            d.text((PAD, y + 22 + k * 18), line, fill=(15, 15, 15), font=f_pr)
        yy = y + 26 + len(wrapped) * 18
        for k, key in enumerate(("random", "argmax", "reward")):
            d.text((PAD, yy + 18 * k), f"{key:7s} {item[key]:.2f}", fill=SCORE_COL[key], font=f_sc)
        for c, (name, lab, steps, key, framed) in enumerate(COLUMNS):
            p = root / f"gen_{args.bench}" / "images" / lab / f"p{i:05d}" / f"s{steps}" / "cand0.png"
            x = LABEL_W + c * (CELL + PAD)
            sheet.paste(cell(p), (x, y))
            if framed:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(20, 110, 40), width=3)
            if key:
                d.text((x + 4, y + CELL + 4), f"{key} {item[key]:.2f}", fill=SCORE_COL[key], font=f_sc)
    rule = (args.rule if args.rule else
            "Selection rule: uniform random draw of prompts (numpy seed 2026), no score involved. Representative of typical behaviour."
            if random_rows else
            "Selection rule: per-prompt (exact reward - argmax) benchmark margin of the two shown models (one seed each), "
            "max 2 per CompBench category, top 8 then bottom 4. Not representative of typical behaviour.")
    d.text((PAD, H - 26), rule, fill=(110, 110, 110), font=font(13))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); sheet.save(args.out)
    print(f"wrote {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
