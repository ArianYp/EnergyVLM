#!/usr/bin/env python3
"""Contact sheet for the qualitative comparison.

Every column shares the SAME initial noise for a given row: `exp0/phaseA_generate.py` seeds with
`--seed + pool_idx` and all six conditions were generated from pool manifests with an identical
prompt-list hash, so differences down a row are attributable to the model / step count / guidance
and not to the draw.

The rows are CHERRY-PICKED and the figure says so on its face. The selection rule is fixed and
stated: largest (scored - naive) per-prompt benchmark margin averaged over 3 training seeds, with
the additional requirement that seed 0 -- the seed actually displayed -- also shows the gap, and at
most 2 prompts per CompBench category so the sheet is not all one failure mode. This is a "where
does it work" panel, not evidence of typical behaviour.
"""
from __future__ import annotations

import argparse, json, textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_W, HEAD_H, FOOT_H = 256, 6, 330, 68, 30

COLUMNS = [
    ("base, 4 step\nno CFG",      "gen_{b}/images/BASE_cfg1/p{i:05d}/s4/cand0.png",   False),
    ("base, 4 step\nCFG 7",       "gen_{b}/images/BASE_cfg7/p{i:05d}/s4/cand0.png",   False),
    ("naive distill\n4 step",     "EVAL_B2",                                          True),
    ("DINO-patch scored\n4 step", "EVAL_DINOP",                                       True),
    ("base, 28 step\nno CFG",     "gen_{b}/images/BASE_cfg1/p{i:05d}/s28/cand0.png",  False),
    ("base, 28 step\nCFG 7",      "gen_{b}/images/BASE_cfg7/p{i:05d}/s28/cand0.png",  False),
]


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
    ap.add_argument("--root", default="phaseW/qual")
    ap.add_argument("--b2", required=True)
    ap.add_argument("--dinop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single_seed", action="store_true",
                    help="models have one seed: omit the 3-seed line and say so in the title/footer")
    ap.add_argument("--tag", default="seed 0", help="text after the benchmark name in the title")
    args = ap.parse_args()

    root = Path(args.root)
    sel = json.loads((root / "selection.json").read_text())[args.bench]
    b2lab, dplab = Path(args.b2).name.split("_", 1)[1].rsplit("_", 1)[0], None
    b2lab = [p.name for p in (Path(args.b2) / args.bench / "images").iterdir()][0]
    dplab = [p.name for p in (Path(args.dinop) / args.bench / "images").iterdir()][0]

    ncol, nrow = len(COLUMNS), len(sel)
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = HEAD_H + nrow * (CELL + FOOT_H + PAD) + PAD + 34
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    f_hd, f_pr, f_sc, f_ti = font(15, True), font(14), font(14, True), font(15, True)

    title = ("CHERRY-PICKED: 8 largest scored-over-naive margins"
             f"  |  {args.bench}  |  {args.tag}, identical initial noise across all columns")
    d.text((PAD, 8), title, fill=(150, 40, 40), font=f_ti)

    for c, (name, _, _) in enumerate(COLUMNS):
        x = LABEL_W + c * (CELL + PAD)
        col = (20, 90, 170) if "scored" in name else (60, 60, 60)
        for k, line in enumerate(name.split("\n")):
            d.text((x + 4, 32 + k * 17), line, fill=col, font=f_hd)

    for r, item in enumerate(sel):
        y = HEAD_H + r * (CELL + FOOT_H + PAD)
        i = item["idx"]
        wrapped = textwrap.wrap(item["prompt"], 34)[:5]
        d.text((PAD, y + 4), f"p{i:05d}  [{item['category']}]", fill=(120, 120, 120), font=font(12))
        for k, line in enumerate(wrapped):
            d.text((PAD, y + 22 + k * 18), line, fill=(15, 15, 15), font=f_pr)
        yy = y + 26 + len(wrapped) * 18
        d.text((PAD, yy), f"naive  {item['b2_s0']:.2f}", fill=(150, 60, 60), font=f_sc)
        d.text((PAD, yy + 18), f"scored {item['dinop_s0']:.2f}", fill=(20, 110, 40), font=f_sc)
        if not args.single_seed:
            d.text((PAD, yy + 40), f"(3-seed {item['b2']:.2f} -> {item['dinop']:.2f})",
                   fill=(120, 120, 120), font=font(12))

        for c, (name, tmpl, is_eval) in enumerate(COLUMNS):
            if tmpl == "EVAL_B2":
                p = Path(args.b2) / args.bench / "images" / b2lab / f"p{i:05d}" / "s4" / "cand0.png"
            elif tmpl == "EVAL_DINOP":
                p = Path(args.dinop) / args.bench / "images" / dplab / f"p{i:05d}" / "s4" / "cand0.png"
            else:
                p = root / tmpl.format(b=args.bench, i=i)
            x = LABEL_W + c * (CELL + PAD)
            sheet.paste(cell(p), (x, y))
            if "scored" in name:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(20, 90, 170), width=3)

        s = f"naive {item['b2_s0']:.2f}"
        d.text((LABEL_W + 2 * (CELL + PAD) + 4, y + CELL + 4), s, fill=(150, 60, 60), font=f_sc)
        d.text((LABEL_W + 3 * (CELL + PAD) + 4, y + CELL + 4),
               f"scored {item['dinop_s0']:.2f}", fill=(20, 110, 40), font=f_sc)

    rule = ("Selection rule: top (scored - naive) per-prompt margin of the two shown models (one seed each), "
            "max 2 per CompBench category. Not representative of typical behaviour."
            if args.single_seed else
            "Selection rule: top (scored - naive) 3-seed per-prompt margin, seed 0 must also show it, "
            "max 2 per CompBench category. Not representative of typical behaviour.")
    d.text((PAD, H - 26), rule, fill=(110, 110, 110), font=font(13))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
