#!/usr/bin/env python3
"""Contact sheet for the inference step sweep: one student (argmax + exact reward, averaged seed 0)
sampled at 2, 4, 8, 16 and 28 Euler steps (w = 1) from identical initial noise per prompt, with the
base model at 28 steps / w = 7 as the reference column. Rows are the randomly drawn prompts of the
representative sheet (no score involved). Per-prompt scores under each column come from the sweep's
evaluation score files (one image per prompt).
"""
from __future__ import annotations

import argparse, glob, json, textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_W, HEAD_H, FOOT_H = 256, 6, 330, 68, 30
STEPS = [2, 4, 8, 16, 28]


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell(path):
    if path.exists():
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    im = Image.new("RGB", (CELL, CELL), (32, 32, 36)); ImageDraw.Draw(im).text((10, CELL // 2 - 8), "missing", fill=(200, 90, 90), font=font(15)); return im


def scores(bench, st, pool):
    """idx -> per-prompt score of the sweep evaluation at `st` steps."""
    lab = "S4_CD_dinop_hard-rewXi-avglast3_s0" if st == 4 else f"S4_CD_dinop_hard-rewXi-avglast3-st{st}_s0"
    dirs = sorted(glob.glob(f"phaseN/eval_{lab}_*"))
    out = {}
    for d in dirs:
        if bench == "compbench":
            for p in glob.glob(f"{d}/compbench_scores/*/scores.json"):
                for r in json.load(open(p))["per_prompt"]:
                    key = (r["category"], r["prompt"])
                    if key in pool: out[pool[key]] = float(r["score"])
        else:
            for p in glob.glob(f"{d}/geneval2_scores/*/scores.json"):
                for r in json.load(open(p))["per_prompt"]:
                    out[int(r["idx"])] = float(r["score"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["compbench", "geneval2"])
    ap.add_argument("--sel_root", default="phaseW/qual_rewx_random")
    ap.add_argument("--gen_root", default="phaseW/qual_steps")
    ap.add_argument("--pool", default="phaseFP/eval_pool_101203")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = json.loads((Path(args.sel_root) / "selection.json").read_text())[args.bench]
    cb_pool = {(r["category"], r["prompt"]): int(r["idx"]) for r in json.load(open(f"{args.pool}/compbench_prompts.json"))}
    sc = {st: scores(args.bench, st, cb_pool) for st in STEPS}
    cols = [(f"student\n{st} steps, no CFG", st) for st in STEPS] + [("base, 28 step\nCFG 7", None)]
    ncol, nrow = len(cols), len(sel)
    W = LABEL_W + ncol * (CELL + PAD) + PAD; H = HEAD_H + nrow * (CELL + FOOT_H + PAD) + PAD + 34
    sheet = Image.new("RGB", (W, H), (250, 250, 250)); d = ImageDraw.Draw(sheet)
    f_hd, f_pr, f_sc, f_ti = font(15, True), font(14), font(14, True), font(15, True)
    d.text((PAD, 8), f"INFERENCE STEPS: argmax + exact reward student (averaged, seed 0) at 2 / 4 / 8 / 16 / 28 Euler steps, w = 1  |  {args.bench}  |  "
                     "randomly drawn prompts, identical initial noise across columns", fill=(40, 60, 120), font=f_ti)
    for c, (name, st) in enumerate(cols):
        x = LABEL_W + c * (CELL + PAD); col = (20, 110, 40) if st == 4 else (60, 60, 60)
        for k, line in enumerate(name.split("\n")):
            d.text((x + 4, 32 + k * 17), line, fill=col, font=f_hd)
    for r, item in enumerate(sel):
        y = HEAD_H + r * (CELL + FOOT_H + PAD); i = item["idx"]
        wrapped = textwrap.wrap(item["prompt"], 34)[:5]
        d.text((PAD, y + 4), f"p{i:05d}  [{item['category']}]", fill=(120, 120, 120), font=font(12))
        for k, line in enumerate(wrapped):
            d.text((PAD, y + 22 + k * 18), line, fill=(15, 15, 15), font=f_pr)
        for c, (name, st) in enumerate(cols):
            x = LABEL_W + c * (CELL + PAD)
            if st is None:
                p = Path(args.sel_root) / f"gen_{args.bench}" / "images" / "BASE_cfg7" / f"p{i:05d}" / "s28" / "cand0.png"
            else:
                p = Path(args.gen_root) / f"gen_{args.bench}" / "images" / "STEPS" / f"p{i:05d}" / f"s{st}" / "cand0.png"
            sheet.paste(cell(p), (x, y))
            if st == 4:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(20, 110, 40), width=3)
            if st is not None and i in sc[st]:
                d.text((x + 4, y + CELL + 4), f"score {sc[st][i]:.2f}", fill=(20, 110, 40) if st == 4 else (60, 60, 60), font=f_sc)
    d.text((PAD, H - 26), "Prompts: uniform random draw (numpy seed 2026). Scores: the step-sweep evaluations (one image per prompt, same noise). The 4-step column is the deployment setting.", fill=(110, 110, 110), font=font(13))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); sheet.save(args.out); print(f"wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
