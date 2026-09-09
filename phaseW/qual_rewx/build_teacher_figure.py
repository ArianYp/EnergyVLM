#!/usr/bin/env python3
"""Contact sheet: the best student (argmax + exact reward, averaged seed 0) at its 4-step deployment
setting and pushed to 28 steps, against the frozen teacher at 28 steps without and with guidance
(w = 7), identical initial noise per prompt. Per-prompt scores from the corresponding evaluations
(student: the averaged-model and step-sweep evaluations; teacher: the REF_teacher_s28cfg7 run).
"""
from __future__ import annotations

import argparse, glob, json, textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_W, HEAD_H, FOOT_H = 256, 6, 330, 68, 30


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell(path):
    if path.exists():
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    im = Image.new("RGB", (CELL, CELL), (32, 32, 36)); ImageDraw.Draw(im).text((10, CELL // 2 - 8), "missing", fill=(200, 90, 90), font=font(15)); return im


def scores(bench, eval_glob, pool):
    out = {}
    for d in sorted(glob.glob(eval_glob)):
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
    ap.add_argument("--sel_root", required=True, help="dir with selection.json")
    ap.add_argument("--student_root", required=True, help="gen_<bench>/images/STEPS/p*/s{4,28}")
    ap.add_argument("--base_root", required=True, help="gen_<bench>/images/BASE_cfg{1,7}/p*/s28")
    ap.add_argument("--student_eval4", default="phaseN/eval_S4_CD_dinop_hard-rewXi-avglast3_s0_*")
    ap.add_argument("--student_eval28", default="phaseN/eval_S4_CD_dinop_hard-rewXi-avglast3-st28_s0_*")
    ap.add_argument("--teacher_eval", default="phaseN/eval_REF_teacher_s28cfg7_*")
    ap.add_argument("--pool", default="phaseFP/eval_pool_101203")
    ap.add_argument("--title", default=None); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = json.loads((Path(args.sel_root) / "selection.json").read_text())[args.bench]
    cb_pool = {(r["category"], r["prompt"]): int(r["idx"]) for r in json.load(open(f"{args.pool}/compbench_prompts.json"))}
    sc = {"s4": scores(args.bench, args.student_eval4, cb_pool), "s28": scores(args.bench, args.student_eval28, cb_pool), "t28": scores(args.bench, args.teacher_eval, cb_pool)}
    cols = [("best student\n4 steps, no CFG", Path(args.student_root) / f"gen_{args.bench}/images/STEPS", "s4", "s4", True),
            ("best student\n28 steps, no CFG", Path(args.student_root) / f"gen_{args.bench}/images/STEPS", "s28", "s28", False),
            ("frozen teacher\n28 steps, no CFG", Path(args.base_root) / f"gen_{args.bench}/images/BASE_cfg1", "s28", None, False),
            ("frozen teacher\n28 steps, CFG 7", Path(args.base_root) / f"gen_{args.bench}/images/BASE_cfg7", "s28", "t28", False)]
    ncol, nrow = len(cols), len(sel)
    W = LABEL_W + ncol * (CELL + PAD) + PAD; H = HEAD_H + nrow * (CELL + FOOT_H + PAD) + PAD + 34
    sheet = Image.new("RGB", (W, H), (250, 250, 250)); d = ImageDraw.Draw(sheet)
    f_hd, f_pr, f_sc, f_ti = font(15, True), font(14), font(14, True), font(15, True)
    d.text((PAD, 8), args.title or f"STUDENT vs TEACHER  |  {args.bench}  |  argmax + exact reward student (averaged, seed 0) at 4 and 28 steps vs the frozen SD3.5-M teacher at 28 steps  |  identical initial noise", fill=(40, 60, 120), font=f_ti)
    for c, (name, *_r) in enumerate(cols):
        x = LABEL_W + c * (CELL + PAD); col = (20, 110, 40) if c == 0 else ((120, 40, 40) if c == 3 else (60, 60, 60))
        for k, line in enumerate(name.split("\n")):
            d.text((x + 4, 32 + k * 17), line, fill=col, font=f_hd)
    for r, item in enumerate(sel):
        y = HEAD_H + r * (CELL + FOOT_H + PAD); i = item["idx"]
        wrapped = textwrap.wrap(item["prompt"], 34)[:5]
        d.text((PAD, y + 4), f"p{i:05d}  [{item['category']}]", fill=(120, 120, 120), font=font(12))
        for k, line in enumerate(wrapped):
            d.text((PAD, y + 22 + k * 18), line, fill=(15, 15, 15), font=f_pr)
        for c, (name, root, sdir, skey, framed) in enumerate(cols):
            x = LABEL_W + c * (CELL + PAD)
            sheet.paste(cell(root / f"p{i:05d}" / sdir / "cand0.png"), (x, y))
            if framed:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(20, 110, 40), width=3)
            if skey and i in sc[skey]:
                d.text((x + 4, y + CELL + 4), f"score {sc[skey][i]:.2f}", fill=(20, 110, 40) if c == 0 else ((120, 40, 40) if c == 3 else (60, 60, 60)), font=f_sc)
    d.text((PAD, H - 26), "Rows: uniform random draw within the stated prompt set, no score involved. Scores: single-image evaluations of the same models (teacher: 28 steps, CFG 7).", fill=(110, 110, 110), font=font(13))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); sheet.save(args.out); print(f"wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
