#!/usr/bin/env python3
"""Contact sheet of the REFERENCE images that become the projector-reward targets (docs/bench/): for
each category, a few prompts showing the best-of-N pick (what the reward points at) next to the worst
of the same N, both annotated with the category's official evaluator score. 40 steps, cfg 4.5, the
teacher's own documented setting. CPU only.

    python eval/bench_ref_sheet.py --per_cat 3
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="cache/bench/scores")
    ap.add_argument("--label", default="bench_ref_k40cfg45")
    ap.add_argument("--images", default="cache/bench/images/bench_ref_k40cfg45")
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--per_cat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/bench_ref_sheet.jpg")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for c in CATS:
        f = f"{args.scores}/{args.label}_s{args.K}_{c}/scores.json"
        if not os.path.exists(f):
            print(f"[skip] {c}: not scored yet"); continue
        pp = json.load(open(f))["per_prompt"]
        # prefer prompts where selection actually matters (a wide spread over the candidates)
        pp = [r for r in pp if len(r["image_scores"]) >= 2]
        spread = np.array([max(r["image_scores"]) - min(r["image_scores"]) for r in pp])
        pick = np.argsort(-spread)[:max(args.per_cat * 4, 12)]
        for k in rng.choice(pick, size=min(args.per_cat, len(pick)), replace=False):
            r = pp[int(k)]
            s = r["image_scores"]
            best, worst = int(np.argmax(s)), int(np.argmin(s))
            rows.append((c, r["prompt"], int(r["idx"]), (best, s[best]), (worst, s[worst])))
    if not rows:
        print("nothing scored yet"); return
    T, pad = 236, 6
    font = ImageFont.load_default()
    W = pad + 2 * (T + pad) + 430
    im = Image.new("RGB", (W, pad + 20 + len(rows) * (T + pad)), "white")
    dr = ImageDraw.Draw(im)
    dr.text((pad, pad), "Reward references: best-of-8 teacher samples, 40 steps cfg 4.5, scored by each category's official evaluator", fill="#000", font=font)
    dr.text((pad, pad + 10), "left = BEST (the reward target)   middle = WORST of the same 8   right = prompt", fill="#555", font=font)
    for i, (c, prompt, idx, (bj, bs), (wj, ws)) in enumerate(rows):
        y = pad + 20 + i * (T + pad)
        for col, (j, sc, tag) in enumerate([(bj, bs, "BEST"), (wj, ws, "worst")]):
            p = f"{args.images}/p{idx:05d}/s{args.K}/cand{j}.png"
            x = pad + col * (T + pad)
            if os.path.exists(p):
                im.paste(Image.open(p).convert("RGB").resize((T, T)), (x, y))
            else:
                dr.rectangle([x, y, x + T, y + T], outline="#ccc")
            dr.text((x + 3, y + 3), f"{tag}  {sc:.3f}", fill="#0a0" if tag == "BEST" else "#a00", font=font)
        tx = pad + 2 * (T + pad) + 4
        dr.text((tx, y + 4), f"[{c}]  idx {idx}", fill="#555", font=font)
        dr.multiline_text((tx, y + 18), "\n".join(textwrap.wrap(prompt, 52)[:6]), fill="#000", font=font, spacing=3)
        dr.text((tx, y + T - 14), f"selection gain on this prompt: {bs - ws:+.3f}", fill="#333", font=font)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    im.save(args.out, quality=90)
    print(f"wrote {args.out}  ({len(rows)} prompts, {len({r[0] for r in rows})} categories)")


if __name__ == "__main__":
    main()
