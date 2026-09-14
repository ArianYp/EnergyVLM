#!/usr/bin/env python3
"""Figures for iclr2027/iclr2027_conference.tex. Every number is read from iclr2027/numbers.json
(written by verify_numbers.py from the raw evaluation records) or iclr2027/monitor.json (wandb
export); the qualitative sheets read the evaluation images directly.

    python3 iclr2027/make_paper_figures.py               # all figures -> iclr2027/figs/
    python3 iclr2027/make_paper_figures.py --candidates  # also dump per-category candidate sheets for picking rows
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "iclr2027"))
from verify_numbers import CATS, CAT_NAMES, load_dir  # noqa: E402

OUT = os.path.join(ROOT, "iclr2027", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "serif", "mathtext.fontset": "cm"})
NUM = json.load(open("iclr2027/numbers.json"))
MON = json.load(open("iclr2027/monitor.json"))["data"] if os.path.isfile("iclr2027/monitor.json") else {}

C_NAIVE, C_ARGMAX, C_PROJ, C_OURS, C_EXACT, C_TEACH = "#7a7a7a", "#4c72b0", "#8fbc8f", "#1a7f37", "#d9822b", "#333333"

ARM_ROWS = [  # numbers.json key, short display name, colour, bold?
    ("S4_B2-avglast3", "Naive CD (random trajectory, no reward)", C_NAIVE, False),
    ("S4_CD_dinop_hard-rewF-avglast3", "Scored + projector reward, frozen projector", C_PROJ, False),
    ("S4_CD_dinop_hard-rewR-avglast3", "Scored + projector reward, refresh /100, 4 steps", C_PROJ, False),
    ("S4_CD_dinop_hard-rewRi-e25-avglast3", "Scored + projector reward, refresh /25, 4 steps", C_PROJ, False),
    ("S4_CD_dinop_hard-rewRi-s16-avglast3", "Scored + projector reward, refresh /100, 16 steps (ours)", C_OURS, True),
    ("S4_CD_dinop_hard-rewRi-s16-maxpool-avglast3", "Ours with max-pooled DINOv2 patches", C_PROJ, False),
]


# ----------------------------------------------------------------------------- 1. all arms, 3k
def fig_arms():
    m = NUM["main3k"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(ARM_ROWS))[::-1]
    for ax, key, title in zip(axes, ("cb", "ge2"), ("T2I-CompBench (mean of 8 categories)", "GenEval2 ($\\times$100)")):
        for yi, (k, name, col, bold) in zip(y, ARM_ROWS):
            v = np.array([x for s, x in zip(m[k][key]["seeds"], m[k][key]["values"]) if s in (0, 1, 2)])
            ax.scatter(v, [yi] * len(v), s=14, color=col, alpha=0.6, zorder=3, linewidths=0)
            ax.errorbar([v.mean()], [yi], xerr=[v.std(ddof=1)], fmt="D", ms=4.5, color=col, capsize=2.5, lw=1, zorder=4,
                        markeredgecolor="black" if bold else col, markeredgewidth=0.8 if bold else 0)
        ax.axvline(np.mean([x for s, x in zip(m["S4_B2-avglast3"][key]["seeds"], m["S4_B2-avglast3"][key]["values"]) if s in (0, 1, 2)]), color=C_NAIVE, lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5); ax.grid(axis="x", lw=0.3, alpha=0.5); ax.tick_params(labelsize=7.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([n for _, n, _, _ in ARM_ROWS], fontsize=7.5)
    for lab, (_, _, _, bold) in zip(axes[0].get_yticklabels(), ARM_ROWS):
        if bold:
            lab.set_fontweight("bold")
    axes[0].set_xlabel("3k captions, seeds 0--2, averaged checkpoints; dots = seeds, diamond = mean $\\pm$ sd; dotted = naive mean", fontsize=7)
    fig.savefig(os.path.join(OUT, "arms_3k.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------- 2. per-category gains
def fig_category():
    """Per-category CompBench of naive CD and ours (seeds 0-2) and the seed-paired gain."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(CATS)); w = 0.38
    c3 = NUM["cat3k"]
    ax = axes[0]
    nv = [c3[c]["naive"] for c in CATS]; ov = [c3[c]["ours"] for c in CATS]
    ax.bar(x - w / 2, [r["mean"] for r in nv], w, color=C_NAIVE, alpha=0.85, label="naive CD")
    ax.bar(x + w / 2, [r["mean"] for r in ov], w, color=C_OURS, alpha=0.9, label="ours")
    for i in range(len(CATS)):
        ax.scatter([x[i] - w / 2] * len(nv[i]["values"]), nv[i]["values"], s=7, color="black", zorder=3, alpha=0.6, linewidths=0)
        ax.scatter([x[i] + w / 2] * len(ov[i]["values"]), ov[i]["values"], s=7, color="black", zorder=3, alpha=0.6, linewidths=0)
    ax.set_xticks(x); ax.set_xticklabels([CAT_NAMES[c] for c in CATS], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("T2I-CompBench score", fontsize=7.5); ax.set_title("(a) per-category score, 3 seeds each", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right"); ax.tick_params(labelsize=7)
    ax = axes[1]
    gains = [c3[c]["ours_vs_naive"] for c in CATS]
    ax.bar(x, [g_["mean"] for g_ in gains], 0.6, color=C_OURS, alpha=0.9)
    for i, g_ in enumerate(gains):
        ax.scatter([x[i]] * len(g_["per_seed"]), g_["per_seed"], s=8, color="black", zorder=3, alpha=0.65, linewidths=0)
        if g_["t_p"] < 0.05:
            ax.text(x[i], max(g_["per_seed"]) + 0.002, "*", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([CAT_NAMES[c] for c in CATS], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("ours $-$ naive (seed-paired)", fontsize=7.5); ax.set_title("(b) gain per category; * $p<0.05$", fontsize=8.5); ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "category_gains.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------- 3. training-time monitor
def _series(arm, seed, key):
    pts = [(st, float(v)) for st, v in MON[arm][str(seed)].get(key, []) if isinstance(v, (int, float)) and v == v and st % 100 == 0]
    pts = sorted(set(pts))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def _smooth(y, k=5):
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([y[: k - 1], c])


def fig_monitor():
    if not MON:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    c_frozen = "#8172b3"
    arms = [("rewRi-s16", "ours: argmax + projector reward, refreshed", C_OURS), ("rewF", "argmax + projector reward, frozen", c_frozen),
            ("rewXi", "argmax + decode-based DINOv2 reward", C_EXACT)]
    ax = axes[0]
    for arm, name, col in arms:
        for i, s in enumerate(sorted(MON[arm], key=int)):
            t, v = _series(arm, s, "reward/rgb_score")
            ax.plot(t, _smooth(v), color=col, lw=0.9, alpha=0.85, label=name if i == 0 else None)
    ax.set_xlabel("training update", fontsize=7.5); ax.set_ylabel("true DINOv2 score of decoded $\\hat x_0$", fontsize=7.5); ax.set_title("(a) true score of the predictions", fontsize=8)
    h_a, l_a = ax.get_legend_handles_labels()
    ax = axes[1]
    for i, s in enumerate(sorted(MON["rewRi-s16"], key=int)):
        t, v = _series("rewRi-s16", s, "reward/proj_score"); t2, v2 = _series("rewRi-s16", s, "reward/rgb_score")
        ax.plot(t, _smooth(v), color=C_OURS, lw=0.9, ls="-", label="ours: projector score (the reward)" if i == 0 else None)
        ax.plot(t2, _smooth(v2), color=C_OURS, lw=0.9, ls="--", alpha=0.6, label="ours: true score of the same latents" if i == 0 else None)
    ax.set_xlabel("training update", fontsize=7.5); ax.set_title("(b) ours: reward proxy vs. truth", fontsize=8)
    h_b, l_b = ax.get_legend_handles_labels()
    ax = axes[2]
    for arm, name, col in arms[:2]:
        for i, s in enumerate(sorted(MON[arm], key=int)):
            t, v = _series(arm, s, "reward/rgb_proj_corr")
            ax.plot(t, _smooth(v), color=col, lw=0.9, alpha=0.85)
    ax.set_xlabel("training update", fontsize=7.5); ax.set_ylabel("corr(projector, true), 32 latents", fontsize=7.5); ax.set_title("(c) proxy--truth correlation", fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    for ax in axes:
        ax.tick_params(labelsize=7); ax.grid(lw=0.3, alpha=0.4)
    fig.subplots_adjust(wspace=0.38)
    fig.legend(h_a + h_b, l_a + l_b, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.16), handlelength=2.2)
    fig.savefig(os.path.join(OUT, "monitor.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------- 4. 118k checkpoint curve
def fig_ckpt():
    cc = NUM["ckpt_curve"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for base, name, col in (("W_B2_118k", "naive CD (random trajectory)", C_NAIVE), ("W_CD_dinop_hard_118k", "argmax selection", C_ARGMAX)):
        pts = sorted((int(k), v) for k, v in cc[base]["steps"].items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.8, lw=1.0, color=col, label=name)
        ax.plot([56974], [cc[base]["avg"]], marker="*", ms=10, color=col, ls="none", markeredgecolor="black", markeredgewidth=0.4)
        ax.annotate(f"avg {cc[base]['avg']:.4f}", (56974, cc[base]["avg"]), xytext=(6, 6 if base.startswith("W_CD") else -8), textcoords="offset points", fontsize=6.5, color=col, va="center")
    ax.set_xlabel("optimiser update (118k pool, seed 0)", fontsize=7.5); ax.set_ylabel("T2I-CompBench", fontsize=7.5); ax.set_xlim(0, 72000)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.tick_params(labelsize=7); ax.grid(lw=0.3, alpha=0.4)
    ax.set_title("single checkpoints oscillate; stars = last-5 average", fontsize=8)
    fig.savefig(os.path.join(OUT, "ckpt_118k.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------- 5. qualitative sheets
CELL, PAD = 300, 8
D = NUM["dirs"]
BASE_DIR = sorted(glob.glob("phaseN/eval_REF_base_s4cfg7_*"))[-1]
# Column spec: (header, image root, label-subdir or None (= any), steps subdir, eval dir holding the per-prompt scores or None).
# The student evals no longer keep their images, so students were regenerated at the evaluation seeding into
# iclr2027/qual/gen_<bench>/images/<LABEL>/ (iclr2027/qual/gen.lsf); the teacher and base-model evals still hold theirs.
COLS_3K = [("Base model, 4 steps, w=7", BASE_DIR + "/{bench}", None, "s4", None),
           ("Naive CD, 4 steps", "iclr2027/qual/gen_{bench}", "NAIVE3K", "s4", D["naive_s0"]),
           ("Ours, 4 steps", "iclr2027/qual/gen_{bench}", "OURS3K", "s4", D["ours_s0"]),
           ("Teacher, 28 steps, w=7", D["teacher28"] + "/{bench}", None, "s28", D["teacher28"])]
COLS_3K_ABL = [("Naive CD", "iclr2027/qual/gen_{bench}", "NAIVE3K", "s4", D["naive_s0"]),
               ("Argmax selection only", "iclr2027/qual/gen_{bench}", "ARGMAX3K", "s4", D["argmax_s0"]),
               ("Ours (argmax + projector reward)", "iclr2027/qual/gen_{bench}", "OURS3K", "s4", D["ours_s0"]),
               ("Argmax + decode-based reward", "iclr2027/qual/gen_{bench}", "DECODE3K", "s4", sorted(glob.glob("phaseN/eval_S4_CD_dinop_hard-rewXi-avglast3_s0_*"))[-1])]
COLS_118K = [("Base model, 4 steps, w=7", BASE_DIR + "/{bench}", None, "s4", None),
             ("Naive CD, 4 steps (118k)", "iclr2027/qual/gen_{bench}", "NAIVE118K", "s4", D["naive118k_s0"]),
             ("Argmax selection, 4 steps (118k)", "iclr2027/qual/gen_{bench}", "ARGMAX118K", "s4", D["argmax118k_s0"]),
             ("Teacher, 28 steps, w=7", D["teacher28"] + "/{bench}", None, "s28", D["teacher28"])]


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def img_path(root, bench, label, idx, steps):
    c = glob.glob(os.path.join(root.format(bench=bench), "images", label or "*", f"p{idx:05d}", steps, "cand0.png"))
    return c[0] if c else None


_PP = {}


def per_prompt(eval_dir):
    if eval_dir not in _PP:
        r = load_dir(eval_dir)
        _PP[eval_dir] = {"cb": {k[1]: v for k, v in r["cb_prompt"].items()}, "ge2": {k: v["score"] for k, v in r["ge2_prompt"].items()}}
    return _PP[eval_dir]


def sheet(rows, cols, bench, out, title=None, label_w=300, note=None):
    """rows: list of (idx, prompt); cols: list of (header, image root, label, steps, score eval dir)."""
    f_h, f_p, f_s = font(15, True), font(13), font(12)
    head_h = 46 if title is None else 76
    W = label_w + len(cols) * (CELL + PAD) + PAD
    H = head_h + len(rows) * (CELL + PAD + 18) + PAD + (26 if note else 0)
    im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    if title:
        d.text((PAD, 8), title, fill=(30, 30, 30), font=f_h)
    for j, (hdr, _, _, _, _) in enumerate(cols):
        x = label_w + PAD + j * (CELL + PAD)
        for li, line in enumerate(hdr.split("\n")):
            d.text((x, head_h - 40 + li * 17), line, fill=(30, 30, 30), font=f_h if li == 0 else f_p)
    for i, (idx, prompt) in enumerate(rows):
        y = head_h + i * (CELL + PAD + 18)
        for li, line in enumerate(textwrap.wrap(f"“{prompt}”", 34)[:6]):
            d.text((PAD, y + 4 + li * 17), line, fill=(20, 20, 20), font=f_p)
        for j, (hdr, root, label, steps, score_dir) in enumerate(cols):
            x = label_w + PAD + j * (CELL + PAD)
            p = img_path(root, bench, label, idx, steps)
            if p:
                im.paste(Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS), (x, y))
            else:
                d.rectangle([x, y, x + CELL, y + CELL], fill=(230, 230, 230)); d.text((x + 10, y + 10), "missing", fill=(90, 90, 90), font=f_p)
            d.rectangle([x - 1, y - 1, x + CELL, y + CELL], outline=(200, 200, 200), width=1)
            if score_dir:
                sc = per_prompt(score_dir)["cb" if bench == "compbench" else "ge2"].get(prompt)
                if sc is not None:
                    d.text((x, y + CELL + 3), f"score {sc:.2f}", fill=(60, 60, 60), font=f_s)
    if note:
        d.text((PAD, H - 22), note, fill=(90, 90, 90), font=f_s)
    im.save(out, quality=92, optimize=True)
    print("wrote", out, im.size)


def prompts_json(eval_dir, bench):
    return {r["idx"]: r for r in json.load(open(os.path.join(eval_dir, bench, "prompts.json")))}


def candidates():
    """Per-category candidate sheets (top-8 by ours-minus-naive margin) so rows can be picked by eye."""
    q = NUM["qual"]
    for c in CATS:
        rows = [(r["idx"], r["prompt"]) for r in q["compbench"][c]]
        sheet(rows, COLS_3K, "compbench", os.path.join(OUT, f"cand_{c}.jpg"), title=f"candidates: {c} (ours - naive, seed 0 shown)")
    sheet([(r["idx"], r["prompt"]) for r in q["geneval2"][:12]], COLS_3K, "geneval2", os.path.join(OUT, "cand_geneval2_a.jpg"), title="candidates: GenEval2 top 1-12")
    sheet([(r["idx"], r["prompt"]) for r in q["geneval2"][12:24]], COLS_3K, "geneval2", os.path.join(OUT, "cand_geneval2_b.jpg"), title="candidates: GenEval2 top 13-24")
    sheet([(r["idx"], r["prompt"]) for r in q["regress_cb"]], COLS_3K, "compbench", os.path.join(OUT, "cand_regress_cb.jpg"), title="candidates: regressions (naive - ours)")
    sheet([(r["idx"], r["prompt"]) for r in q["regress_ge2"]], COLS_3K, "geneval2", os.path.join(OUT, "cand_regress_ge2.jpg"), title="candidates: GenEval2 regressions")


# Final picks, chosen by eye from the candidate sheets (top-8 ours-minus-naive margin per category), keeping only
# rows where the difference is visible in the image and not just in the evaluator's score; one row per category.
PICKS_CB = [("color", "a white cat and a black whisker"),
            ("shape", "a rectangular laptop and a teardrop laptop bag"),
            ("texture", "The leather wallet and fluffy keychain hang on the metallic hook by the wooden door."),
            ("spatial", "a giraffe next to a television"),
            ("3d_spatial", "a television in front of a woman"),
            ("numeracy", "four cameras and three horses"),
            ("complex", "The shiny silver car zoomed past the old rusty truck.")]
PICKS_GE2 = ["three white turtles", "a penguin on top of a white elephant behind a flamingo", "five cats on top of a suitcase in front of a truck",
             "three yellow monkeys on top of a green clock", "a bagel to the right of four kangaroos"]
PICKS_REGRESS_CB = [("spatial", "a pig near a clock"), ("texture", "a metallic bracelet and a wooden knife")]
PICKS_REGRESS_GE2 = ["seven black dogs", "four wooden backpacks"]


def final_sheets():
    pj = prompts_json(D["ours_s0"], "compbench"); gj = prompts_json(D["ours_s0"], "geneval2")
    cb_by = {(r["category"], r["prompt"]): i for i, r in pj.items()}; ge_by = {r["prompt"]: i for i, r in gj.items()}
    sheet([(cb_by[k], k[1]) for k in PICKS_CB], COLS_3K, "compbench", os.path.join(OUT, "qual_compbench.jpg"))
    sheet([(ge_by[k], k) for k in PICKS_GE2], COLS_3K, "geneval2", os.path.join(OUT, "qual_geneval2.jpg"))
    sheet([(cb_by[k], k[1]) for k in PICKS_REGRESS_CB], COLS_3K, "compbench", os.path.join(OUT, "qual_regress_compbench.jpg"))
    sheet([(ge_by[k], k) for k in PICKS_REGRESS_GE2], COLS_3K, "geneval2", os.path.join(OUT, "qual_regress_geneval2.jpg"))
    # uncurated: the fixed random draw of prompts used when the subsets were regenerated (iclr2027/qual/uncurated.json)
    unc = json.load(open("iclr2027/qual/uncurated.json"))
    sheet([(i, pj[i]["prompt"]) for i in unc["uncurated_compbench"]], COLS_3K, "compbench", os.path.join(OUT, "qual_uncurated_compbench.jpg"))
    sheet([(i, gj[i]["prompt"]) for i in unc["uncurated_geneval2"]], COLS_3K, "geneval2", os.path.join(OUT, "qual_uncurated_geneval2.jpg"))
    json.dump({"uncurated": unc, "picks_cb": [cb_by[k] for k in PICKS_CB], "picks_ge2": [ge_by[k] for k in PICKS_GE2],
               "regress_cb": [cb_by[k] for k in PICKS_REGRESS_CB], "regress_ge2": [ge_by[k] for k in PICKS_REGRESS_GE2]},
              open(os.path.join(OUT, "qual_rows.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidates", action="store_true"); ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if not a.no_plots:
        fig_arms(); fig_category(); fig_monitor(); fig_ckpt()
        print("plots written to", OUT)
    if a.candidates:
        candidates()
    else:
        final_sheets()


if __name__ == "__main__":
    main()
