#!/usr/bin/env python3
"""Figures for docs/report.tex, exactly as they were produced. The script reads the evaluation
records and candidate caches of the experimental repository that produced the report (the
`phaseN/eval_*/alignment.json` records and `phaseN/coco_selection_*/selection_rank*.jsonl`
caches referenced below); with this release code the same records are written by
eval/compbench.py + eval/geneval2.py (alignment.json per evaluated model) and
data/build_candidates.py (the cache), so point the globs at `out/eval/eval_<label>_*` and
`cache/*` to regenerate from new runs. The per-run numbers the figures are built from are also
printed to stdout so they can be checked against the tables in the report.

    python3 docs/figs/make_figures.py          # writes docs/figs/*.pdf and prints tables
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "figs")
os.chdir(ROOT)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_NAIVE, C_EXACT, C_SAMPLED, C_DINO, C_VQA = "#7f7f7f", "#1f77b4", "#d62728", "#1f77b4", "#ff7f0e"


def eval_records(pattern):
    """label -> {seed: (compbench_mean, geneval2 x100, per-category dict)}"""
    rows = defaultdict(dict)
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, "alignment.json")
        if not os.path.isfile(f):
            continue
        a = json.load(open(f))
        m = re.match(r"(.*)_s(\d)$", a["label"])
        if not m:
            continue
        cats = {k: (v if not isinstance(v, dict) else v.get("mean", v.get("score"))) for k, v in a["compbench"].items()}
        rows[m.group(1)][int(m.group(2))] = (a["compbench_mean"], a["geneval2"] * 100, cats)
    return rows


def seed_stats(rows, label):
    v = np.array([rows[label][s][0] for s in sorted(rows[label])])
    g = np.array([rows[label][s][1] for s in sorted(rows[label])])
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, g.mean(), g.std(ddof=1) if len(g) > 1 else 0.0, len(v)


# ----------------------------------------------------------------------------- selection rule
def fig_selection_rule(s4):
    """CompBench vs effective sample size of the selection weights, exact vs one-sample."""
    ess = {"S4_CD_dinop_hard": 1.0, "S4_CD_dinop_full-T0.04": 2.54, "S4_CD_dinop_full-T0.08": 3.33,
           "S4_CD_dinop_full-T1e6": 4.0, "S4_CD_dinop_cat-T0.04": 2.54, "S4_CD_dinop_cat-T0.08": 3.33,
           "S4_CD_uniform_visit": 4.0}
    exact = ["S4_CD_dinop_hard", "S4_CD_dinop_full-T0.04", "S4_CD_dinop_full-T0.08", "S4_CD_dinop_full-T1e6"]
    sampled = ["S4_CD_dinop_hard", "S4_CD_dinop_cat-T0.04", "S4_CD_dinop_cat-T0.08", "S4_CD_uniform_visit"]
    naive = seed_stats(s4, "S4_B2")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, col in zip(axes, (0, 2)):
        for labs, color, name, mk in ((exact, C_EXACT, "exact weighting (all four candidates)", "o"),
                                      (sampled, C_SAMPLED, "one sampled candidate per visit", "s")):
            xs = [ess[l] for l in labs]
            st = [seed_stats(s4, l) for l in labs]
            ys = [s[col] for s in st]; es = [s[col + 1] for s in st]
            ax.errorbar(xs, ys, yerr=es, color=color, marker=mk, ms=4, lw=1.2, capsize=2, label=name)
        ax.axhline(naive[col], color=C_NAIVE, ls="--", lw=1, label="random selection (fixed draw)")
        ax.axhspan(naive[col] - naive[col + 1], naive[col] + naive[col + 1], color=C_NAIVE, alpha=0.12, lw=0)
        ax.set_xticks([1, 2.54, 3.33, 4.0])
        ax.set_xticklabels(["argmax\nESS 1", "T=0.04\nESS 2.5", "T=0.08\nESS 3.3", "uniform\nESS 4"])
        ax.set_xlabel("selection distribution over the four candidates")
    axes[0].set_ylabel("T2I-CompBench")
    axes[1].set_ylabel("GenEval2 ($\\times$100)")
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    fig.savefig(os.path.join(OUT, "selection_rule.pdf")); fig.savefig(os.path.join(OUT, "selection_rule.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- score statistics
def cache_scores(pattern):
    S = []
    for f in sorted(glob.glob(pattern)):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                if "dino_patch_cos" in r:
                    S.append(r["dino_patch_cos"])
    return np.asarray(S, float)


def entropy_norm(S, T):
    z = (S - S.max(1, keepdims=True)) / T
    W = np.exp(z); W /= W.sum(1, keepdims=True)
    return -(W * np.log(np.clip(W, 1e-300, None))).sum(1) / np.log(S.shape[1]), W


def fig_score_entropy(caches):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    fig.subplots_adjust(wspace=0.32)
    S = caches["8 steps, w=7 (118k)"]
    ax = axes[0]
    for T, color in ((0.04, C_EXACT), (0.08, C_SAMPLED), (0.16, "#2ca02c")):
        H, _ = entropy_norm(S, T)
        ax.hist(H, bins=40, range=(0, 1), histtype="step", lw=1.3, color=color, density=True,
                label=f"T={T}: mean {H.mean():.2f}, ESS {np.exp(H * np.log(4)).mean():.2f}")
    ax.set_xlabel("normalised selection entropy per caption\n(0: one candidate, 1: all four alike)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("selection entropy, 118k pool", fontsize=9)

    ax = axes[1]
    for (name, Sc), color in zip(caches.items(), (C_EXACT, "#9467bd", C_SAMPLED, "#2ca02c")):
        if Sc.shape[1] != 4:
            continue
        spread = Sc.std(1)
        ax.hist(spread, bins=40, range=(0, 0.3), histtype="step", lw=1.3, color=color, density=True,
                label=f"{name}: mean {spread.mean():.3f}")
    ax.set_xlabel("within-caption std of the DINO patch score")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("candidate spread by teacher", fontsize=9)

    ax = axes[2]
    Ts = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    for (name, Sc), color in zip(caches.items(), (C_EXACT, "#9467bd", C_SAMPLED, "#2ca02c")):
        n = Sc.shape[1]
        ess = [np.median(np.exp(entropy_norm(Sc, T)[0] * np.log(n))) for T in Ts]
        ax.plot(Ts, ess, marker="o", ms=3, lw=1.2, color=color, label=name)
    ax.set_xscale("log"); ax.set_xlabel("temperature T"); ax.set_ylabel("median effective sample size")
    ax.axvline(0.04, color="k", lw=0.6, ls=":"); ax.axvline(0.08, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("ESS of the Boltzmann weights", fontsize=9)
    fig.savefig(os.path.join(OUT, "score_entropy.pdf")); fig.savefig(os.path.join(OUT, "score_entropy.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- checkpoint curve
def fig_checkpoint_curve(w):
    steps = [5000, 10000, 15000, 20000, 30000, 40000, 50000, "final"]
    arms = (("W_B2_118k", "random selection", C_NAIVE), ("W_CD_dinop_hard_118k", "DINOv2 patches", C_DINO),
            ("W_B4_118k", "VQAScore", C_VQA))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for base, name, color in arms:
        xs, ys = [], []
        for s in steps:
            lab = base if s == "final" else f"{base}-step{s}"
            if lab in w and 0 in w[lab]:
                xs.append(56974 if s == "final" else s); ys.append(w[lab][0][0])
        ax.plot(xs, ys, marker="o", ms=3, lw=1.1, color=color, label=name)
        avg = w[f"{base}-avglast5"][0][0]
        ax.plot([56974], [avg], marker="*", ms=10, color=color, ls="none")
        ax.annotate(f"avg {avg:.4f}", (56974, avg), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=color, va="center")
    ax.set_xlabel("optimizer step (118k pool, seed 0)"); ax.set_ylabel("T2I-CompBench")
    ax.set_xlim(0, 68000)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("single checkpoints oscillate; stars = average of the last five", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "checkpoint_curve.pdf")); fig.savefig(os.path.join(OUT, "checkpoint_curve.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- per-category gap
CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
CAT_NAMES = ["color", "shape", "texture", "2D-spatial", "3D-spatial", "numeracy", "non-spatial", "complex"]


def cat_means(rows, label):
    seeds = sorted(rows[label])
    return {c: np.mean([rows[label][s][2][c] for s in seeds]) for c in CATS}, len(seeds)


def fig_per_category(w, ten):
    n1, _ = cat_means(w, "W_B2_118k-avglast5"); d1, _ = cat_means(w, "W_CD_dinop_hard_118k-avglast5")
    n10, _ = cat_means(ten, "W_B2_118k-avglast5"); d10, _ = cat_means(ten, "W_CD_dinop_hard_118k-avglast5")
    x = np.arange(len(CATS)); wd = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - wd / 2, [d1[c] - n1[c] for c in CATS], wd, color=C_DINO, label="one image per prompt")
    ax.bar(x + wd / 2, [d10[c] - n10[c] for c in CATS], wd, color="#aec7e8", label="official protocol, ten images per prompt")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(CAT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("CompBench gain\n(DINOv2 patches $-$ random)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("118k pool, weight-averaged students, mean over three training seeds", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "per_category_gap.pdf")); fig.savefig(os.path.join(OUT, "per_category_gap.png"), dpi=110)
    plt.close(fig)
    print("\n%% per-category, 118k, 3-seed means: 1-image | 10-image")
    for c, nm in zip(CATS, CAT_NAMES):
        print(f"{nm} & {n1[c]:.4f} & {d1[c]:.4f} & {d1[c]-n1[c]:+.4f} & {n10[c]:.4f} & {d10[c]:.4f} & {d10[c]-n10[c]:+.4f} \\\\")
    print(f"mean & {np.mean(list(n1.values())):.4f} & {np.mean(list(d1.values())):.4f} & {np.mean([d1[c]-n1[c] for c in CATS]):+.4f} & "
          f"{np.mean(list(n10.values())):.4f} & {np.mean(list(d10.values())):.4f} & {np.mean([d10[c]-n10[c] for c in CATS]):+.4f} \\\\")


def main():
    os.makedirs(OUT, exist_ok=True)
    w = eval_records("phaseN/eval_W_*_118k*_s*_*")
    ten = eval_records("phaseN/eval10_W_*_118k-avglast5_s*_*")
    s4 = eval_records("phaseN/eval_S4_*_s*_*")
    print("%% S4 arms (3k pool): label, CompBench mean+-sd, GenEval2 mean+-sd, seeds")
    for lab in sorted(s4):
        m, sd, g, gsd, n = seed_stats(s4, lab)
        print(f"{lab:32s} {m:.4f} +- {sd:.4f}   {g:.2f} +- {gsd:.2f}   n={n}")
    print("%% 118k arms")
    for lab in sorted(w):
        if "step" in lab:
            continue
        m, sd, g, gsd, n = seed_stats(w, lab)
        print(f"{lab:36s} {m:.4f} +- {sd:.4f}   {g:.2f} +- {gsd:.2f}   n={n}")
    print("%% official 10-image")
    for lab in sorted(ten):
        m, sd, g, gsd, n = seed_stats(ten, lab)
        print(f"{lab:36s} {m:.4f} +- {sd:.4f}   n={n}")
    fig_selection_rule(s4)
    fig_checkpoint_curve(w)
    fig_per_category(w, ten)
    caches = {"8 steps, w=7 (118k)": cache_scores("phaseN/coco_selection_118k/selection_rank*.jsonl"),
              "8 steps, w=7 (3k)": cache_scores("phaseN/coco_selection_dinopatch/selection_rank*.jsonl"),
              "16 steps, w=4.5 (3k)": cache_scores("phaseN/coco_selection_k16/selection_rank*.jsonl"),
              "28 steps, w=7 (3k)": cache_scores("phaseN/coco_selection_k28/selection_rank*.jsonl")}
    fig_score_entropy(caches)
    print("%% cache score statistics")
    for name, S in caches.items():
        H4, W4 = entropy_norm(S, 0.04); H8, _ = entropy_norm(S, 0.08)
        srt = np.sort(S, 1)
        print(f"{name:24s} n={len(S):6d} within-std {S.std(1).mean():.4f} range {(srt[:,-1]-srt[:,0]).mean():.4f} "
              f"top2-margin {(srt[:,-1]-srt[:,-2]).mean():.4f} | T=0.04: H {H4.mean():.3f} ESS {np.median(np.exp(H4*np.log(4))):.2f} "
              f"p(argmax) {W4.max(1).mean():.3f} | T=0.08: H {H8.mean():.3f} ESS {np.median(np.exp(H8*np.log(4))):.2f}")
    for f in ("fig_compbench.png", "fig_geneval2.png"):
        src = os.path.join("phaseW", "qual118k", f)
        if os.path.isfile(src):
            import shutil
            from PIL import Image; im = Image.open(src).convert("RGB"); im.resize((1500, round(im.height * 1500 / im.width)), Image.LANCZOS).save(os.path.join(OUT, "qual_" + f.replace("fig_", "").replace(".png", ".jpg")), quality=90, optimize=True)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
