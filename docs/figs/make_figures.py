#!/usr/bin/env python3
"""Figures for reports/scored_distillation_final.tex. All numbers are read from the evaluation
records under phaseN/ and the candidate caches; the few that come from finished analyses are
inlined with their source named.

    python3 reports/figs/make_figures.py          # writes reports/figs/*.pdf and prints tables
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
    """CompBench vs Kish count of the selection weights, exact vs one-sample estimators, raw final
    checkpoints and the average of the 2k/4k/6k checkpoints (same rule for every arm)."""
    kish = {"S4_CD_dinop_hard": 1.0, "S4_CD_dinop_full-T0.04": 2.19, "S4_CD_dinop_full-T0.08": 2.89,
            "S4_CD_dinop_full-T1e6": 4.0, "S4_CD_dinop_cat-T0.04": 2.19, "S4_CD_dinop_cat-T0.08": 2.89,
            "S4_CD_uniform_visit": 4.0, "S4_CD_dinop_catfreeze-T0.04": 2.19}
    exact = ["S4_CD_dinop_hard", "S4_CD_dinop_full-T0.04", "S4_CD_dinop_full-T0.08", "S4_CD_dinop_full-T1e6"]
    sampled = ["S4_CD_dinop_hard", "S4_CD_dinop_cat-T0.04", "S4_CD_dinop_cat-T0.08", "S4_CD_uniform_visit"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, suf, title in zip(axes, ("", "-avglast3"), ("raw final checkpoint", "average of the 2k, 4k, 6k checkpoints")):
        if "S4_B2" + suf not in s4:
            continue
        naive = seed_stats(s4, "S4_B2" + suf)
        for labs, color, name, mk in ((exact, C_EXACT, "exact weighting (all four candidates)", "o"),
                                      (sampled, C_SAMPLED, "one sampled candidate per visit", "s")):
            pts = [(kish[l], seed_stats(s4, l + suf)) for l in labs if l + suf in s4]
            ax.errorbar([x for x, _ in pts], [st[0] for _, st in pts], yerr=[st[1] for _, st in pts],
                        color=color, marker=mk, ms=4, lw=1.2, capsize=2, label=name)
        fz = "S4_CD_dinop_catfreeze-T0.04" + suf
        if fz in s4:
            st = seed_stats(s4, fz)
            ax.errorbar([kish[fz.replace(suf, "")]], [st[0]], yerr=[st[1]], color="#2ca02c", marker="D", ms=4, lw=0,
                        elinewidth=1.2, capsize=2, label="one draw per caption, kept (frozen)")
        lat = [(1.0, "S4_CD_latent_hard" + suf), (2.19, "S4_CD_latent_full-T0.04" + suf)]
        lat = [(x, seed_stats(s4, l)) for x, l in lat if l in s4]
        if lat:
            ax.errorbar([x for x, _ in lat], [st[0] for _, st in lat], yerr=[st[1] for _, st in lat], color="#9467bd",
                        marker="^", ms=4, lw=1.0, ls=":", capsize=2, label="latent scorer (no decode): argmax, exact T=0.04")
        ax.axhline(naive[0], color=C_NAIVE, ls="--", lw=1, label="random selection (fixed draw)")
        ax.axhspan(naive[0] - naive[1], naive[0] + naive[1], color=C_NAIVE, alpha=0.12, lw=0)
        ax.set_xticks([1, 2.19, 2.89, 4.0])
        ax.set_xticklabels(["argmax\n$N_K$ 1", "T=0.04\n$N_K$ 2.2", "T=0.08\n$N_K$ 2.9", "uniform\n$N_K$ 4"])
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("T2I-CompBench")
    axes[0].legend(fontsize=6.5, loc="lower left", frameon=False)
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
    ax.set_xscale("log"); ax.set_xlabel("temperature T"); ax.set_ylabel("median entropy count $N_H$")
    ax.axvline(0.04, color="k", lw=0.6, ls=":"); ax.axvline(0.08, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("entropy count of the Boltzmann weights", fontsize=9)
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


# ----------------------------------------------------------------------------- arm comparison figures
ARM_ORDER = [  # label prefix, display name, family
    ("S4_B2", "random (fixed draw)", "reference"),
    ("S4_CD_dinop_hard", "argmax, RGB DINO", "reference"),
    ("S4_CD_dinop_full-T0.04", "Boltzmann T=0.04, exact", "soft weighting"),
    ("S4_CD_dinop_full-T0.08", "Boltzmann T=0.08, exact", "soft weighting"),
    ("S4_CD_dinop_full-T1e6", "uniform, exact", "soft weighting"),
    ("S4_CD_dinop_cat-T0.04", "Boltzmann T=0.04, sampled", "sampled estimator"),
    ("S4_CD_dinop_cat-T0.08", "Boltzmann T=0.08, sampled", "sampled estimator"),
    ("S4_CD_uniform_visit", "uniform, redrawn per visit", "sampled estimator"),
    ("S4_CD_dinop_catfreeze-T0.04", "Boltzmann T=0.04, one draw kept", "sampled estimator"),
    ("S4_CD_latent_hard", "argmax, latent scorer", "latent scorer"),
    ("S4_CD_latent_full-T0.04", "Boltzmann T=0.04 exact, latent scorer", "latent scorer"),
    ("S4_CD_dinop_hard-rewF", "argmax + projector reward, frozen", "projector reward"),
    ("S4_CD_dinop_hard-rewR", "argmax + projector reward, refreshed", "projector reward"),
    ("S4_CD_dinop_hard-rewX", "argmax + exact DINO reward", "exact reward"),
]
FAM_COLOR = {"reference": "#333333", "soft weighting": "#1f77b4", "sampled estimator": "#d62728", "latent scorer": "#9467bd", "projector reward": "#2ca02c", "exact reward": "#ff7f0e"}


def fidelity_rows(path="phaseW/fidelity_s4_report.md"):
    out = {}
    if not os.path.isfile(path):
        return out
    for ln in open(path):
        m = re.match(r"\|\s*(S4_\S+?)(-avglast3)?_s(\d)@4\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
        if m:
            out.setdefault(m.group(1) + (m.group(2) or ""), {})[int(m.group(3))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)))
    return out


def fig_all_arms(s4):
    """Every 3k arm on one axis: seed points, mean and sd. Two figures: CompBench (raw, averaged) and
    GenEval2 (raw, averaged) + CMMD (averaged)."""
    fid = fidelity_rows()
    arms = [(k, n, f) for k, n, f in ARM_ORDER if k in s4]
    y = np.arange(len(arms))[::-1]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="D", color=c, lw=0, label=f) for f, c in FAM_COLOR.items()]
    handles += [Line2D([0], [0], color="#333333", ls=":", label="random mean"), Line2D([0], [0], color="#333333", ls="--", label="argmax mean")]
    groups = [("all_arms_compbench", [("CompBench, raw final checkpoint", lambda k: [s4[k][s][0] for s in sorted(s4[k])] if k in s4 else []),
                                      ("CompBench, averaged checkpoints", lambda k: [s4[k + "-avglast3"][s][0] for s in sorted(s4[k + "-avglast3"])] if k + "-avglast3" in s4 else [])]),
              ("all_arms_other", [("GenEval2 x100, raw final", lambda k: [s4[k][s][1] for s in sorted(s4[k])] if k in s4 else []),
                                  ("GenEval2 x100, averaged", lambda k: [s4[k + "-avglast3"][s][1] for s in sorted(s4[k + "-avglast3"])] if k + "-avglast3" in s4 else []),
                                  ("CMMD, averaged (lower is better)", lambda k: [fid[k + "-avglast3"][s][1] for s in sorted(fid[k + "-avglast3"])] if k + "-avglast3" in fid else [])])]
    for fname, panels in groups:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 2.4, 5.4), sharey=True)
        for ax, (title, get) in zip(axes, panels):
            ref = {}
            for yi, (k, n, fam) in zip(y, arms):
                v = get(k)
                if not v:
                    continue
                col = FAM_COLOR[fam]
                ax.scatter(v, [yi] * len(v), s=16, color=col, alpha=0.55, zorder=3)
                ax.errorbar([np.mean(v)], [yi], xerr=[np.std(v, ddof=1)] if len(v) > 1 else None, fmt="D", ms=5, color=col, capsize=3, zorder=4)
                if k in ("S4_B2", "S4_CD_dinop_hard"):
                    ref[k] = np.mean(v)
            for k, ls in (("S4_B2", ":"), ("S4_CD_dinop_hard", "--")):
                if k in ref:
                    ax.axvline(ref[k], color="#333333", lw=0.8, ls=ls)
            ax.set_title(title, fontsize=9); ax.grid(axis="x", lw=0.3, alpha=0.5); ax.tick_params(axis="x", labelsize=8)
        axes[0].set_yticks(y); axes[0].set_yticklabels([n for _, n, _ in arms], fontsize=8)
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.55, -0.06))
        fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, fname + ".png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def fig_contrasts():
    """Forest plot of seed-paired CompBench contrasts against argmax (and random), averaged checkpoints."""
    d = json.load(open("phaseW/s4_seed_stats-avglast3.json")); r = json.load(open("phaseW/s4_seed_stats-raw.json"))
    names = {k: n for k, n, _ in ARM_ORDER}; fams = {k: f for k, _, f in ARM_ORDER}
    rows = []
    for k, n, f in ARM_ORDER:
        if k in ("S4_B2",):
            continue
        key = f"{k}-avglast3 vs S4_CD_dinop_hard-avglast3"; keyr = f"{k} vs S4_CD_dinop_hard"
        if key in d:
            rows.append((n, d[key]["compbench"], r.get(keyr, {}).get("compbench"), f))
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (n, c, cr, f) in zip(y, rows):
        col = FAM_COLOR[f]
        if cr:
            ax.errorbar([cr["mean"]], [yi + 0.18], xerr=[[cr["mean"] - cr["ci95"][0]], [cr["ci95"][1] - cr["mean"]]], fmt="s", ms=3.5, color=col, alpha=0.45, capsize=2, lw=0.9)
        ax.errorbar([c["mean"]], [yi - 0.18], xerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]], fmt="D", ms=4.5, color=col, capsize=3, lw=1.2)
        ax.scatter(c["per_seed"], [yi - 0.18] * len(c["per_seed"]), s=10, color=col, alpha=0.6, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _, _ in rows], fontsize=8)
    ax.set_xlabel("CompBench difference to argmax, seed-paired: mean, per-seed dots, 95% t-interval (n = 3)", fontsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="D", color="#555555", lw=0, label="averaged checkpoints"), Line2D([0], [0], marker="s", color="#555555", alpha=0.45, lw=0, label="raw final checkpoint")], fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.savefig(os.path.join(OUT, "contrasts.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "contrasts.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_raw_vs_avg(s4):
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); legend_lines = []
    for k, n, f in ARM_ORDER:
        if k not in s4 or k + "-avglast3" not in s4:
            continue
        raw = [s4[k][s][0] for s in sorted(s4[k])]; avg = [s4[k + "-avglast3"][s][0] for s in sorted(s4[k + "-avglast3"])]
        col = FAM_COLOR[f]
        ax.scatter(raw, avg, s=12, color=col, alpha=0.5)
        ax.errorbar([np.mean(raw)], [np.mean(avg)], xerr=[np.std(raw, ddof=1)], yerr=[np.std(avg, ddof=1)], fmt="D", ms=4, color=col, capsize=2, lw=0.9)
        idx = [k2 for k2, _, _ in ARM_ORDER].index(k) + 1
        ax.annotate(str(idx), (np.mean(raw), np.mean(avg)), xytext=(4, 3), textcoords="offset points", fontsize=7, color=col, weight="bold")
        legend_lines.append(f"{idx}: {n}")
    lo, hi = 0.43, 0.50
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls=":"); ax.set_xlim(lo, hi); ax.set_ylim(0.465, 0.495)
    ax.set_xlabel("CompBench, raw final checkpoint"); ax.set_ylabel("CompBench, average of the 2k, 4k, 6k checkpoints")
    ax.set_title("averaging lifts every arm and compresses the spread", fontsize=8.5)
    ax.text(1.02, 0.98, "\n".join(legend_lines), transform=ax.transAxes, fontsize=6.5, va="top", ha="left")
    fig.savefig(os.path.join(OUT, "raw_vs_avg.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "raw_vs_avg.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_category_heatmap(s4):
    """CompBench per category, averaged checkpoints, 3-seed means, difference to the random arm."""
    arms = [(k, n) for k, n, _ in ARM_ORDER if k + "-avglast3" in s4]
    base = cat_means(s4, "S4_B2-avglast3")[0]
    M = np.array([[cat_means(s4, k + "-avglast3")[0][c] - base[c] for c in CATS] for k, _ in arms])
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(arms) + 1.2))
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CAT_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([n for _, n in arms], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6, color="white" if abs(M[i, j]) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="CompBench, difference to random (averaged checkpoints, 3-seed means)")
    fig.savefig(os.path.join(OUT, "category_heatmap.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "category_heatmap.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    s4 = eval_records("phaseN/eval_S4_*_s*_*")
    fig_all_arms(s4); fig_contrasts(); fig_raw_vs_avg(s4); fig_category_heatmap(s4)
    print("arm comparison figures written")
