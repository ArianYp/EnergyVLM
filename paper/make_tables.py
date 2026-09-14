#!/usr/bin/env python3
"""Emit every results table of iclr2027/iclr2027_conference.tex as LaTeX from iclr2027/numbers.json
(itself computed from the raw evaluation records by verify_numbers.py), so no number in the paper
is typed by hand. Every comparison is against naive CD (random trajectory, no reward).

    python3 iclr2027/make_tables.py        # writes iclr2027/tables/*.tex
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = os.path.join(ROOT, "iclr2027", "tables")
os.makedirs(OUT, exist_ok=True)
N = json.load(open("iclr2027/numbers.json"))

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
CAT_NAMES = {"color": "Color", "shape": "Shape", "texture": "Texture", "spatial": "2D-spatial", "3d_spatial": "3D-spatial",
             "numeracy": "Numeracy", "non_spatial": "Non-spatial", "complex": "Complex"}
SKILLS = ["object", "count", "attribute", "position", "verb"]

m, c3, g, fid, mon = N["main3k"], N["cat3k"], N["ge2_3k"], N["fid"], N.get("monitor", {})
K_NAIVE, K_OURS = "S4_B2-avglast3", "S4_CD_dinop_hard-rewRi-s16-avglast3"
K_F, K_R, K_E25 = "S4_CD_dinop_hard-rewF-avglast3", "S4_CD_dinop_hard-rewR-avglast3", "S4_CD_dinop_hard-rewRi-e25-avglast3"
K_MAX = "S4_CD_dinop_hard-rewRi-s16-maxpool-avglast3"
S3 = [0, 1, 2]


def p(x):
    if x != x:
        return "--"
    return "$<0.001$" if x < 0.001 else (f"{x:.3f}" if x < 0.01 else f"{x:.2f}")


def pm(mean, sd, nd=4):
    return f"${mean:.{nd}f}\\pm{sd:.{nd}f}$" if sd == sd else f"${mean:.{nd}f}$"


def d(c, nd=4, show_p=True):
    s = f"${c['mean']:+.{nd}f}\\pm{c['sd']:.{nd}f}$"
    return s + (f" ({p(c['t_p'])})" if show_p else "")


def sub3(st):
    """restrict a seed_stats record to seeds 0-2 (the seeds our arm has), for like-for-like columns"""
    v = np.array([x for s, x in zip(st["seeds"], st["values"]) if s in S3])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist(), "seeds": [s for s in st["seeds"] if s in S3]}


def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)
    print("wrote", os.path.join(OUT, name))


# ------------------------------------------------------------------ main table (3k): naive vs ours
naive3, ours3 = sub3(m[K_NAIVE]["cb"]), m[K_OURS]["cb"]
naive3g, ours3g = sub3(m[K_NAIVE]["ge2"]), m[K_OURS]["ge2"]
rows = [
    f"Naive CD (random trajectory, no reward) & {pm(naive3['mean'], naive3['sd'])} & -- & {pm(naive3g['mean'], naive3g['sd'], 2)} & -- \\\\",
    f"\\textbf{{Ours}} (scored trajectory + projector reward) & $\\mathbf{{{ours3['mean']:.4f}}}\\pm{ours3['sd']:.4f}$ & "
    f"{d(m[K_OURS]['vs_naive_cb_s012'])} & $\\mathbf{{{ours3g['mean']:.2f}}}\\pm{ours3g['sd']:.2f}$ & {d(m[K_OURS]['vs_naive_ge2_s012'], 2)} \\\\",
]
R = N["refs"]
ref_rows = [
    f"Base model, 4 steps, $w{{=}}7$ (no distillation) & ${R['REF_base_s4cfg7']['cb_mean']:.4f}$ & -- & ${R['REF_base_s4cfg7']['ge2']:.2f}$ & -- \\\\",
    f"Teacher, 8 steps, $w{{=}}7$ (the distillation source) & ${R['teacher_s8cfg7']['cb_mean']:.4f}$ & -- & ${R['teacher_s8cfg7']['ge2']:.2f}$ & -- \\\\",
    f"Teacher, 28 steps, $w{{=}}7$ & ${R['REF_teacher_s28cfg7']['cb_mean']:.4f}$ & -- & ${R['REF_teacher_s28cfg7']['ge2']:.2f}$ & -- \\\\",
]
tab = r"""\begin{tabular}{l c c c c}
\toprule
Model & CompBench & $\Delta$ vs.\ naive ($p$) & GenEval2 ($\times100$) & $\Delta$ vs.\ naive ($p$) \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + "\n".join(ref_rows) + r"""
\bottomrule
\end{tabular}"""
write("tab_main.tex", tab)
print("  per-seed CompBench naive", naive3["values"], "ours", ours3["values"], "| per-seed diff", m[K_OURS]["vs_naive_cb_s012"]["per_seed"])

# ------------------------------------------------------------------ per-category (3k)
rows = []
for c in CATS:
    r = c3[c]
    rows.append(f"{CAT_NAMES[c]} & {pm(r['naive']['mean'], r['naive']['sd'])} & {pm(r['ours']['mean'], r['ours']['sd'])} & {d(r['ours_vs_naive'])} \\\\")
mean_row = f"\\textbf{{Mean}} & {pm(naive3['mean'], naive3['sd'])} & {pm(ours3['mean'], ours3['sd'])} & {d(m[K_OURS]['vs_naive_cb_s012'])} \\\\"
tab = r"""\begin{tabular}{l c c c}
\toprule
Category & Naive CD & Ours & Ours $-$ naive ($p$) \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + mean_row + r"""
\bottomrule
\end{tabular}"""
write("tab_cat.tex", tab)

# ------------------------------------------------------------------ projector-reward ablation (3k), all vs naive


def cmmd3(lab):
    r = fid.get(lab)
    if not r:
        return "--"
    v = [x for s, x in zip(r["seeds"], r["cmmd"]["per_seed"]) if s in S3]
    return f"${np.mean(v):.3f}$"


def corr_end(arm):
    if arm not in mon:
        return "--"
    return "--"


abl = [
    (K_NAIVE, "Naive CD (no selection, no reward)", "S4_B2-avglast3"),
    (K_F, "Projector frozen", "S4_CD_dinop_hard-rewF-avglast3"),
    (K_R, "Refresh every 100 updates, 4 steps", "S4_CD_dinop_hard-rewR-avglast3"),
    (K_E25, "Refresh every 25 updates, 4 steps", "S4_CD_dinop_hard-rewRi-e25-avglast3"),
    (K_OURS, "\\textbf{Refresh every 100 updates, 16 steps (ours)}", "S4_CD_dinop_hard-rewRi-s16-avglast3"),
    (K_MAX, "Ours, max-pooled DINOv2 patches", None),
]
rows = []
for k, name, flab in abl:
    r = m[k]
    cb = sub3(r["cb"]) if k == K_NAIVE else r["cb"]; ge = sub3(r["ge2"]) if k == K_NAIVE else r["ge2"]
    dvn = "--" if k == K_NAIVE else d(r["vs_naive_cb_s012"]); dgn = "--" if k == K_NAIVE else d(r["vs_naive_ge2_s012"], 2)
    rows.append(f"{name} & {pm(cb['mean'], cb['sd'])} & {dvn} & {pm(ge['mean'], ge['sd'], 1)} & {dgn} & {cmmd3(flab) if flab else '--'} \\\\")
tab = r"""\begin{tabular}{l c c c c c}
\toprule
Reward variant & CompBench & $\Delta$ vs.\ naive ($p$) & GenEval2 & $\Delta$ vs.\ naive ($p$) & CMMD$\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
write("tab_ablation.tex", tab)

# ------------------------------------------------------------------ GenEval2 breakdown (3k)
lines = [("Overall (official; per-prompt geometric mean)", "overall", 1), ("Collapsed prompts (\\%; score $<0.05$)", "collapsed", 1)]
lines += [(f"Skill: {s} (per atom)", f"skill:{s}", 1) for s in SKILLS]
lines += [(f"Atoms $={n}$ (per atom)", f"atoms={n} (atom)", 1) for n in range(3, 11)]
rows = []
for name, key, nd in lines:
    r = g[key]
    rows.append(f"{name} & {pm(r['naive']['mean'], r['naive']['sd'], nd)} & {pm(r['ours']['mean'], r['ours']['sd'], nd)} & {d(r['ours_vs_naive'], nd)} \\\\")
    if key in ("collapsed", "skill:verb"):
        rows.append("\\midrule")
tab = r"""\begin{tabular}{l c c c}
\toprule
GenEval2 measure ($\times100$), 3k, seeds 0--2 & Naive CD & Ours & Ours $-$ naive ($p$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
write("tab_ge2_3k.tex", tab)

# ------------------------------------------------------------------ fidelity 3k (seeds 0-2): naive vs ours


def fid3(lab):
    r = fid[lab]
    sel = [i for i, s in enumerate(r["seeds"]) if s in S3]
    out = {}
    for q in ("cmmd", "prec", "rec", "fid"):
        v = np.array([r[q]["per_seed"][i] for i in sel]); out[q] = (v.mean(), v.std(ddof=1), v.tolist())
    return out


fn, fo = fid3(K_NAIVE), fid3(K_OURS)
rows = [f"Naive CD & {pm(*fn['cmmd'][:2], 3)} & {pm(*fn['prec'][:2], 3)} & {pm(*fn['rec'][:2], 3)} & {pm(*fn['fid'][:2], 2)} \\\\",
        f"\\textbf{{Ours}} & {pm(*fo['cmmd'][:2], 3)} & {pm(*fo['prec'][:2], 3)} & {pm(*fo['rec'][:2], 3)} & {pm(*fo['fid'][:2], 2)} \\\\"]
con = fid["contrasts"]
crow = "Ours $-$ naive (seed-paired, $p$) & " + " & ".join(d(con[f"{K_OURS} vs {K_NAIVE}:{q}"], 3 if q != "fid" else 2) for q in ("cmmd", "prec", "rec", "fid")) + " \\\\"
tab = r"""\begin{tabular}{l c c c c}
\toprule
 & CMMD $\downarrow$ & Precision $\uparrow$ & Recall $\uparrow$ & FID $\downarrow$ \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + crow + r"""
\bottomrule
\end{tabular}"""
write("tab_fid3k.tex", tab)

# ------------------------------------------------------------------ 118k: naive vs ours (old schedule, 3 seeds, last-5 average)
S = N["scale"]; kn, ko = "W_B2_118k-avglast5", "W_CD_dinop_hard-rewRi-s16_118k-avglast5"
if ko in S:
    raw = N.get("scale_raw", {})
    rows = [
        f"Naive CD (random trajectory, no reward) & {pm(S[kn]['cb']['mean'], S[kn]['cb']['sd'])} & -- & -- & {pm(S[kn]['ge2']['mean'], S[kn]['ge2']['sd'], 2)} & -- & -- \\\\",
        f"\\textbf{{Ours}} (scored trajectory + projector reward) & $\\mathbf{{{S[ko]['cb']['mean']:.4f}}}\\pm{S[ko]['cb']['sd']:.4f}$ & {d(S[ko]['vs_naive_cb_seed'])} & {p(S[ko]['vs_naive_cb_prompt_seedavg']['t_p'])} & "
        f"$\\mathbf{{{S[ko]['ge2']['mean']:.2f}}}\\pm{S[ko]['ge2']['sd']:.2f}$ & {d(S[ko]['vs_naive_ge2_seed'], 2)} & {p(S[ko]['vs_naive_ge2_prompt_seedavg']['t_p'])} \\\\",
    ]
    rawnote = ""
    if "W_B2_118k" in raw and "W_CD_dinop_hard-rewRi-s16_118k" in raw:
        rn, ro = raw["W_B2_118k"]["cb"], raw["W_CD_dinop_hard-rewRi-s16_118k"]["cb"]
        rawnote = (r"\multicolumn{7}{l}{\footnotesize Raw final checkpoints (before averaging), CompBench per seed: naive "
                   + " / ".join(f"{x:.4f}" for x in rn["values"]) + "; ours " + " / ".join(f"{x:.4f}" for x in ro["values"]) + r"} \\" + "\n")
    tab = r"""\begin{tabular}{l c c c c c c}
\toprule
 & \multicolumn{3}{c}{CompBench} & \multicolumn{3}{c}{GenEval2 ($\times100$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Recipe (118k captions, 3 seeds, last-5 average) & mean $\pm$ sd & $\Delta$ vs.\ naive, seed-paired ($p$) & prompt-level $p$ & mean $\pm$ sd & $\Delta$ vs.\ naive ($p$) & prompt-level $p$ \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + rawnote + r"""\bottomrule
\end{tabular}"""
    write("tab_scale.tex", tab)
    rows = []
    for c in CATS:
        n_, o_, dd = S[kn]["cb_cat"][c], S[ko]["cb_cat"][c], S[ko]["cb_cat_vs_naive"][c]
        rows.append(f"{CAT_NAMES[c]} & {pm(n_['mean'], n_['sd'])} & {pm(o_['mean'], o_['sd'])} & {d(dd)} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Mean}} & {pm(S[kn]['cb']['mean'], S[kn]['cb']['sd'])} & {pm(S[ko]['cb']['mean'], S[ko]['cb']['sd'])} & {d(S[ko]['vs_naive_cb_seed'])} \\\\")
    tab = r"""\begin{tabular}{l c c c}
\toprule
Category (118k, 3 seeds, last-5 average) & Naive CD & Ours & Ours $-$ naive ($p$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write("tab_scale_cat.tex", tab)

# ------------------------------------------------------------------ fidelity 118k (per seed): naive vs ours
KN118, KO118 = "W_B2_118k-avglast5", "W_CD_dinop_hard-rewRi-s16_118k-avglast5"
if KO118 in fid and KN118 in fid:
    rows = []
    for lab, name in [(KN118, "Naive CD"), (KO118, "\\textbf{Ours}")]:
        r = fid[lab]
        rows.append(f"{name} & " + " & ".join("/".join(f"{x:.{2 if q in ('fid', 'cmmd') else 3}f}" for x in r[q]["per_seed"]) for q in ("cmmd", "prec", "rec", "fid")) + " \\\\")
    c118 = fid["contrasts"]
    crow = "Ours $-$ naive (seed-paired, $p$) & " + " & ".join(d(c118[f"{KO118} vs {KN118}:{q}"], 3 if q != "fid" else 2) for q in ("cmmd", "prec", "rec", "fid")) + " \\\\"
    tab = r"""\begin{tabular}{l c c c c}
\toprule
118k, seeds 0 / 1 / 2, last-5 average & CMMD $\downarrow$ & Precision $\uparrow$ & Recall $\uparrow$ & FID $\downarrow$ \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + crow + r"""
\bottomrule
\end{tabular}"""
    write("tab_fid118k.tex", tab)

# ------------------------------------------------------------------ numbers used inline in the text
inline = {"ours_vs_naive_cb": m[K_OURS]["vs_naive_cb_s012"], "ours_vs_naive_ge2": m[K_OURS]["vs_naive_ge2_s012"],
          "naive3": naive3, "ours3": ours3, "naive3g": naive3g, "ours3g": ours3g,
          "cat_ours_vs_naive": {c: c3[c]["ours_vs_naive"] for c in CATS},
          "abl_vs_naive": {k: m[k]["vs_naive_cb_s012"] for k in (K_F, K_R, K_E25, K_OURS, K_MAX)},
          "ge2_rows": {key: g[key]["ours_vs_naive"] for _, key, _ in lines},
          "fid_contrasts": {q: con[f"{K_OURS} vs {K_NAIVE}:{q}"] for q in ("cmmd", "prec", "rec", "fid")},
          "monitor_ours": mon.get("rewRi-s16"), "monitor_frozen": mon.get("rewF"),
          "refs": N["refs"]}
json.dump(inline, open(os.path.join(OUT, "inline.json"), "w"), indent=1)
for k in ("ours_vs_naive_cb", "ours_vs_naive_ge2"):
    print(" ", k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in inline[k].items()})
print("  category ours-naive:", {c: (round(v["mean"], 4), round(v["t_p"], 3)) for c, v in inline["cat_ours_vs_naive"].items()})
print("  fidelity contrasts:", {q: (round(v["mean"], 3), round(v["t_p"], 3)) for q, v in inline["fid_contrasts"].items()})
