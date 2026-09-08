#!/usr/bin/env python3
"""Absolute (not delta) GenEval2 and T2I-CompBench scores per model, averaged over seeds.

GenEval2 here is the Soft-TIFA geometric-mean evaluator (Qwen3-VL): a prompt is decomposed into
atoms, each atom is tagged with a skill (object / count / attribute / position / verb), the prompt
score is the geometric mean of its atom scores, and the official number is the mean prompt score
x100. The per-skill columns are the mean ATOM score for that skill; the per-atom-count columns are
the mean PROMPT score in each complexity bucket (100 prompts each, 3..10 atoms).

    python3 phaseW/absolute_scores.py --out phaseW/absolute_scores.json
"""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = [  # (display name, eval-dir glob; one dir per seed)
    ("SD3.5-M teacher, 28 steps, cfg 7", "phaseN/eval_REF_teacher_s28cfg7_*"),
    ("SD3.5-M teacher, 8 steps, cfg 7 (distillation source)", "phaseN/eval_teacher_s8cfg7_*"),
    ("SD3.5-M base, 4 steps, cfg 7", "phaseN/eval_REF_base_s4cfg7_*"),
    ("naive CD (random pick)", "phaseN/eval_T_B2_s[0-9]_*"),
    ("scored CD, DINO CLS", "phaseN/eval_V_CD_dinocls_hard_s[0-9]_*"),
    ("scored CD, DINO mean-pooled patches", "phaseN/eval_V_CD_dinop_hard_s[0-9]_*"),
    ("scored CD, VQAScore", "phaseN/eval_T_B4_s[0-9]_*"),
]
SKILLS = ["object", "count", "attribute", "position", "verb"]
CB = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]


def load_dir(d: Path) -> dict:
    al = json.load(open(d / "alignment.json"))
    sc = json.load(open(next(iter(glob.glob(str(d / "geneval2_scores/*/scores.json"))))))
    assert sc["method"] == "soft_tifa_gm" and sc["n"] == 800, (d, sc["method"], sc["n"])
    skill_atoms = defaultdict(list); bucket = defaultdict(list); arith = []; dead = []
    for r in sc["per_prompt"]:
        bucket[int(r["atom_count"])].append(r["score"])
        arith.append(float(np.mean(r["atom_scores"]))); dead.append(r["score"] < 0.05)
        for s, a in zip(r["skills"], r["atom_scores"]):
            skill_atoms[s].append(a)
    return {"steps": al["steps"], "cfg": al["cfg"], "geneval2": sc["mean"],
            "arith": float(np.mean(arith)), "dead": float(np.mean(dead)),
            "skill": {s: float(np.mean(skill_atoms[s])) for s in SKILLS},
            "atoms": {k: float(np.mean(v)) for k, v in bucket.items()},
            "compbench": al["compbench"], "compbench_mean": al["compbench_mean"]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="phaseW/absolute_scores.json")
    ap.add_argument("--tex", default=None, help="also write the three tables as LaTeX (booktabs) to this path")
    args = ap.parse_args()
    res = {}
    for name, pat in MODELS:
        dirs = sorted(Path(p) for p in glob.glob(pat)); assert dirs, pat
        per = [load_dir(d) for d in dirs]
        agg = lambda f: (float(np.mean([f(p) for p in per])), float(np.std([f(p) for p in per], ddof=1)) if len(per) > 1 else 0.0)
        res[name] = {"n_seeds": len(per), "dirs": [str(d) for d in dirs], "steps": per[0]["steps"], "cfg": per[0]["cfg"],
                     "geneval2": agg(lambda p: p["geneval2"]), "arith": agg(lambda p: p["arith"]), "dead": agg(lambda p: p["dead"]),
                     "skill": {s: agg(lambda p, s=s: p["skill"][s]) for s in SKILLS},
                     "atoms": {k: agg(lambda p, k=k: p["atoms"][k]) for k in range(3, 11)},
                     "compbench_mean": agg(lambda p: p["compbench_mean"]),
                     "compbench": {c: agg(lambda p, c=c: p["compbench"][c]) for c in CB}}
    json.dump(res, open(args.out, "w"), indent=1)

    def row(name, cells): print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n### GenEval2 (Soft-TIFA gmean, Qwen3-VL), x100; skills = mean atom score, mean +- seed std\n")
    row("model", ["steps/cfg", "seeds", "**overall (gmean)**"] + SKILLS + ["atom arith. mean", "% prompts gmean<0.05"])
    row("---", ["---"] * (5 + len(SKILLS)))
    for name, r in res.items():
        row(name, [f"{r['steps']}/{r['cfg']:g}", str(r["n_seeds"]), f"**{100*r['geneval2'][0]:.2f}** +- {100*r['geneval2'][1]:.2f}"]
            + [f"{100*r['skill'][s][0]:.1f} +- {100*r['skill'][s][1]:.1f}" for s in SKILLS]
            + [f"{100*r['arith'][0]:.1f}", f"{100*r['dead'][0]:.1f}"])
    print("\n### GenEval2 by prompt complexity (atoms per prompt, 100 prompts each), x100\n")
    row("model", [str(k) for k in range(3, 11)]); row("---", ["---"] * 8)
    for name, r in res.items():
        row(name, [f"{100*r['atoms'][k][0]:.1f}" for k in range(3, 11)])
    print("\n### T2I-CompBench (official 8 categories), mean +- seed std\n")
    row("model", ["**mean**"] + CB); row("---", ["---"] * (1 + len(CB)))
    for name, r in res.items():
        row(name, [f"**{r['compbench_mean'][0]:.4f}** +- {r['compbench_mean'][1]:.4f}"] + [f"{r['compbench'][c][0]:.4f}" for c in CB])
    print(f"\nwrote {args.out}")
    if args.tex:
        Path(args.tex).write_text(to_latex(res)); print(f"wrote {args.tex}")


SHORT = {"SD3.5-M teacher, 28 steps, cfg 7": "Teacher, 28 steps, CFG 7",
         "SD3.5-M teacher, 8 steps, cfg 7 (distillation source)": "Teacher, 8 steps, CFG 7 (source)",
         "SD3.5-M base, 4 steps, cfg 7": "Base, 4 steps, CFG 7",
         "naive CD (random pick)": "Naive CD (random pick)",
         "scored CD, DINO CLS": "Scored CD, DINO CLS",
         "scored CD, DINO mean-pooled patches": "Scored CD, DINO patches",
         "scored CD, VQAScore": "Scored CD, VQAScore",
         "scored CD, teacher drift (reference-free)": "Scored CD, teacher drift"}


def to_latex(res: dict) -> str:
    """Three booktabs tables. Bold = best among the 4-step distilled arms in that column."""
    distilled = [n for n in res if n.startswith("naive") or n.startswith("scored")]

    def cell(v, best, fmt):
        s = fmt.format(v)
        return f"\\textbf{{{s}}}" if best else s

    def col_best(getter):
        return max(distilled, key=lambda n: getter(res[n]))

    out = []
    # 1. GenEval2 per skill
    skills = ["object", "count", "attribute", "position", "verb"]
    best_overall = col_best(lambda r: r["geneval2"][0]); best_skill = {s: col_best(lambda r, s=s: r["skill"][s][0]) for s in skills}
    best_arith = col_best(lambda r: r["arith"][0]); best_dead = min(distilled, key=lambda n: res[n]["dead"][0])
    out.append(r"""\begin{table}[ht]
\centering
\caption{GenEval2 (Soft-TIFA, Qwen3-VL judge), $\times 100$, on the sealed 800-prompt pool. \emph{Overall} is the official score, the mean over prompts of the geometric mean of a prompt's atom scores ($\pm$ standard deviation over training seeds). Skill columns are the mean score of the atoms tagged with that skill. \emph{Atom mean} is the arithmetic mean over all atoms. \emph{Collapsed} is the share of prompts whose geometric mean falls below 0.05. Teacher and base rows are single deterministic runs. Bold marks the best 4-step distilled arm in each column.}
\label{tab:geneval2_skills}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c ccccc c c}
\toprule
Model & Steps / CFG & Seeds & Overall & Object & Count & Attribute & Position & Verb & Atom mean & Collapsed (\%) \\
\midrule""")
    for n, r in res.items():
        d = n in distilled
        ov = f"{100*r['geneval2'][0]:.2f}" + (f" $\\pm$ {100*r['geneval2'][1]:.2f}" if r["n_seeds"] > 1 else "")
        cells = [SHORT[n], f"{r['steps']} / {r['cfg']:g}", str(r["n_seeds"]), cell(ov, d and n == best_overall, "{}")]
        cells += [cell(100*r["skill"][s][0], d and n == best_skill[s], "{:.1f}") for s in skills]
        cells += [cell(100*r["arith"][0], d and n == best_arith, "{:.1f}"), cell(100*r["dead"][0], d and n == best_dead, "{:.1f}")]
        out.append(" & ".join(cells) + r" \\")
        if n.startswith("SD3.5-M base"): out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")
    # 2. GenEval2 by atom count
    best_k = {k: col_best(lambda r, k=k: r["atoms"][k][0]) for k in range(3, 11)}
    out.append(r"""\begin{table}[ht]
\centering
\caption{GenEval2 by prompt complexity: mean prompt score ($\times 100$) in each atom-count bucket, 100 prompts per bucket, averaged over training seeds where applicable. Bold marks the best 4-step distilled arm in each column.}
\label{tab:geneval2_complexity}
\begin{tabular}{l cccccccc}
\toprule
Model & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\
\midrule""")
    for n, r in res.items():
        d = n in distilled
        out.append(" & ".join([SHORT[n]] + [cell(100*r["atoms"][k][0], d and n == best_k[k], "{:.1f}") for k in range(3, 11)]) + r" \\")
        if n.startswith("SD3.5-M base"): out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    # 3. CompBench
    best_mean = col_best(lambda r: r["compbench_mean"][0]); best_c = {c: col_best(lambda r, c=c: r["compbench"][c][0]) for c in CB}
    heads = {"color": "Color", "shape": "Shape", "texture": "Texture", "spatial": "Spatial", "3d_spatial": "3D-spatial",
             "numeracy": "Numeracy", "non_spatial": "Non-spatial", "complex": "Complex"}
    out.append(r"""\begin{table}[ht]
\centering
\caption{T2I-CompBench on the sealed 2{,}398-prompt pool, official scorers, one image per prompt. \emph{Mean} is the unweighted mean of the eight categories ($\pm$ standard deviation over training seeds); category columns are averaged over seeds. Bold marks the best 4-step distilled arm in each column.}
\label{tab:compbench_categories}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c cccccccc}
\toprule
Model & Mean & """ + " & ".join(heads[c] for c in CB) + r""" \\
\midrule""")
    for n, r in res.items():
        d = n in distilled
        mv = f"{r['compbench_mean'][0]:.4f}" + (f" $\\pm$ {r['compbench_mean'][1]:.4f}" if r["n_seeds"] > 1 else "")
        cells = [SHORT[n], cell(mv, d and n == best_mean, "{}")] + [cell(r["compbench"][c][0], d and n == best_c[c], "{:.4f}") for c in CB]
        out.append(" & ".join(cells) + r" \\")
        if n.startswith("SD3.5-M base"): out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")
    return "\n".join(out)


if __name__ == "__main__":
    main()
