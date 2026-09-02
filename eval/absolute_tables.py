#!/usr/bin/env python3
"""Absolute (not delta) GenEval2 and T2I-CompBench scores per model, averaged over seeds.

GenEval2 here is the Soft-TIFA geometric-mean evaluator: a prompt is decomposed into atoms, each
atom is tagged with a skill (object / count / attribute / position / verb), the prompt score is
the geometric mean of its atom scores, and the official number is the mean prompt score x100.
Skill columns are the mean ATOM score for that skill; the atom-count columns are the mean PROMPT
score per complexity bucket. "Atom mean" is the arithmetic mean over all atoms and "collapsed" is
the share of prompts whose geometric mean is below 0.05; both are reported because the geometric
mean gives partial credit and can rank a decisive teacher below hedging students.

Each --model is NAME=GLOB, where GLOB matches one evaluation directory per seed
(the output of scripts/eval_alignment.lsf, holding alignment.json and geneval2_scores/*/scores.json).

    python eval/absolute_tables.py \
        --model "naive=out/eval/eval_random_s[0-9]" \
        --model "DINO patches=out/eval/eval_dino_patch_s[0-9]" \
        --model "teacher 28 steps=out/eval/eval_base28" \
        --out out/absolute_scores.json --tex out/absolute_tables.tex
"""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

SKILLS = ["object", "count", "attribute", "position", "verb"]
CB = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
CB_HEAD = {"color": "Color", "shape": "Shape", "texture": "Texture", "spatial": "Spatial", "3d_spatial": "3D-spatial",
           "numeracy": "Numeracy", "non_spatial": "Non-spatial", "complex": "Complex"}


def load_dir(d: Path) -> dict:
    al = json.load(open(d / "alignment.json"))
    sc = json.load(open(next(iter(glob.glob(str(d / "geneval2_scores/*/scores.json"))))))
    assert sc["method"] == "soft_tifa_gm", (d, sc["method"])
    skill_atoms = defaultdict(list); bucket = defaultdict(list); arith = []; dead = []
    for r in sc["per_prompt"]:
        bucket[int(r["atom_count"])].append(r["score"])
        arith.append(float(np.mean(r["atom_scores"]))); dead.append(r["score"] < 0.05)
        for s, a in zip(r["skills"], r["atom_scores"]):
            skill_atoms[s].append(a)
    return {"steps": al["steps"], "cfg": al["cfg"], "geneval2": sc["mean"],
            "arith": float(np.mean(arith)), "dead": float(np.mean(dead)),
            "skill": {s: float(np.mean(skill_atoms[s])) if skill_atoms[s] else float("nan") for s in SKILLS},
            "atoms": {k: float(np.mean(v)) for k, v in bucket.items()},
            "compbench": al["compbench"], "compbench_mean": al["compbench_mean"]}


def aggregate(models: list[tuple[str, str]]) -> dict:
    res = {}
    for name, pat in models:
        dirs = sorted(Path(p) for p in glob.glob(pat)); assert dirs, f"no evaluation directory matches {pat}"
        per = [load_dir(d) for d in dirs]
        def agg(f):
            v = [f(p) for p in per]
            return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        buckets = sorted({k for p in per for k in p["atoms"]})
        res[name] = {"n_seeds": len(per), "dirs": [str(d) for d in dirs], "steps": per[0]["steps"], "cfg": per[0]["cfg"],
                     "geneval2": agg(lambda p: p["geneval2"]), "arith": agg(lambda p: p["arith"]), "dead": agg(lambda p: p["dead"]),
                     "skill": {s: agg(lambda p, s=s: p["skill"][s]) for s in SKILLS},
                     "atoms": {k: agg(lambda p, k=k: p["atoms"].get(k, float("nan"))) for k in buckets},
                     "compbench_mean": agg(lambda p: p["compbench_mean"]),
                     "compbench": {c: agg(lambda p, c=c: p["compbench"].get(c, float("nan"))) for c in CB}}
    return res


def markdown(res: dict) -> str:
    out = []
    def row(cells): out.append("| " + " | ".join(cells) + " |")
    buckets = sorted({k for r in res.values() for k in r["atoms"]})
    out.append("### GenEval2 (Soft-TIFA gmean), x100; skills = mean atom score, mean +- seed std\n")
    row(["model", "steps/cfg", "seeds", "**overall (gmean)**"] + SKILLS + ["atom arith. mean", "% prompts gmean<0.05"]); row(["---"] * (5 + len(SKILLS)))
    for n, r in res.items():
        row([n, f"{r['steps']}/{r['cfg']:g}", str(r["n_seeds"]), f"**{100*r['geneval2'][0]:.2f}** +- {100*r['geneval2'][1]:.2f}"]
            + [f"{100*r['skill'][s][0]:.1f} +- {100*r['skill'][s][1]:.1f}" for s in SKILLS]
            + [f"{100*r['arith'][0]:.1f}", f"{100*r['dead'][0]:.1f}"])
    out.append("\n### GenEval2 by prompt complexity (atoms per prompt), x100\n")
    row(["model"] + [str(k) for k in buckets]); row(["---"] * (1 + len(buckets)))
    for n, r in res.items():
        row([n] + [f"{100*r['atoms'][k][0]:.1f}" for k in buckets])
    out.append("\n### T2I-CompBench (official 8 categories), mean +- seed std\n")
    row(["model", "**mean**"] + CB); row(["---"] * (1 + len(CB)))
    for n, r in res.items():
        row([n, f"**{r['compbench_mean'][0]:.4f}** +- {r['compbench_mean'][1]:.4f}"] + [f"{r['compbench'][c][0]:.4f}" for c in CB])
    return "\n".join(out)


def latex(res: dict, bold: set[str]) -> str:
    """Three booktabs tables. Bold marks the best model among `bold` in each column."""
    def cell(v, best, fmt):
        s = fmt.format(v); return f"\\textbf{{{s}}}" if best else s
    def best(getter, lo=False):
        c = [n for n in res if n in bold]
        return (min if lo else max)(c, key=lambda n: getter(res[n])) if c else None
    buckets = sorted({k for r in res.values() for k in r["atoms"]})
    out = []
    bo = best(lambda r: r["geneval2"][0]); bs = {s: best(lambda r, s=s: r["skill"][s][0]) for s in SKILLS}
    ba = best(lambda r: r["arith"][0]); bd = best(lambda r: r["dead"][0], lo=True)
    out.append(r"""\begin{table}[ht]
\centering
\caption{GenEval2 (Soft-TIFA, geometric mean), $\times 100$. \emph{Overall} is the official score ($\pm$ standard deviation over seeds). Skill columns are the mean score of the atoms tagged with that skill. \emph{Atom mean} is the arithmetic mean over all atoms. \emph{Collapsed} is the share of prompts whose geometric mean falls below 0.05.}
\label{tab:geneval2_skills}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c ccccc c c}
\toprule
Model & Steps / CFG & Seeds & Overall & Object & Count & Attribute & Position & Verb & Atom mean & Collapsed (\%) \\
\midrule""")
    for n, r in res.items():
        d = n in bold
        ov = f"{100*r['geneval2'][0]:.2f}" + (f" $\\pm$ {100*r['geneval2'][1]:.2f}" if r["n_seeds"] > 1 else "")
        cells = [n, f"{r['steps']} / {r['cfg']:g}", str(r["n_seeds"]), cell(ov, d and n == bo, "{}")]
        cells += [cell(100*r["skill"][s][0], d and n == bs[s], "{:.1f}") for s in SKILLS]
        cells += [cell(100*r["arith"][0], d and n == ba, "{:.1f}"), cell(100*r["dead"][0], d and n == bd, "{:.1f}")]
        out.append(" & ".join(cells) + r" \\")
    out.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    bk = {k: best(lambda r, k=k: r["atoms"][k][0]) for k in buckets}
    out.append(r"""\begin{table}[ht]
\centering
\caption{GenEval2 by prompt complexity: mean prompt score ($\times 100$) in each atom-count bucket.}
\label{tab:geneval2_complexity}
\begin{tabular}{l """ + "c" * len(buckets) + r"""}
\toprule
Model & """ + " & ".join(str(k) for k in buckets) + r""" \\
\midrule""")
    for n, r in res.items():
        d = n in bold
        out.append(" & ".join([n] + [cell(100*r["atoms"][k][0], d and n == bk[k], "{:.1f}") for k in buckets]) + r" \\")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    bm = best(lambda r: r["compbench_mean"][0]); bc = {c: best(lambda r, c=c: r["compbench"][c][0]) for c in CB}
    out.append(r"""\begin{table}[ht]
\centering
\caption{T2I-CompBench, official scorers. \emph{Mean} is the unweighted mean of the eight categories ($\pm$ standard deviation over seeds).}
\label{tab:compbench_categories}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c cccccccc}
\toprule
Model & Mean & """ + " & ".join(CB_HEAD[c] for c in CB) + r""" \\
\midrule""")
    for n, r in res.items():
        d = n in bold
        mv = f"{r['compbench_mean'][0]:.4f}" + (f" $\\pm$ {r['compbench_mean'][1]:.4f}" if r["n_seeds"] > 1 else "")
        out.append(" & ".join([n, cell(mv, d and n == bm, "{}")] + [cell(r["compbench"][c][0], d and n == bc[c], "{:.4f}") for c in CB]) + r" \\")
    out.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=GLOB, one evaluation directory per seed")
    ap.add_argument("--bold", default=None, help="comma-separated NAMEs eligible for bold (default: all)")
    ap.add_argument("--out", default="out/absolute_scores.json")
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()
    models = [tuple(m.split("=", 1)) for m in args.model]
    res = aggregate(models)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(markdown(res)); print(f"\nwrote {args.out}")
    if args.tex:
        bold = set(args.bold.split(",")) if args.bold else set(res)
        Path(args.tex).write_text(latex(res, bold)); print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
