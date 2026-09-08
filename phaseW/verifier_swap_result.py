#!/usr/bin/env python3
"""The declared endpoints of phaseV/PREREGISTRATION_verifier_swap.md, and nothing else.

The preregistration fixed the primary alignment endpoint as the CompBench **UniDet-detection
subset** (spatial, 3d_spatial, numeracy), paired on seed against B2, with **Holm correction across
the two selector contrasts** -- and stated that each arm is compared to B2 independently, so
whichever arm happens to win is NOT the one reported. This script computes exactly that, plus the
secondary endpoints, and does not search over anything.

Prompts are joined on (category, TEXT). `idx` is assigned per evaluation job in this repo, so an
idx join silently compares different captions -- see the join traps note.

Leave-family-out matters for the interpretation and is printed alongside:
    UniDet detection  spatial, 3d_spatial, numeracy   outside BOTH new verifiers' families
    BLIP-VQA          color, shape, texture           ImageReward is BLIP-based
    CLIPScore         non_spatial                     PickScore is a fine-tuned CLIP-H
    GenEval2          Qwen3-VL                        outside every selector family
"""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

UNIDET = ("spatial", "3d_spatial", "numeracy")
BLIPVQA = ("color", "shape", "texture")
CLIPSC = ("non_spatial",)


def load(root: str):
    runs = defaultdict(dict)
    for d in sorted(glob.glob(f"{root}/eval_*")):
        a = Path(d) / "alignment.json"
        if not a.exists():
            continue
        lb = json.loads(a.read_text())["label"]
        if json.loads(a.read_text()).get("cfg") != 1.0:
            continue                                   # students trained with student_cfg 0
        cb, ge = {}, {}
        for p in glob.glob(str(Path(d) / "compbench_scores" / "*" / "scores.json")):
            for r in json.loads(Path(p).read_text())["per_prompt"]:
                cb[(r["category"], r["prompt"])] = float(r["score"])
        for p in glob.glob(str(Path(d) / "geneval2_scores" / "*" / "scores.json")):
            for r in json.loads(Path(p).read_text())["per_prompt"]:
                ge[r["prompt"]] = float(r["score"])
        if cb:
            runs[lb] = {"compbench": cb, "geneval2": ge}
    return runs


def arm_seeds(runs, arm, seeds=(0, 1, 2)):
    out = {}
    for s in seeds:
        k = f"{arm}_s{s}"
        if k in runs:
            out[s] = runs[k]
    return out


def subset(d, cats):
    return {k: v for k, v in d["compbench"].items() if k[0] in cats}


def contrast(A, B, cats, label, log):
    """Paired per-prompt difference, averaged over the seeds both arms share."""
    seeds = sorted(set(A) & set(B))
    keys = sorted(set.intersection(*[set(subset(A[s], cats)) for s in seeds],
                                   *[set(subset(B[s], cats)) for s in seeds]))
    if not keys:
        log(f"    {label:22s} no shared prompts"); return None, 0.0
    da = np.mean([[A[s]["compbench"][k] for k in keys] for s in seeds], 0)
    db = np.mean([[B[s]["compbench"][k] for k in keys] for s in seeds], 0)
    d = da - db
    tp = ttest_rel(da, db)[1]
    wp = wilcoxon(da, db)[1] if np.any(d != 0) else 1.0
    per_seed = [np.mean([A[s]["compbench"][k] for k in keys]) - np.mean([B[s]["compbench"][k] for k in keys])
                for s in seeds]
    log(f"    {label:22s} n={len(keys):5d}  delta={d.mean():+.4f}  "
        f"per-seed {'/'.join(f'{x:+.4f}' for x in per_seed)}  t p={tp:.3e}  wilcoxon p={wp:.3e}")
    return tp, float(d.mean())


def geneval(A, B, log, label):
    seeds = sorted(set(A) & set(B))
    keys = sorted(set.intersection(*[set(A[s]["geneval2"]) for s in seeds],
                                   *[set(B[s]["geneval2"]) for s in seeds]))
    if not keys:
        log(f"    {label:22s} no shared GenEval2 prompts"); return
    da = np.mean([[A[s]["geneval2"][k] for k in keys] for s in seeds], 0)
    db = np.mean([[B[s]["geneval2"][k] for k in keys] for s in seeds], 0)
    log(f"    {label:22s} n={len(keys):5d}  delta={np.mean(da-db):+.4f}  t p={ttest_rel(da,db)[1]:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="phaseN")
    ap.add_argument("--baseline", default="T_B2")
    ap.add_argument("--reference", default="T_B4")
    ap.add_argument("--out", default="phaseW/verifier_swap_result.json")
    ap.add_argument("--arms", default="V_CD_pick_hard,V_CD_imgrwd_hard",
                    help="comma-separated eval-label prefixes, compared to --baseline independently")
    ap.add_argument("--family", default=None,
                    help="comma-separated subset of --arms that forms the preregistered Holm family "
                         "(default: all of --arms)")
    args = ap.parse_args()
    lines = []
    def log(s): print(s, flush=True); lines.append(s)

    runs = load(args.root)
    B = arm_seeds(runs, args.baseline)
    if not B:
        raise SystemExit(f"no seeds for baseline {args.baseline}; available: {sorted(runs)[:20]}")
    log(f"baseline {args.baseline}: seeds {sorted(B)}")

    names = [a for a in args.arms.split(",") if a]
    arms = {a: arm_seeds(runs, a) for a in names}
    ref = arm_seeds(runs, args.reference)
    if ref:
        arms[f"{args.reference} (VQAScore, existing)"] = ref

    prim = {}
    for name, A in arms.items():
        if not A:
            log(f"\n{name}: NO EVALS FOUND"); continue
        log(f"\n{name} vs {args.baseline}   seeds {sorted(A)}")
        prim[name] = contrast(A, B, UNIDET, "PRIMARY UniDet det.", log)
        contrast(A, B, BLIPVQA, "BLIP-VQA", log)
        contrast(A, B, CLIPSC, "CLIPScore non_spatial", log)
        contrast(A, B, UNIDET + BLIPVQA + CLIPSC + ("complex",), "CompBench overall", log)
        geneval(A, B, log, "GenEval2 (Qwen3-VL)")

    # Holm across the TWO declared selector contrasts only; the existing VQAScore arm is a reference
    # point that was not part of the preregistered family and does not enter the correction.
    famnames = [a for a in (args.family or args.arms).split(",") if a]
    fam = {k: v for k, v in prim.items() if k in famnames and v[0] is not None}
    log(f"\nHolm correction across the {len(fam)} preregistered selector contrasts, alpha=0.05:")
    for i, (k, (p, dl)) in enumerate(sorted(fam.items(), key=lambda t: t[1][0])):
        a = 0.05 / (len(fam) - i)
        # One-sided: the arm must BEAT the baseline. A small p with a negative delta is a
        # significant LOSS and must never print as a pass.
        v = "PASS" if (p < a and dl > 0) else ("WORSE (significant)" if (p < a and dl < 0) else "fail")
        log(f"  {k:16s} delta={dl:+.4f}  p={p:.3e}  alpha={a:.4f}  {v}")

    json.dump({"primary": {k: {"p": v[0], "delta": v[1]} for k, v in prim.items() if v[0] is not None},
               "log": lines}, open(args.out, "w"), indent=1)
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
