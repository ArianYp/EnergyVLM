#!/usr/bin/env python3
"""Recompute every number used in iclr2027/iclr2027_conference.tex from the raw evaluation records.

Sources (nothing is taken from a markdown summary or from memory):
  phaseN/eval_<label>_s<seed>_<job>/alignment.json                 CompBench per category + mean, GenEval2
  phaseN/eval_<label>_s<seed>_<job>/compbench_scores/*/scores.json  per-prompt CompBench scores
  phaseN/eval_<label>_s<seed>_<job>/geneval2_scores/*/scores.json   per-prompt GenEval2 (atoms, skills)
  phaseN/eval10_<label>_s<seed>_<job>/...                            official ten-images-per-prompt CompBench
  phaseW/fidelity_*_report.md                                        FID / CMMD / precision / recall tables

    python3 iclr2027/verify_numbers.py            # prints every table, writes iclr2027/numbers.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
CAT_NAMES = {"color": "Color", "shape": "Shape", "texture": "Texture", "spatial": "2D-spatial", "3d_spatial": "3D-spatial",
             "numeracy": "Numeracy", "non_spatial": "Non-spatial", "complex": "Complex"}
SKILLS = ["object", "count", "attribute", "position", "verb"]

# label -> display name.  3k labels live under phaseN/eval_S4_*, 118k under phaseN/eval_W_* / eval10_W_*.
ARMS_3K = {
    "S4_B2-avglast3": "Random selection, no reward (naive CD)",
    "S4_CD_dinop_hard-avglast3": "Argmax selection, no reward",
    "S4_CD_dinop_hard-rewF-avglast3": "Argmax + projector reward, frozen",
    "S4_CD_dinop_hard-rewR-avglast3": "Argmax + projector reward, refreshed every 100, 4 steps",
    "S4_CD_dinop_hard-rewRi-e25-avglast3": "Argmax + projector reward, refreshed every 25, 4 steps",
    "S4_CD_dinop_hard-rewRi-s16-avglast3": "Argmax + projector reward, refreshed every 100, 16 steps (ours)",
    "S4_CD_dinop_hard-rewRi-s16-maxpool-avglast3": "Ours, DINOv2 max-pooled patches (selection + reward)",
    "S4_CD_dinop_hard-rewX-avglast3": "Argmax + decode-based DINOv2 reward (first round, unmatched order)",
    "S4_CD_dinop_hard-rewXi-avglast3": "Argmax + decode-based DINOv2 reward (matched caption order)",
    "S4_B2-rewXi-avglast3": "Random + decode-based DINOv2 reward",
}
RAW_3K = {k.replace("-avglast3", ""): v for k, v in ARMS_3K.items()}
ARMS_118K = {
    "W_B2_118k-avglast5": "Random selection (naive CD)",
    "W_CD_dinop_hard_118k-avglast5": "Argmax selection",
    "W_CD_dinop_hard_118k-rewX-avglast5": "Argmax + decode-based DINOv2 reward",
    "W_CD_dinop_hard-rewRi-s16_118k-avglast5": "Ours: scored + refreshed projector reward",
}
RAW_118K = {"W_B2_118k": "naive raw final", "W_CD_dinop_hard-rewRi-s16_118k": "ours raw final"}
REFS = {"REF_teacher_s28cfg7": "Teacher, 28 steps, w=7", "teacher_s8cfg7": "Teacher, 8 steps, w=7 (distillation source)",
        "REF_base_s4cfg7": "Base model, 4 steps, w=7"}


# ----------------------------------------------------------------------------- loading
def eval_dirs(label: str, prefix="eval") -> dict[int, str]:
    """{seed: dir} for one label; newest job wins if a seed was evaluated twice."""
    out = {}
    for d in sorted(glob.glob(f"phaseN/{prefix}_{label}_s[0-9]_*")):
        m = re.search(r"_s(\d)_(\d+)$", d)
        if not m or not os.path.isfile(os.path.join(d, "alignment.json")):
            continue
        s, job = int(m.group(1)), int(m.group(2))
        if s in out:
            prev = int(out[s].rsplit("_", 1)[1])
            print(f"  [warn] {label} seed {s}: two evals ({prev}, {job}); using the newer one", file=sys.stderr)
            if job < prev:
                continue
        out[s] = d
    return out


def load_dir(d: str) -> dict:
    a = json.load(open(os.path.join(d, "alignment.json")))
    # a few seed-4 evals carry an "s4_" (steps) prefix on the category keys; strip it
    cats = {re.sub(r"^s\d+_", "", k): (v if not isinstance(v, dict) else v.get("mean", v.get("score"))) for k, v in a["compbench"].items()}
    rec = {"dir": d, "label": a["label"], "steps": a["steps"], "cfg": a["cfg"], "cb_mean": float(a["compbench_mean"]),
           "cb_cat": {c: float(cats[c]) for c in CATS}, "ge2": float(a["geneval2"]) * 100.0}
    assert abs(np.mean([rec["cb_cat"][c] for c in CATS]) - rec["cb_mean"]) < 1e-9, (d, "compbench_mean is not the category mean")
    pp = {}
    for p in glob.glob(os.path.join(d, "compbench_scores", "*", "scores.json")):
        for r in json.load(open(p))["per_prompt"]:
            pp[(r["category"], r["prompt"])] = float(r["score"])
    rec["cb_prompt"] = pp
    gp = {}
    gfiles = glob.glob(os.path.join(d, "geneval2_scores", "*", "scores.json"))
    if gfiles:
        sc = json.load(open(gfiles[0]))
        assert sc["method"] == "soft_tifa_gm" and sc["n"] == 800, (d, sc["method"], sc["n"])
        assert abs(sc["mean"] * 100 - rec["ge2"]) < 1e-6, (d, "geneval2 in alignment.json != scores.json mean")
        skill_atoms, bucket = defaultdict(list), defaultdict(list)
        collapsed = []
        for r in sc["per_prompt"]:
            gp[r["prompt"]] = {"score": float(r["score"]), "atoms": r["atom_scores"], "skills": r["skills"], "n": int(r["atom_count"])}
            for s, v in zip(r["skills"], r["atom_scores"]):
                skill_atoms[s].append(float(v))
            bucket[int(r["atom_count"])].append(float(r["score"]))
            collapsed.append(r["score"] < 0.05)
        rec["ge2_skill"] = {s: 100 * float(np.mean(skill_atoms[s])) for s in SKILLS}
        rec["ge2_bucket"] = {n: 100 * float(np.mean(bucket[n])) for n in sorted(bucket)}          # mean PROMPT score per bucket
        atom_bucket = defaultdict(list)
        for r in sc["per_prompt"]:
            atom_bucket[int(r["atom_count"])] += [float(v) for v in r["atom_scores"]]
        rec["ge2_bucket_atom"] = {n: 100 * float(np.mean(atom_bucket[n])) for n in sorted(atom_bucket)}  # mean ATOM score per bucket
        rec["ge2_collapsed"] = 100 * float(np.mean(collapsed))
    rec["ge2_prompt"] = gp
    return rec


def load_arm(label: str, prefix="eval") -> dict[int, dict]:
    return {s: load_dir(d) for s, d in eval_dirs(label, prefix).items()}


def fidelity_tables() -> dict[str, dict[int, dict]]:
    """label -> seed -> {fid, cmmd, prec, rec}; labels as written in the md tables (S4_..., W_...)."""
    out = defaultdict(dict)
    pat = re.compile(r"^\|\s*(\S+?)_s(\d)@4\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
    for f in sorted(glob.glob("phaseW/fidelity_*_report.md")):
        for ln in open(f):
            m = pat.match(ln)
            if m:
                lab, s = m.group(1), int(m.group(2))
                assert int(m.group(3)) == 5000, (f, ln)
                row = {"fid": float(m.group(4)), "cmmd": float(m.group(5)), "prec": float(m.group(6)), "rec": float(m.group(7)), "src": f}
                if s in out[lab] and out[lab][s] != row:
                    print(f"  [warn] fidelity {lab} s{s}: differs between files ({out[lab][s]['src']} vs {f}); keeping the newer file", file=sys.stderr)
                out[lab][s] = row
    return out


# ----------------------------------------------------------------------------- statistics
def seed_stats(arm: dict[int, dict], key="cb_mean"):
    v = np.array([arm[s][key] for s in sorted(arm)])
    return {"seeds": sorted(arm), "values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def paired(a: dict[int, dict], b: dict[int, dict], key="cb_mean", seeds=None):
    """seed-paired contrast a - b over the seeds both arms have (optionally restricted)."""
    ss = sorted(set(a) & set(b))
    if seeds is not None:
        ss = [s for s in ss if s in seeds]
    d = np.array([a[s][key] - b[s][key] for s in ss])
    out = {"seeds": ss, "per_seed": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan")}
    out["t_p"] = float(stats.ttest_rel([a[s][key] for s in ss], [b[s][key] for s in ss]).pvalue) if len(d) > 1 else float("nan")
    return out


def paired_cat(a, b, cat, seeds=None):
    aa = {s: {"v": a[s]["cb_cat"][cat]} for s in a}
    bb = {s: {"v": b[s]["cb_cat"][cat]} for s in b}
    return paired(aa, bb, "v", seeds)


def paired_ge2(a, b, field, sub, seeds=None):
    aa = {s: {"v": a[s][field][sub]} for s in a if field in a[s]}
    bb = {s: {"v": b[s][field][sub]} for s in b if field in b[s]}
    return paired(aa, bb, "v", seeds)


def prompt_pooled(a: dict[int, dict], b: dict[int, dict], bench="cb"):
    """per-prompt paired test, pooled over the shared seeds (the 118k protocol of the internal report)."""
    ss = sorted(set(a) & set(b))
    xa, xb = [], []
    for s in ss:
        if bench == "cb":
            pa, pb = a[s]["cb_prompt"], b[s]["cb_prompt"]
            keys = sorted(set(pa) & set(pb))
            xa += [pa[k] for k in keys]; xb += [pb[k] for k in keys]
        else:
            pa, pb = a[s]["ge2_prompt"], b[s]["ge2_prompt"]
            keys = sorted(set(pa) & set(pb))
            xa += [pa[k]["score"] for k in keys]; xb += [pb[k]["score"] for k in keys]
    xa, xb = np.array(xa), np.array(xb)
    d = xa - xb
    return {"n": int(len(d)), "seeds": ss, "mean": float(d.mean()) * (100 if bench == "ge2" else 1),
            "t_p": float(stats.ttest_rel(xa, xb).pvalue), "wilcoxon_p": float(stats.wilcoxon(xa, xb).pvalue) if np.any(d != 0) else 1.0}


def prompt_seedavg(a: dict[int, dict], b: dict[int, dict], bench="cb"):
    """per-prompt paired test after averaging each prompt's score over the shared seeds (n = prompts)."""
    ss = sorted(set(a) & set(b))
    if bench == "cb":
        keys = sorted(set.intersection(*[set(a[s]["cb_prompt"]) for s in ss], *[set(b[s]["cb_prompt"]) for s in ss]))
        xa = np.array([np.mean([a[s]["cb_prompt"][k] for s in ss]) for k in keys]); xb = np.array([np.mean([b[s]["cb_prompt"][k] for s in ss]) for k in keys])
    else:
        keys = sorted(set.intersection(*[set(a[s]["ge2_prompt"]) for s in ss], *[set(b[s]["ge2_prompt"]) for s in ss]))
        xa = np.array([np.mean([a[s]["ge2_prompt"][k]["score"] for s in ss]) for k in keys]); xb = np.array([np.mean([b[s]["ge2_prompt"][k]["score"] for s in ss]) for k in keys])
    d = xa - xb
    return {"n": int(len(d)), "seeds": ss, "mean": float(d.mean()) * (100 if bench == "ge2" else 1),
            "t_p": float(stats.ttest_rel(xa, xb).pvalue), "wilcoxon_p": float(stats.wilcoxon(xa, xb).pvalue) if np.any(d != 0) else 1.0}


def fmt_p(p):
    if p != p:
        return "--"
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return f"{m:.0f}\\times10^{{{e}}}" if round(m) != 1 else f"10^{{{e}}}"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def pm(x, sd, nd=4):
    return f"{x:.{nd}f} +- {sd:.{nd}f}" if sd == sd else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- main
def main():
    out = {}
    print("== loading 3k arms")
    A = {k: load_arm(k) for k in ARMS_3K}
    R = {k: load_arm(k) for k in RAW_3K}
    for k in ARMS_3K:
        print(f"  {k:48s} seeds {sorted(A[k])}")
    print("== loading 118k arms")
    W = {k: load_arm(k) for k in ARMS_118K}
    T = {k: load_arm(k, "eval10") for k in ["W_B2_118k-avglast5", "W_CD_dinop_hard_118k-avglast5"]}
    for k in ARMS_118K:
        print(f"  {k:48s} seeds {sorted(W[k])}  ten-image seeds {sorted(T.get(k, {}))}")
    refs = {k: load_arm(k) for k in REFS}
    for k in REFS:
        refs[k] = {0: load_dir(sorted(glob.glob(f"phaseN/eval_{k}_*"))[-1])} if not refs[k] else refs[k]
    fid = fidelity_tables()

    naive, argmax, ours = A["S4_B2-avglast3"], A["S4_CD_dinop_hard-avglast3"], A["S4_CD_dinop_hard-rewRi-s16-avglast3"]
    S3 = [0, 1, 2]

    # ---- Table: main result (3k, averaged checkpoints)
    print("\n== TABLE main (3k, averaged checkpoints)")
    out["main3k"] = {}
    for k, name in ARMS_3K.items():
        st, sg = seed_stats(A[k]), seed_stats(A[k], "ge2")
        vn, vn3 = paired(A[k], naive), paired(A[k], naive, seeds=S3)
        gn3 = paired(A[k], naive, "ge2", seeds=S3)
        va3, ga3 = paired(A[k], argmax, seeds=S3), paired(A[k], argmax, "ge2", seeds=S3)
        va, ga, gn = paired(A[k], argmax), paired(A[k], argmax, "ge2"), paired(A[k], naive, "ge2")
        out["main3k"][k] = {"name": name, "cb": st, "ge2": sg, "vs_naive_cb_s012": vn3, "vs_naive_ge2_s012": gn3,
                            "vs_naive_cb_allshared": vn, "vs_naive_ge2_allshared": gn, "vs_argmax_cb_s012": va3, "vs_argmax_ge2_s012": ga3,
                            "vs_argmax_cb_allshared": va, "vs_argmax_ge2_allshared": ga,
                            "fid_note": None}
        if len(va["seeds"]) > 3:
            print(f"      all {len(va['seeds'])} seeds: vs naive {vn['mean']:+.4f}+-{vn['sd']:.4f} p={vn['t_p']:.3f} (GE2 {gn['mean']:+.2f}+-{gn['sd']:.2f} p={gn['t_p']:.3f}) | "
                  f"vs argmax {va['mean']:+.4f}+-{va['sd']:.4f} p={va['t_p']:.3f} (GE2 {ga['mean']:+.2f}+-{ga['sd']:.2f} p={ga['t_p']:.3f})")
        print(f"  {name:62s} n={len(st['seeds'])} CB {pm(st['mean'], st['sd'])}  GE2 {pm(sg['mean'], sg['sd'], 2)} | "
              f"vs naive (s0-2) {vn3['mean']:+.4f}+-{vn3['sd']:.4f} p={vn3['t_p']:.3f} | GE2 {gn3['mean']:+.2f}+-{gn3['sd']:.2f} p={gn3['t_p']:.2f} | "
              f"vs argmax (s0-2) {va3['mean']:+.4f}+-{va3['sd']:.4f} p={va3['t_p']:.3f}")
    print("  (five-seed contrasts) argmax vs naive:", {k: round(v, 4) if isinstance(v, float) else v for k, v in paired(argmax, naive).items()})
    print("  (five-seed contrasts) argmax vs naive GE2:", {k: round(v, 3) if isinstance(v, float) else v for k, v in paired(argmax, naive, 'ge2').items()})
    print("  raw-final CompBench means:", {k: round(seed_stats(R[k])['mean'], 4) for k in RAW_3K if R[k]})

    # ---- Table: per-category, ours vs naive and vs argmax (3k)
    print("\n== TABLE per-category (3k, seeds 0-2, ours vs naive / argmax)")
    out["cat3k"] = {}
    for c in CATS:
        n_ = seed_stats({s: {"v": naive[s]["cb_cat"][c]} for s in S3}, "v"); a_ = seed_stats({s: {"v": argmax[s]["cb_cat"][c]} for s in S3}, "v")
        o_ = seed_stats({s: {"v": ours[s]["cb_cat"][c]} for s in S3}, "v")
        dn, da = paired_cat(ours, naive, c, S3), paired_cat(ours, argmax, c, S3)
        out["cat3k"][c] = {"naive": n_, "argmax": a_, "ours": o_, "ours_vs_naive": dn, "ours_vs_argmax": da}
        print(f"  {CAT_NAMES[c]:12s} naive {pm(n_['mean'], n_['sd'])}  argmax {pm(a_['mean'], a_['sd'])}  ours {pm(o_['mean'], o_['sd'])} | "
              f"ours-naive {dn['mean']:+.4f}+-{dn['sd']:.4f} p={dn['t_p']:.3f} | ours-argmax {da['mean']:+.4f}+-{da['sd']:.4f} p={da['t_p']:.3f}")

    # ---- Table: GenEval2 breakdown (3k)
    print("\n== TABLE GenEval2 breakdown (3k, seeds 0-2)")
    out["ge2_3k"] = {}
    rows = ([("overall", "ge2", None), ("collapsed", "ge2_collapsed", None)] + [(f"skill:{s}", "ge2_skill", s) for s in SKILLS]
            + [(f"atoms={n} (prompt)", "ge2_bucket", n) for n in range(3, 11)] + [(f"atoms={n} (atom)", "ge2_bucket_atom", n) for n in range(3, 11)])
    for name, field, sub in rows:
        def get(arm, s):
            return arm[s][field] if sub is None else arm[s][field][sub]
        n_ = seed_stats({s: {"v": get(naive, s)} for s in S3}, "v"); a_ = seed_stats({s: {"v": get(argmax, s)} for s in S3}, "v"); o_ = seed_stats({s: {"v": get(ours, s)} for s in S3}, "v")
        dn = paired({s: {"v": get(ours, s)} for s in S3}, {s: {"v": get(naive, s)} for s in S3}, "v")
        da = paired({s: {"v": get(ours, s)} for s in S3}, {s: {"v": get(argmax, s)} for s in S3}, "v")
        out["ge2_3k"][name] = {"naive": n_, "argmax": a_, "ours": o_, "ours_vs_naive": dn, "ours_vs_argmax": da}
        print(f"  {name:16s} naive {pm(n_['mean'], n_['sd'], 1)}  argmax {pm(a_['mean'], a_['sd'], 1)}  ours {pm(o_['mean'], o_['sd'], 1)} | "
              f"ours-naive {dn['mean']:+.1f}+-{dn['sd']:.1f} p={dn['t_p']:.2f} | ours-argmax {da['mean']:+.1f}+-{da['sd']:.1f} p={da['t_p']:.2f}")

    # ---- 118k
    print("\n== TABLE 118k (averaged checkpoints, 3 seeds)")
    out["scale"] = {}
    wn, wa, wx = W["W_B2_118k-avglast5"], W["W_CD_dinop_hard_118k-avglast5"], W["W_CD_dinop_hard_118k-rewX-avglast5"]
    for k, name in ARMS_118K.items():
        st, sg = seed_stats(W[k]), seed_stats(W[k], "ge2")
        vn, gn = paired(W[k], wn), paired(W[k], wn, "ge2")
        pp, pg = prompt_pooled(W[k], wn), prompt_pooled(W[k], wn, "ge2")
        ps, pgs = prompt_seedavg(W[k], wn), prompt_seedavg(W[k], wn, "ge2")
        va, ga = paired(W[k], wa), paired(W[k], wa, "ge2")
        pa, pga = prompt_seedavg(W[k], wa), prompt_seedavg(W[k], wa, "ge2")
        out["scale"][k] = {"name": name, "cb": st, "ge2": sg, "vs_naive_cb_seed": vn, "vs_naive_ge2_seed": gn, "vs_naive_cb_prompt": pp,
                           "vs_naive_ge2_prompt": pg, "vs_naive_cb_prompt_seedavg": ps, "vs_naive_ge2_prompt_seedavg": pgs,
                           "vs_argmax_cb_seed": va, "vs_argmax_ge2_seed": ga, "vs_argmax_cb_prompt_seedavg": pa, "vs_argmax_ge2_prompt_seedavg": pga,
                           "cb_cat": {c: seed_stats({s: {"v": W[k][s]["cb_cat"][c]} for s in W[k]}, "v") for c in CATS},
                           "cb_cat_vs_naive": {c: paired_cat(W[k], wn, c) for c in CATS},
                           "ge2_skill": {s_: seed_stats({s: {"v": W[k][s]["ge2_skill"][s_]} for s in W[k]}, "v") for s_ in SKILLS},
                           "ge2_bucket": {n: seed_stats({s: {"v": W[k][s]["ge2_bucket"][n]} for s in W[k]}, "v") for n in range(3, 11)},
                           "ge2_bucket_atom": {n: seed_stats({s: {"v": W[k][s]["ge2_bucket_atom"][n]} for s in W[k]}, "v") for n in range(3, 11)},
                           "ge2_collapsed": seed_stats({s: {"v": W[k][s]["ge2_collapsed"]} for s in W[k]}, "v")}
        print(f"  {name:40s} CB {pm(st['mean'], st['sd'])} GE2 {pm(sg['mean'], sg['sd'], 2)} | vs naive: seed-paired {vn['mean']:+.4f}+-{vn['sd']:.4f} p={vn['t_p']:.3f}, "
              f"prompt-pooled p={pp['t_p']:.1e} (n={pp['n']}), prompt-seedavg p={ps['t_p']:.1e} (n={ps['n']}) | GE2 {gn['mean']:+.2f}+-{gn['sd']:.2f} p={gn['t_p']:.3f}, "
              f"prompt p={pg['t_p']:.1e}, seedavg p={pgs['t_p']:.1e} | vs argmax {va['mean']:+.4f}+-{va['sd']:.4f} p={va['t_p']:.3f} (prompt-seedavg p={pa['t_p']:.1e}), GE2 {ga['mean']:+.2f} (p={pga['t_p']:.1e})")
    print("  per-seed CB:", {k: [round(W[k][s]['cb_mean'], 4) for s in sorted(W[k])] for k in ARMS_118K})
    out["scale_raw"] = {}
    for k, name in RAW_118K.items():
        R_ = load_arm(k)
        if R_:
            out["scale_raw"][k] = {"cb": seed_stats(R_), "ge2": seed_stats(R_, "ge2")}
            print(f"  {name:20s} raw-final per seed CB {[round(R_[s]['cb_mean'], 4) for s in sorted(R_)]} GE2 {[round(R_[s]['ge2'], 2) for s in sorted(R_)]}")
    print("  official ten-image CompBench:")
    out["scale10"] = {}
    for k in T:
        st = seed_stats(T[k]); vn = paired(T[k], T["W_B2_118k-avglast5"]); pp = prompt_pooled(T[k], T["W_B2_118k-avglast5"]); ps = prompt_seedavg(T[k], T["W_B2_118k-avglast5"])
        out["scale10"][k] = {"cb": st, "vs_naive_seed": vn, "vs_naive_prompt": pp, "vs_naive_prompt_seedavg": ps,
                             "cb_cat": {c: seed_stats({s: {"v": T[k][s]["cb_cat"][c]} for s in T[k]}, "v") for c in CATS},
                             "cb_cat_vs_naive": {c: paired_cat(T[k], T["W_B2_118k-avglast5"], c) for c in CATS}}
        print(f"    {k:36s} CB {pm(st['mean'], st['sd'])} per-seed {[round(T[k][s]['cb_mean'], 4) for s in sorted(T[k])]} | vs naive {vn['mean']:+.4f} seed p={vn['t_p']:.3f} prompt-pooled p={pp['t_p']:.1e} n={pp['n']} prompt-seedavg p={ps['t_p']:.1e} n={ps['n']}")
    print("  per-category 118k (one-image | ten-image):")
    for c in CATS:
        a1 = out["scale"]["W_CD_dinop_hard_118k-avglast5"]["cb_cat"][c]["mean"]; n1 = out["scale"]["W_B2_118k-avglast5"]["cb_cat"][c]["mean"]
        x1 = out["scale"]["W_CD_dinop_hard_118k-rewX-avglast5"]["cb_cat"][c]["mean"]
        a10 = out["scale10"]["W_CD_dinop_hard_118k-avglast5"]["cb_cat"][c]["mean"]; n10 = out["scale10"]["W_B2_118k-avglast5"]["cb_cat"][c]["mean"]
        print(f"    {CAT_NAMES[c]:12s} naive {n1:.4f} argmax {a1:.4f} ({a1-n1:+.4f}) reward {x1:.4f} ({x1-n1:+.4f}) | ten: naive {n10:.4f} argmax {a10:.4f} ({a10-n10:+.4f})")
    print("  GenEval2 breakdown 118k (naive | argmax | argmax+decode reward):")
    for name, field, sub in rows:
        vals = []
        for k in ARMS_118K:
            v = np.array([(W[k][s][field] if sub is None else W[k][s][field][sub]) for s in sorted(W[k])]); vals.append(v)
        d = vals[1] - vals[0]; p = stats.ttest_rel(vals[1], vals[0]).pvalue
        print(f"    {name:16s} " + " | ".join(f"{v.mean():.1f}+-{v.std(ddof=1):.1f}" for v in vals) + f" | argmax-naive {d.mean():+.1f}+-{d.std(ddof=1):.1f} p={p:.2f}")
    print("  references:")
    out["refs"] = {}
    for k, name in REFS.items():
        r = refs[k][0]; out["refs"][k] = {"name": name, "cb_mean": r["cb_mean"], "ge2": r["ge2"], "cb_cat": r["cb_cat"], "steps": r["steps"], "cfg": r["cfg"]}
        print(f"    {name:44s} steps {r['steps']} cfg {r['cfg']} CB {r['cb_mean']:.4f} GE2 {r['ge2']:.2f}")

    # ---- training-time monitor (wandb export, iclr2027/monitor.json): true RGB DINO score of decoded predictions
    if os.path.isfile("iclr2027/monitor.json"):
        print("\n== MONITOR (true RGB DINOv2 score of the student's decoded predictions; start = updates 100-300, end = 5800-6000)")
        mon = json.load(open("iclr2027/monitor.json"))["data"]
        out["monitor"] = {}
        for arm in ["rewF", "rewR", "rewRi-e25", "rewRi-s16", "rewX", "rewXi"]:
            if arm not in mon:
                continue
            out["monitor"][arm] = {}
            for s in sorted(mon[arm], key=int):
                def ser(key):
                    pts = [(st, float(v)) for st, v in mon[arm][s].get(key, []) if isinstance(v, (int, float)) and v == v and st % 100 == 0]
                    return sorted(set(pts))
                rgb, prj, rm = ser("reward/rgb_score"), ser("reward/proj_score"), ser("reward/r_mean")
                st_ = float(np.mean([v for t, v in rgb if 100 <= t <= 300])); en_ = float(np.mean([v for t, v in rgb if t >= 5800]))
                pst = float(np.mean([v for t, v in prj if 100 <= t <= 300])); pen = float(np.mean([v for t, v in prj if t >= 5800]))
                out["monitor"][arm][s] = {"rgb_start": st_, "rgb_end": en_, "proj_start": pst, "proj_end": pen,
                                          "r_mean_end": float(np.mean([v for t, v in rm if t >= 5800])), "n_points": len(rgb)}
                print(f"  {arm:10s} s{s}: true score {st_:.3f} -> {en_:.3f} ({en_-st_:+.3f}) | reward proxy {pst:.3f} -> {pen:.3f} ({pen-pst:+.3f}) | n={len(rgb)}")

    # ---- checkpoint curve at 118k (seed 0)
    out["ckpt_curve"] = {}
    for base in ("W_B2_118k", "W_CD_dinop_hard_118k", "W_CD_dinop_hard_118k-rewX"):
        pts = {}
        for s in (5000, 10000, 15000, 20000, 30000, 40000, 50000):
            d = eval_dirs(f"{base}-step{s}")
            if 0 in d:
                pts[s] = load_dir(d[0])["cb_mean"]
        fin = eval_dirs(base); avg = eval_dirs(f"{base}-avglast5")
        if 0 in fin:
            pts[56974] = load_dir(fin[0])["cb_mean"]
        out["ckpt_curve"][base] = {"steps": pts, "avg": load_dir(avg[0])["cb_mean"] if 0 in avg else None,
                                   "raw_final_seeds": {s: load_dir(d)["cb_mean"] for s, d in fin.items()}}
    print("  checkpoint curve (seed 0):", json.dumps(out["ckpt_curve"]))

    # ---- fidelity
    print("\n== FIDELITY")
    out["fid"] = {}
    for lab in ["S4_B2-avglast3", "S4_CD_dinop_hard-avglast3", "S4_CD_dinop_hard-rewF-avglast3", "S4_CD_dinop_hard-rewR-avglast3",
                "S4_CD_dinop_hard-rewRi-e25-avglast3", "S4_CD_dinop_hard-rewRi-s16-avglast3", "S4_CD_dinop_hard-rewX-avglast3",
                "S4_CD_dinop_hard-rewXi-avglast3", "S4_B2-rewXi-avglast3",
                "W_B2_118k-avglast5", "W_CD_dinop_hard_118k-avglast5", "W_CD_dinop_hard_118k-rewX-avglast5",
                "W_CD_dinop_hard-rewRi-s16_118k-avglast5"]:
        if lab not in fid:
            print(f"  {lab}: MISSING"); continue
        rows_ = fid[lab]; ss = sorted(rows_)
        m = {q: np.array([rows_[s][q] for s in ss]) for q in ("fid", "cmmd", "prec", "rec")}
        out["fid"][lab] = {"seeds": ss, **{q: {"per_seed": m[q].tolist(), "mean": float(m[q].mean()), "sd": float(m[q].std(ddof=1)) if len(ss) > 1 else float("nan")} for q in m}}
        print(f"  {lab:44s} seeds {ss} FID {m['fid'].mean():.2f}+-{m['fid'].std(ddof=1):.2f} CMMD {m['cmmd'].mean():.3f}+-{m['cmmd'].std(ddof=1):.3f} "
              f"prec {m['prec'].mean():.3f}+-{m['prec'].std(ddof=1):.3f} rec {m['rec'].mean():.3f}+-{m['rec'].std(ddof=1):.3f} | per-seed CMMD {m['cmmd'].tolist()}")
    for a_, b_ in [("S4_CD_dinop_hard-rewRi-s16-avglast3", "S4_B2-avglast3"), ("S4_CD_dinop_hard-rewRi-s16-avglast3", "S4_CD_dinop_hard-avglast3"),
                   ("S4_CD_dinop_hard-avglast3", "S4_B2-avglast3"), ("W_CD_dinop_hard_118k-avglast5", "W_B2_118k-avglast5"),
                   ("W_CD_dinop_hard_118k-rewX-avglast5", "W_CD_dinop_hard_118k-avglast5"),
                   ("W_CD_dinop_hard-rewRi-s16_118k-avglast5", "W_B2_118k-avglast5")]:
        if a_ in fid and b_ in fid:
            ss = sorted(set(fid[a_]) & set(fid[b_]) & {0, 1, 2})
            line = []
            for q in ("cmmd", "prec", "rec", "fid"):
                d = np.array([fid[a_][s][q] - fid[b_][s][q] for s in ss])
                p = stats.ttest_rel([fid[a_][s][q] for s in ss], [fid[b_][s][q] for s in ss]).pvalue
                line.append(f"{q} {d.mean():+.3f}+-{d.std(ddof=1):.3f} p={p:.2f}")
                out["fid"].setdefault("contrasts", {})[f"{a_} vs {b_}:{q}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "t_p": float(p), "per_seed": d.tolist()}
            print(f"  {a_} - {b_} (seeds {ss}): " + " | ".join(line))

    # ---- qualitative candidates: ours vs naive per prompt (3k, seed 0 shown, margin averaged over seeds 0-2)
    print("\n== QUALITATIVE candidate prompts (ours - naive), 3k averaged models")
    qual = {"compbench": {}, "geneval2": [], "regress_cb": [], "regress_ge2": []}
    keys = sorted(set.intersection(*[set(ours[s]["cb_prompt"]) for s in S3], *[set(naive[s]["cb_prompt"]) for s in S3]))
    marg = {k: float(np.mean([ours[s]["cb_prompt"][k] - naive[s]["cb_prompt"][k] for s in S3])) for k in keys}
    m0 = {k: ours[0]["cb_prompt"][k] - naive[0]["cb_prompt"][k] for k in keys}
    pj = {(r["category"], r["prompt"]): r["idx"] for r in json.load(open(os.path.join(ours[0]["dir"], "compbench", "prompts.json")))}
    for c in CATS:
        ks = sorted([k for k in keys if k[0] == c], key=lambda k: -(marg[k] + m0[k]))
        qual["compbench"][c] = [{"idx": pj[k], "prompt": k[1], "margin_avg": marg[k], "margin_s0": m0[k],
                                 "ours_s0": ours[0]["cb_prompt"][k], "naive_s0": naive[0]["cb_prompt"][k]} for k in ks[:8]]
        wins = np.mean([marg[k] > 0.05 for k in keys if k[0] == c]); loss = np.mean([marg[k] < -0.05 for k in keys if k[0] == c])
        print(f"  {CAT_NAMES[c]:12s} win {wins:.2f} loss {loss:.2f} | top: " + "; ".join(f"[{pj[k]}] {k[1][:40]} ({m0[k]:+.2f})" for k in ks[:4]))
    ks = sorted(keys, key=lambda k: (marg[k] + m0[k]))
    qual["regress_cb"] = [{"idx": pj[k], "category": k[0], "prompt": k[1], "margin_avg": marg[k], "margin_s0": m0[k]} for k in ks[:12]]
    gkeys = sorted(set.intersection(*[set(ours[s]["ge2_prompt"]) for s in S3], *[set(naive[s]["ge2_prompt"]) for s in S3]))
    gm = {k: float(np.mean([ours[s]["ge2_prompt"][k]["score"] - naive[s]["ge2_prompt"][k]["score"] for s in S3])) for k in gkeys}
    g0 = {k: ours[0]["ge2_prompt"][k]["score"] - naive[0]["ge2_prompt"][k]["score"] for k in gkeys}
    gj = {r["prompt"]: r["idx"] for r in json.load(open(os.path.join(ours[0]["dir"], "geneval2", "prompts.json")))}
    gk = sorted(gkeys, key=lambda k: -(gm[k] + g0[k]))
    qual["geneval2"] = [{"idx": gj[k], "prompt": k, "margin_avg": gm[k], "margin_s0": g0[k], "ours_s0": ours[0]["ge2_prompt"][k]["score"],
                         "naive_s0": naive[0]["ge2_prompt"][k]["score"], "atoms": ours[0]["ge2_prompt"][k]["n"]} for k in gk[:24]]
    qual["regress_ge2"] = [{"idx": gj[k], "prompt": k, "margin_avg": gm[k], "margin_s0": g0[k]} for k in sorted(gkeys, key=lambda k: gm[k] + g0[k])[:12]]
    print("  GenEval2 top: " + "; ".join(f"[{q['idx']}] {q['prompt'][:50]} ({q['margin_s0']:+.2f})" for q in qual["geneval2"][:8]))
    out["qual"] = qual
    out["dirs"] = {"naive_s0": naive[0]["dir"], "ours_s0": ours[0]["dir"], "argmax_s0": argmax[0]["dir"],
                   "teacher28": refs["REF_teacher_s28cfg7"][0]["dir"], "naive118k_s0": wn[0]["dir"], "argmax118k_s0": wa[0]["dir"], "rewx118k_s0": wx[0]["dir"]}

    json.dump(out, open("iclr2027/numbers.json", "w"), indent=1, default=float)
    print("\nwrote iclr2027/numbers.json")


if __name__ == "__main__":
    main()
