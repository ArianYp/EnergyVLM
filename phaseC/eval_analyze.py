#!/usr/bin/env python3
"""
Phase C1/D — the treatment-vs-control verdict on the pre-registered independent metrics.

Treatment = models[0], control = models[1] (a third model, e.g. M1, is shown for context and
`base*` triggers the teacher-gap section). So:
  * Phase C1  : --models B4,B2,M1         → the oracle-trajectory (B4) vs random (B2) verdict
  * Phase D1  : --models B5_latent,B2,M1  → the target-modification (B5-latent) vs random (B2) verdict
The comparison is generic; only the free-text "Read" footer branches on the specific pair, so a
`B4,B2,M1` run reproduces the locked Phase-C1 verdict byte-for-byte.

Reads the official-evaluator per-prompt scores collected by `compbench_eval.py`
(`{scores_dir}/{model}_s{step}_{cat}/scores.json`, field per_prompt[].score) and, if present,
GenEval2 (`geneval2_eval.py`), fidelity (`fidelity_eval.py`) and diversity (`diversity_eval.py`),
then computes, per the pre-registration + its deviations:

  * per-category treat−control, paired over prompt idx, 10k-bootstrap 95% CI  (primary granularity)
  * PRIMARY 1 = equal-category-weighted mean of per-category treat−control over the SEVEN
    categories whose official evaluator the pre-registration names as primary (BLIP-VQA ×3,
    UniDet ×3, 3-in-1 ×1) — not prompt-pooled (Deviation 1)
  * PRIMARY 2 = GenEval2 (Soft-TIFA), paired over prompts
  * non_spatial is scored by CLIPScore, which §3 lists as SECONDARY (Deviation 2)
  * transfer ratio T = indep(treat−control) / VQA(treat−control)  if VQA scores are provided
  * the gate: PRIMARY CI excludes 0 **and** no fidelity loss **and** no diversity collapse.

Usage:
  python phaseC/eval_analyze.py --scores_dir phaseC/eval_scores --step 4 --models B4,B2,M1 \
      [--geneval2_dir phaseC/geneval2_scores] [--vqa phaseC/eval/scores] \
      [--fidelity phaseC/fidelity_report.json] [--diversity phaseC/diversity_report.json]
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
BOOT = 10_000

# Categories entering the pooled primary, with the official evaluator behind each.
# non_spatial is deliberately absent: its official evaluator is CLIPScore, which the
# pre-registration §3 lists under "Secondary (reported, never the primary proof)".
PRIMARY_CATS = {
    "color": "BLIP-VQA", "shape": "BLIP-VQA", "texture": "BLIP-VQA",
    "spatial": "UniDet", "3d_spatial": "UniDet", "numeracy": "UniDet",
    "complex": "3-in-1",
}
SECONDARY_CATS = {"non_spatial": "CLIPScore"}
CATS = list(PRIMARY_CATS) + list(SECONDARY_CATS)
# Zero-inflated in practice — measured, not assumed. Deviation 1 expected this of all three
# UniDet categories; job 85660 showed it is specific to 2D `spatial` (29/40 validation pairs
# tied at exactly 0, vs 1/40 for 3d_spatial and 0/40 for numeracy), because only
# 2D_spatial_eval.py hard-thresholds <0.5 -> 0. So one primary category is under-powered,
# not three. See Deviation 3.
ZERO_INFLATED = {"spatial"}


def load_scores(scores_dir, model, step, cat, prompt_out=None):
    """-> {idx: score} for one (model, step, cat); fills prompt_out[idx]=prompt if given."""
    f = Path(scores_dir) / f"{model}_s{step}_{cat}" / "scores.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    if prompt_out is not None:
        prompt_out.update({r["idx"]: r["prompt"] for r in d["per_prompt"] if "prompt" in r})
    return {r["idx"]: r["score"] for r in d["per_prompt"]}


def seen_unseen(dataset_dir, cat):
    """T2I-CompBench's own seen/unseen split of the val set, for the generalization read.

    Upstream ships `{cat}_val_{seen,unseen}.txt` for the three attribute-binding
    categories: 200 prompts whose attribute-object pairings also occur in train ("seen")
    and 100 whose pairings do not ("unseen"). shape and texture partition their val set
    exactly; **color's two files cover only 295 of its 300 val prompts** (5 are in
    neither, an upstream inconsistency), so 5 color prompts fall in neither bucket and
    are counted as such rather than forced into one.
    """
    out = {}
    for name in ("seen", "unseen"):
        f = Path(dataset_dir) / f"{cat}_val_{name}.txt"
        out[name] = ({l.strip() for l in f.read_text().splitlines() if l.strip()}
                     if f.exists() else set())
    return out


def load_geneval2(geneval2_dir, model, step):
    f = Path(geneval2_dir) / f"{model}_s{step}" / "scores.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {r["idx"]: r["score"] for r in d["per_prompt"]}


def paired(a, b):
    """paired (a-b) over shared idx -> np.array."""
    ks = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in ks]), ks


def boot_mean_ci(vals):
    vals = np.asarray(vals)
    if len(vals) < 2:
        return float(vals.mean()) if len(vals) else float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(vals), size=(BOOT, len(vals)))
    m = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def gate_word(lo, hi, t, c):
    return f"PASS ({t} > {c})" if lo > 0 else (f"FAIL ({c} > {t})" if hi < 0 else "NULL (CI spans 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", default="phaseC/eval_scores")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--models", default="B4,B2,M1",
                    help="treatment,control[,context...] — treat=models[0], control=models[1]")
    ap.add_argument("--geneval2_dir", default="phaseC/geneval2_scores")
    ap.add_argument("--vqa", default=None, help="dir with phaseA_score jsonl for transfer ratio T")
    ap.add_argument("--fidelity", default="phaseC/fidelity_report.json")
    ap.add_argument("--diversity", default="phaseC/diversity_report.json")
    ap.add_argument("--dataset_dir", default="T2I-CompBench/examples/dataset",
                    help="for the exploratory seen/unseen split of the attribute categories")
    ap.add_argument("--out", default="phaseC/eval_verdict.md")
    ap.add_argument("--title", default="Phase C1",
                    help="report title prefix (e.g. 'Phase F') — the harness is shared")
    args = ap.parse_args()
    models = args.models.split(",")
    TREAT, CTRL = models[0], models[1]   # treatment vs control

    PROMPT = {c: {} for c in CATS}
    S = {m: {c: load_scores(args.scores_dir, m, args.step, c, PROMPT[c]) for c in CATS}
         for m in models}
    present = [c for c in CATS if S[TREAT].get(c) and S[CTRL].get(c)]
    prim_present = [c for c in present if c in PRIMARY_CATS]
    missing_prim = [c for c in PRIMARY_CATS if c not in present]
    summary = {"step": args.step, "treatment": TREAT, "control": CTRL, "per_category": {},
               "primary_categories": prim_present, "missing_primary_categories": missing_prim}

    md = [f"# {args.title} — {TREAT} vs {CTRL} verdict (independent metrics, {args.step}-step)\n",
          f"scores: `{args.scores_dir}` | {BOOT} bootstrap | primary categories present: "
          f"{len(prim_present)}/{len(PRIMARY_CATS)} {prim_present}\n"]
    if missing_prim:
        md.append(f"\n> **Incomplete primary:** {missing_prim} not yet scored — the pooled "
                  f"primary below is over the categories present, not the full "
                  f"pre-registered set.\n")
    if any(c in SECONDARY_CATS for c in present):
        md.append(f"\nSecondary, reported but NOT in the pooled primary: "
                  f"{[c for c in present if c in SECONDARY_CATS]} "
                  f"(CLIPScore — pre-registration §3 lists it as secondary; Deviation 2).\n")
    md.append("\n")

    # per-model per-category mean (context)
    md.append("## per-category mean score\n\n| category | evaluator | " + " | ".join(models)
              + " |\n" + "|---" * (len(models) + 2) + "|\n")
    for c in present:
        row = []
        for m in models:
            v = list(S[m].get(c, {}).values())
            row.append(f"{np.mean(v):.4f}" if v else "—")
        ev = PRIMARY_CATS.get(c) or SECONDARY_CATS[c]
        md.append(f"| {c} | {ev} | " + " | ".join(row) + " |\n")

    # per-category treat - control (paired)
    md.append(f"\n## {TREAT} − {CTRL} (paired, per category)\n\n"
              "`0-tie` = share of pairs where both models scored exactly 0, i.e. the part of n "
              "carrying no information. ⚠ marks the one category known to be zero-inflated.\n\n"
              f"| category | role | Δ{TREAT}−{CTRL} | 95% CI | n | 0-tie | sig |\n"
              "|---|---|---|---|---|---|---|\n")
    for c in present:
        d, ks = paired(S[TREAT][c], S[CTRL][c])
        m, lo, hi = boot_mean_ci(d)
        sig = "YES" if (lo > 0 or hi < 0) else "n.s."
        role = "primary" if c in PRIMARY_CATS else "secondary"
        # zero-inflation is a property of the scores in hand, so report it rather than
        # relying on the category name
        tied = sum(1 for k in ks if S[TREAT][c][k] == 0 and S[CTRL][c][k] == 0)
        md.append(f"| {c}{' ⚠' if c in ZERO_INFLATED else ''} | {role} | {m:+.4f} | "
                  f"[{lo:+.4f}, {hi:+.4f}] | {len(d)} | {100 * tied / max(len(ks), 1):.0f}% | {sig} |\n")
        summary["per_category"][c] = {"delta": m, "ci": [lo, hi], "n": len(d), "role": role,
                                      "pairs_tied_at_zero": tied}

    # PRIMARY 1: equal-category-weighted (bootstrap: resample prompts within each cat,
    # then average the category means, so each category contributes once)
    prim = plo = phi = float("nan")
    if prim_present:
        cat_arrs = {c: paired(S[TREAT][c], S[CTRL][c])[0] for c in prim_present}
        boot = np.empty(BOOT)
        for b in range(BOOT):
            boot[b] = np.mean([cat_arrs[c][RNG.integers(0, len(cat_arrs[c]),
                                                        len(cat_arrs[c]))].mean()
                               for c in prim_present])
        prim = float(np.mean([cat_arrs[c].mean() for c in prim_present]))
        plo, phi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        gate = gate_word(plo, phi, TREAT, CTRL)
        md.append(f"\n## PRIMARY 1 — T2I-CompBench, equal-category-weighted {TREAT} − {CTRL}\n\n"
                  f"**{prim:+.4f}  95% CI [{plo:+.4f}, {phi:+.4f}]  →  H1 gate: {gate}**\n"
                  f"\n(equal weight over {len(prim_present)} categories: "
                  f"{', '.join(prim_present)})\n")
        summary["primary"] = {"delta": prim, "ci": [plo, phi], "gate": gate,
                              "n_categories": len(prim_present)}

    # Teacher reference on the SAME prompts. Phase A measured the 28-step teacher on the
    # train dev pool, not on val, so "how much of the student-to-teacher gap did TREAT close"
    # could not be read off the primary metric's own prompt set until base was generated
    # here too. Fraction closed = (TREAT-CTRL)/(base-CTRL), only where the teacher is actually
    # ahead of CTRL by a usable margin (a near-zero denominator makes the ratio meaningless).
    if any(S.get(m) for m in models if m.startswith("base")):
        tname = next(m for m in models if m.startswith("base"))
        rows = []
        for c in present:
            t = S[tname].get(c)
            if not t:
                continue
            dt, _ = paired(t, S[CTRL][c])
            db, _ = paired(S[TREAT][c], S[CTRL][c])
            gap = float(dt.mean())
            frac = float(db.mean()) / gap if gap > 0.02 else float("nan")
            rows.append(f"| {c} | {np.mean(list(t.values())):.4f} | {gap:+.4f} | "
                        f"{db.mean():+.4f} | "
                        f"{'—  (teacher not ahead)' if np.isnan(frac) else f'{100 * frac:.0f}%'} |\n")
            summary.setdefault("teacher_gap", {})[c] = {
                "teacher_mean": float(np.mean(list(t.values()))),
                "teacher_minus_control": gap, "frac_closed": frac}
        if rows:
            md.append(f"\n## Teacher reference on the val prompts ({tname})\n\n"
                      f"How much of the {CTRL}→teacher gap {TREAT} closes, on the *same* prompts as "
                      "the primary metric. Phase A's teacher numbers were on the train dev pool, so "
                      "this is the first like-for-like reading.\n\n"
                      f"| category | {tname} | {tname}−{CTRL} | {TREAT}−{CTRL} | gap closed |\n"
                      "|---|---|---|---|---|\n")
            md += rows
            md.append("\nCaveat: the teacher runs at CFG 7 and 28 steps, the students at CFG 1 "
                      "and 4, so this is a *budget* comparison, not a controlled one.\n")

    # Exploratory (NOT pre-registered): does the gain hold on attribute compositions whose
    # pairings never occur in train? Same 300 prompts per category, just partitioned by
    # upstream's own seen/unseen files, so it costs nothing extra and adds no new images.
    su_rows = []
    for c in [c for c in present if c in ("color", "shape", "texture")]:
        sets = seen_unseen(args.dataset_dir, c)
        d4, d2 = S[TREAT][c], S[CTRL][c]
        ks = sorted(set(d4) & set(d2))
        buckets = {"seen": [], "unseen": [], "neither": []}
        for k in ks:
            p = PROMPT[c].get(k, "")
            b = "seen" if p in sets["seen"] else ("unseen" if p in sets["unseen"] else "neither")
            buckets[b].append(d4[k] - d2[k])
        for b in ("seen", "unseen", "neither"):
            if not buckets[b]:
                continue
            m, lo, hi = boot_mean_ci(buckets[b])
            su_rows.append(f"| {c} | {b} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {len(buckets[b])} |\n")
            summary.setdefault("seen_unseen", {}).setdefault(c, {})[b] = {
                "delta": m, "ci": [lo, hi], "n": len(buckets[b])}
    if su_rows:
        md.append("\n## Exploratory — seen vs unseen attribute compositions\n\n"
                  "**Not pre-registered; descriptive only, no gate attached.** Upstream's own "
                  "split of the attribute-binding val sets into pairings that do (200) and do not "
                  "(100) occur in train. `neither` is the 5 color prompts upstream leaves out of "
                  "both files. The unseen n=100 per category is small, so CIs are wide.\n\n"
                  f"| category | bucket | Δ{TREAT}−{CTRL} | 95% CI | n |\n|---|---|---|---|---|\n")
        md += su_rows

    # PRIMARY 2: GenEval2
    g4, g2 = (load_geneval2(args.geneval2_dir, m, args.step) for m in (TREAT, CTRL))
    if g4 and g2:
        dg, _ = paired(g4, g2)
        gm, glo, ghi = boot_mean_ci(dg)
        ggate = gate_word(glo, ghi, TREAT, CTRL)
        md.append(f"\n## PRIMARY 2 — GenEval2 (Soft-TIFA) {TREAT} − {CTRL}\n\n"
                  f"**{gm:+.4f}  95% CI [{glo:+.4f}, {ghi:+.4f}]  →  gate: {ggate}**  "
                  f"(n={len(dg)} prompts)\n\n")
        for m in models:
            v = list(load_geneval2(args.geneval2_dir, m, args.step).values())
            if v:
                md.append(f"- {m}: {np.mean(v):.4f} (official ×100 = {100 * np.mean(v):.2f})\n")
        summary["geneval2"] = {"delta": gm, "ci": [glo, ghi], "gate": ggate, "n": len(dg)}
    else:
        md.append("\n## PRIMARY 2 — GenEval2\n\nNot yet scored "
                  f"(`{args.geneval2_dir}/{{{CTRL},{TREAT}}}_s{args.step}/scores.json` absent).\n")

    # transfer ratio T
    if args.vqa:
        vqa = defaultdict(dict)
        for f in glob.glob(f"{args.vqa}/scores_rank*.jsonl") + glob.glob(f"{args.vqa}/scores.jsonl"):
            for ln in open(f):
                try:
                    r = json.loads(ln)
                    if r.get("steps") == args.step:
                        vqa[r["label"]][r["idx"]] = r["vqa"]
                except Exception:
                    pass
        if vqa.get(TREAT) and vqa.get(CTRL):
            dv, _ = paired(vqa[TREAT], vqa[CTRL])
            vgain = float(dv.mean())
            T = prim / vgain if abs(vgain) > 1e-6 else float("nan")
            md.append(f"\n## transfer ratio T = indep({TREAT}−{CTRL})/VQA({TREAT}−{CTRL})\n"
                      f"- VQA {TREAT}−{CTRL} = {vgain:+.4f} ; independent primary = {prim:+.4f} ; "
                      f"**T = {T:.2f}**\n"
                      f"- T≈0 → VQA-overfit (H1 not supported even if VQA gains); "
                      f"T≈1 → genuine transfer.\n")
            summary["transfer_T"] = T

    # fidelity + diversity gates (the other half of a "clean" H1)
    md.append("\n## Gates beyond alignment (pre-registration §5)\n\n")
    fid = Path(args.fidelity)
    if fid.exists():
        fj = json.loads(fid.read_text())
        g = fj.get("gate", {}).get(str(args.step))
        if g:
            md.append(f"- **fidelity:** CMMD Δ({TREAT}−{CTRL}) = {g['d_cmmd']:+.2f} "
                      f"(tol +{fj.get('cmmd_tol', float('nan')):g}), FID Δ = {g['d_fid']:+.2f}, "
                      f"precision Δ = {g['d_precision']:+.3f}, recall Δ = {g['d_recall']:+.3f} → "
                      f"**{'PASS' if g['pass'] else 'FAIL'}**\n")
            summary["fidelity_gate"] = g
        else:
            md.append(f"- **fidelity:** `{fid}` present but has no gate at {args.step} steps.\n")
    else:
        md.append(f"- **fidelity:** not yet run (`{fid}` absent) — run `phaseC/fidelity_eval.py`.\n")
    div = Path(args.diversity)
    if div.exists():
        dj = json.loads(div.read_text())
        g = dj.get("gate", {}).get(str(args.step), {})
        if g:
            for metric, v in g.items():
                md.append(f"- **diversity ({metric}):** Δ({TREAT}−{CTRL}) = {v['delta']:+.4f} "
                          f"CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] → {v['verdict']}\n")
            summary["diversity_gate"] = g
        else:
            md.append(f"- **diversity:** `{div}` present but has no gate at {args.step} steps.\n")
    else:
        md.append(f"- **diversity:** not yet run (`{div}` absent) — run "
                  f"`phaseC/diversity_eval.py`.\n")

    # Read footer — the only decision-specific prose; branch on the pair so the C1 verdict
    # (B4 vs B2) reproduces byte-for-byte while D1 (B5_latent vs B2) gets its own guidance.
    if (TREAT, CTRL) == ("B4", "B2"):
        md.append("\n## Read\n- **PRIMARY gate PASS + no diversity collapse / fidelity loss ⇒ H1 "
                  "supported** → run B5 (mechanism), B6, B3.\n"
                  "- NULL ⇒ pre-specified 2× budget retry, else pivot.\n")
    elif (TREAT, CTRL) == ("B5_latent", "B2"):
        md.append("\n## Read (Phase D — H_mech: does target-modification amortize where "
                  "input-selection did not?)\n"
                  f"- **PRIMARY gate PASS ({TREAT} > {CTRL}) while Phase-C1 gave B4 ≈ B2 on this "
                  "same harness ⇒ H_mech supported**: the axis carrying alignment is "
                  "target-modification, not input-selection → run D2 (reward-weighted, H_nov) and "
                  "D3 (latent verifier, H_deploy).\n"
                  "- **NULL ⇒ H_mech not supported**: one pre-specified 2× budget extension of D1; "
                  "if still null, target-modification also does not amortize → escalate to "
                  "differentiable reward (§7) or stop.\n"
                  "- As in C1: a large in-objective VQA gain with T≈0 is VQA-overfit, not a real "
                  "gain; and a pass needs no fidelity loss / no diversity collapse.\n")
    else:
        md.append(f"\n## Read\n- Gate PASS ⇒ {TREAT} > {CTRL} on the pre-registered primary "
                  "(+ no fidelity loss / diversity collapse). NULL ⇒ CI spans 0.\n")
    md.append("- Per Deviations 1 and 3: **`spatial` alone** is zero-inflated (~¼ effective N — "
              "only 2D_spatial_eval hard-thresholds <0.5→0); if it is individually n.s. at pilot "
              "budget that is expected, and its verdict defers to the full-scale run. "
              "`3d_spatial` and `numeracy` were measured to be well spread (≤1/40 pairs tied at "
              "zero), so they are NOT excused the same way — 6 of the 7 primary categories are "
              "adequately powered.\n"
              "- Per Deviation 2: `non_spatial` (CLIPScore) is secondary — its scale is ~10× "
              "compressed relative to BLIP-VQA, so it is reported, never pooled into the primary.\n")
    Path(args.out).write_text("".join(md))
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    if prim_present:
        print(f"  PRIMARY-1 CompBench {TREAT}−{CTRL} = {prim:+.4f} [{plo:+.4f},{phi:+.4f}]  "
              f"→ {summary['primary']['gate']}  ({len(prim_present)} cats)")
    if "geneval2" in summary:
        print(f"  PRIMARY-2 GenEval2   {TREAT}−{CTRL} = {summary['geneval2']['delta']:+.4f} "
              f"→ {summary['geneval2']['gate']}")


if __name__ == "__main__":
    main()
