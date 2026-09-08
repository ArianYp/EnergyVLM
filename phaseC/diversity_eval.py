#!/usr/bin/env python3
"""
Intra-prompt diversity for the Phase C1 gate: does B4 collapse modes to buy alignment?

The pre-registration (§3, §5) requires, for a "clean" H1, that **intra-prompt
LPIPS/DINO of B4 >= B2 within a pre-set tolerance** — a B4−B2 alignment gain obtained
by generating the same image for every seed is not a pass. B4 is trained on
*oracle-selected* trajectories, so mode narrowing is the specific, plausible failure
here: selecting the best of N candidates is exactly a pressure toward one mode.

Measured per prompt over `--n_seeds` images that differ **only** in initial noise
(same prompt, same step count, same model):

  DINO   mean pairwise cosine *distance* (1 − cos) between DINOv2 CLS embeddings —
         semantic/layout variation. Same feature family as Exp-0's candidate-diversity
         check, so the numbers are readable next to it.
  LPIPS  mean pairwise LPIPS (AlexNet) — perceptual/appearance variation.

Both are per-prompt, so B4 vs B2 is **paired over prompts** with the same
10k-bootstrap as the alignment analysis, and the same seeds are used for every model
(the generator derives candidate j's noise from the prompt index, not from the model),
which makes the comparison within-prompt and within-seed-set.

The gate is one-sided and expressed as a *non-inferiority* test: B4 fails only if its
diversity is *lower* than B2's by more than the tolerance, i.e. if the upper end of
the bootstrap CI for (B4 − B2) sits below −tol.

Usage
  python phaseC/diversity_eval.py --gen_root phaseC/diversity/images \
      --models B4,B2,M1 --steps 4 --n_seeds 4 --out phaseC/diversity_report.md
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BOOT = 10_000


@torch.no_grad()
def dino_embed(paths, model, proc, device, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        h = model(pixel_values=px).last_hidden_state[:, 0]          # CLS token
        out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def lpips_pairwise(paths, net, device):
    """Mean pairwise LPIPS over a small set of images (n_seeds is 4-8)."""
    ts = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        ts.append(torch.from_numpy(a).permute(2, 0, 1))
    x = torch.stack(ts).to(device)
    i, j = zip(*combinations(range(len(paths)), 2))
    d = net(x[list(i)], x[list(j)]).squeeze()
    return float(d.mean())


def boot_ci(vals, rng, reps=BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()) if len(vals) else float("nan")), float("nan"), float("nan")
    idx = rng.integers(0, len(vals), size=(reps, len(vals)))
    m = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="phaseC/diversity/images")
    ap.add_argument("--models", default="B4,B2,M1")
    ap.add_argument("--gate_models", default="B4,B2",
                    help="treatment,control model labels for the non-inferiority gate")
    ap.add_argument("--title", default="Phase C1")
    ap.add_argument("--steps", default="4")
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--prompts", default="phaseC/diversity/prompts.json",
                    help="for per-category breakdown; optional")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="pre-set tolerance: FAIL only if B4 diversity is below B2 by "
                         "more than this (CI upper bound < -tol)")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="phaseC/diversity_report.md")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    rng = np.random.default_rng(args.seed)
    models = args.models.split(",")
    gate_models = args.gate_models.split(",")
    if len(gate_models) != 2:
        raise ValueError("--gate_models must be treatment,control")
    gate_treatment, gate_control = gate_models
    if gate_treatment not in models or gate_control not in models:
        raise ValueError("--gate_models labels must both appear in --models")
    steps = [int(x) for x in args.steps.split(",")]

    import lpips
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.dino_id)
    dino = AutoModel.from_pretrained(args.dino_id, torch_dtype=torch.float16).to(device).eval()
    lp = lpips.LPIPS(net="alex").to(device).eval()

    cat_of = {}
    if Path(args.prompts).exists():
        cat_of = {it["idx"]: it["category"] for it in json.loads(Path(args.prompts).read_text())}

    root = Path(args.gen_root)
    # per (model, step): {idx: {"dino": d, "lpips": l}}
    per = {m: {s: {} for s in steps} for m in models}
    for m in models:
        for s in steps:
            pdirs = sorted((root / m).glob("p*"))
            n_short = 0
            for pd in pdirs:
                files = [pd / f"s{s}" / f"cand{j}.png" for j in range(args.n_seeds)]
                if not all(f.exists() for f in files):
                    n_short += 1
                    continue
                idx = int(pd.name[1:])
                emb = dino_embed([str(f) for f in files], dino, proc, device)
                cos = emb @ emb.T
                iu = torch.triu_indices(len(files), len(files), offset=1)
                per[m][s][idx] = {"dino": float(1.0 - cos[iu[0], iu[1]].mean()),
                                  "lpips": lpips_pairwise([str(f) for f in files], lp, device)}
            print(f"  {m}@{s}: {len(per[m][s])} prompts with {args.n_seeds} seeds"
                  f"{f' ({n_short} incomplete, skipped)' if n_short else ''}", flush=True)

    md = [f"# {args.title} — intra-prompt diversity (mode-collapse gate)\n\n",
          f"`{args.n_seeds}` images per prompt differing only in initial noise | "
          f"DINOv2 `{args.dino_id}` mean pairwise cosine distance | LPIPS (AlexNet) mean "
          f"pairwise | paired over prompts, {args.boot} bootstrap\n\n",
          "Pre-registered role: a **gate**. A B4−B2 alignment gain accompanied by a "
          "diversity collapse is not a pass.\n\n"]

    summary = {
        "n_seeds": args.n_seeds, "tol": args.tol, "gate_models": gate_models,
        "levels": {}, "gate": {},
    }
    md.append("## diversity level (higher = more varied across seeds)\n\n"
              "| model@steps | n prompts | DINO 1−cos | 95% CI | LPIPS | 95% CI |\n"
              "|---|--:|--:|---|--:|---|\n")
    for m in models:
        for s in steps:
            if not per[m][s]:
                continue
            dv = [v["dino"] for v in per[m][s].values()]
            lv = [v["lpips"] for v in per[m][s].values()]
            dm, dlo, dhi = boot_ci(dv, rng, args.boot)
            lm, llo, lhi = boot_ci(lv, rng, args.boot)
            summary["levels"][f"{m}@{s}"] = {"n": len(dv), "dino": dm, "dino_ci": [dlo, dhi],
                                             "lpips": lm, "lpips_ci": [llo, lhi]}
            md.append(f"| {m}@{s} | {len(dv)} | {dm:.4f} | [{dlo:.4f}, {dhi:.4f}] | "
                      f"{lm:.4f} | [{llo:.4f}, {lhi:.4f}] |\n")

    md.append(
        f"\n## Gate: {gate_treatment} − {gate_control}, paired "
        f"(FAIL only if CI upper bound < −{args.tol:g})\n\n"
        f"| steps | metric | Δ{gate_treatment}−{gate_control} | 95% CI | n | verdict |\n"
        "|---|---|--:|---|--:|---|\n"
    )
    for s in steps:
        a = per.get(gate_treatment, {}).get(s, {})
        b = per.get(gate_control, {}).get(s, {})
        ks = sorted(set(a) & set(b))
        if not ks:
            continue
        for metric in ("dino", "lpips"):
            d = [a[k][metric] - b[k][metric] for k in ks]
            dm, lo, hi = boot_ci(d, rng, args.boot)
            if hi < -args.tol:
                verdict = "**FAIL — diversity collapse**"
            elif lo > -args.tol:
                verdict = "PASS (non-inferior)"
            else:
                verdict = "inconclusive (CI spans −tol)"
            summary["gate"].setdefault(str(s), {})[metric] = {
                "delta": dm, "ci": [lo, hi], "n": len(d), "verdict": verdict}
            md.append(f"| {s} | {metric} | {dm:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                      f"{len(d)} | {verdict} |\n")

    # per-category, for whether any collapse is localised
    if cat_of:
        md.append(f"\n## {gate_treatment} − {gate_control} by category (exploratory)\n\n"
                  "| steps | category | Δ DINO | Δ LPIPS | n |\n|---|---|--:|--:|--:|\n")
        for s in steps:
            a = per.get(gate_treatment, {}).get(s, {})
            b = per.get(gate_control, {}).get(s, {})
            ks = sorted(set(a) & set(b))
            bycat = {}
            for k in ks:
                bycat.setdefault(cat_of.get(k, "?"), []).append(
                    (a[k]["dino"] - b[k]["dino"], a[k]["lpips"] - b[k]["lpips"]))
            for c, vs in sorted(bycat.items()):
                md.append(f"| {s} | {c} | {np.mean([v[0] for v in vs]):+.4f} | "
                          f"{np.mean([v[1] for v in vs]):+.4f} | {len(vs)} |\n")

    md.append("\n## Read\n"
              "- **PASS on both metrics** ⇒ the diversity half of the pre-registered gate is met.\n"
              "- **FAIL on either** ⇒ H1 is not clean even if the alignment primary passes: the "
              "gain was bought by mode collapse. Report as such.\n"
              "- DINO down but LPIPS flat ⇒ narrowing of *content/layout* while appearance still "
              "varies; the more relevant collapse for a compositional claim.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(md))
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {**summary, "per_prompt": {m: {str(s): per[m][s] for s in steps} for m in models}},
        indent=1))
    print(f"\nwrote {args.out}")
    for s in steps:
        for metric, g in summary["gate"].get(str(s), {}).items():
            print(f"  s{s} {metric}: Δ={g['delta']:+.4f} [{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] "
                  f"{g['verdict']}")


if __name__ == "__main__":
    main()
