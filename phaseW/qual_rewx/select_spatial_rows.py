#!/usr/bin/env python3
"""Randomly drawn SPATIAL prompts for the student-vs-teacher sheet: N CompBench prompts split evenly
between the 'spatial' and '3d_spatial' categories, and N GenEval2 prompts that contain a 'position'
atom. Uniform draws with a fixed seed from the prompts every evaluated model was scored on.
Writes selection.json (panel 'random') and the subset JSONs for generation.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="phaseN/eval_S4_CD_dinop_hard-rewXi-avglast3_s0_*", help="an evaluated dir: prompt lists and skills come from its score files")
    ap.add_argument("--pool", default="phaseFP/eval_pool_101203"); ap.add_argument("--out_root", default="phaseW/qual_spatial")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    d = sorted(glob.glob(args.eval))[-1]; rng = np.random.default_rng(args.seed)
    cb_pool = {(r["category"], r["prompt"]): int(r["idx"]) for r in json.load(open(f"{args.pool}/compbench_prompts.json"))}
    sel = {}
    rows = []
    for cat in ("spatial", "3d_spatial"):
        items = []
        for p in glob.glob(f"{d}/compbench_scores/*_{cat}/scores.json"):
            for r in json.load(open(p))["per_prompt"]:
                if r["category"] == cat: items.append((cb_pool[(cat, r["prompt"])], r["prompt"]))
        pick = rng.choice(len(items), size=args.n // 2, replace=False)
        rows += [{"idx": items[k][0], "prompt": items[k][1], "category": cat, "panel": "random"} for k in sorted(pick)]
    sel["compbench"] = rows
    g = json.load(open(glob.glob(f"{d}/geneval2_scores/*/scores.json")[0]))["per_prompt"]
    pos = [r for r in g if "position" in r["skills"]]
    pick = rng.choice(len(pos), size=args.n, replace=False)
    sel["geneval2"] = [{"idx": int(pos[k]["idx"]), "prompt": pos[k]["prompt"], "category": f"geneval2 · position · {pos[k]['atom_count']} atoms", "panel": "random"} for k in sorted(pick)]
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(sel, indent=1))
    for b in ("compbench", "geneval2"):
        (root / f"{b}_subset.json").write_text(json.dumps([{"idx": r["idx"], "category": r["category"].split(" ")[0], "prompt": r["prompt"]} for r in sel[b]], indent=1))
        print(b + ": " + "; ".join(f"p{r['idx']:05d} {r['prompt'][:60]}" for r in sel[b]))


if __name__ == "__main__":
    main()
