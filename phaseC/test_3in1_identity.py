#!/usr/bin/env python3
"""
Proof that `compbench_eval._compose_3_in_1` is the official 3-in-1 weighting.

Why this test exists. Every other category in the pre-registered primary metric is
scored by a vendored evaluator run byte-identical to upstream, so the number is the
official one by construction. `complex` is the exception: the upstream composer
`T2I-CompBench/3_in_1_eval/3_in_1.py` cannot be invoked on our layout at all — it
hardcodes `num=10` images per prompt and indexes the three sub-score arrays by line
position in `complex_val.txt`, while the pre-registered protocol generates 1 image
per prompt over a 300-prompt subset. Feeding it our arrays leaves `total_score`
zero-filled.

So the weighting is reimplemented in the driver, and this test closes the gap the
reimplementation opens: it runs the **real upstream script** and checks our function
returns the same number for every prompt.

The upstream composer touches no images and no models — it reads three JSONs and
three prompt lists. That makes an exact comparison possible without a GPU: synthesise
sub-scores, broadcast each prompt's value across the 10 slots upstream expects (which
is what 1-image-per-prompt means in its indexing), run upstream, and compare
prompt-by-prompt against our composition of the same synthetic values.

This exercises the routing too: all 300 complex_val prompts, so every
spatial / action / 3-way branch upstream can take is compared, not just the arithmetic.

The comparison is necessarily val-only — upstream reads `complex_val*.txt` unconditionally,
with no split argument, so it cannot be pointed at complex_train. `--routing_split train`
in the driver therefore gets the weaker check in part 2: that the train routing lists
resolve every train prompt into exactly one branch.

    python phaseC/test_3in1_identity.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compbench_eval import COMPBENCH, _complex_routing, _compose_3_in_1  # noqa: E402

UPSTREAM_NUM = 10          # 3_in_1.py's hardcoded images-per-prompt
DATASET = COMPBENCH / "examples" / "dataset"


def write_result(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"question_id": i, "answer": float(v)} for i, v in enumerate(values)]))


def check_routing_coverage(split: str) -> None:
    """Every prompt resolves to exactly one branch, and the lists don't overlap."""
    prompts = [ln.strip() for ln in
               (DATASET / f"complex_{split}.txt").read_text().splitlines() if ln.strip()]
    spatial_set, action_set = _complex_routing(DATASET, split)
    both = spatial_set & action_set
    branches = {}
    for p in prompts:
        _, br = _compose_3_in_1(0.0, 0.0, 0.0, p, spatial_set, action_set)
        branches[br] = branches.get(br, 0) + 1
    print(f"  {split:5s}: {len(prompts)} prompts -> {branches}"
          f"{'  OVERLAP=' + str(len(both)) if both else ''}")
    if both:
        sys.exit(f"FAIL — {len(both)} prompts are in BOTH the spatial and action "
                 f"routing lists for split={split}; branch order would decide silently")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()
    args.split = "val"          # upstream hardcodes complex_val*.txt; see module docstring

    prompts = [ln.strip() for ln in
               (DATASET / f"complex_{args.split}.txt").read_text().splitlines() if ln.strip()]
    n = len(prompts)
    rng = np.random.default_rng(args.seed)
    # one synthetic value per prompt per sub-evaluator, in the sub-evaluators' own ranges
    attr = rng.uniform(0, 1, n)                                   # BLIP-VQA
    spat = np.where(rng.uniform(0, 1, n) < 0.7, 0.0, rng.uniform(0.5, 1, n))  # UniDet, zero-inflated
    act = rng.uniform(0.15, 0.4, n)                               # CLIPScore cosine

    with tempfile.TemporaryDirectory(prefix="test_3in1_") as td:
        out = Path(td)
        # upstream reads 10 consecutive entries per prompt; 1 image/prompt = the same
        # value in all 10 slots, which isolates the weighting from the averaging.
        for rel, vals in (("annotation_blip", attr),
                          ("labels/annotation_obj_detection_2d", spat),
                          ("annotation_clip", act)):
            write_result(out / rel / "vqa_result.json", np.repeat(vals, UPSTREAM_NUM))

        script = COMPBENCH / "3_in_1_eval" / "3_in_1.py"
        proc = subprocess.run(
            [sys.executable, script.name, "--outpath", str(out), "--data_path", str(DATASET)],
            cwd=script.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"upstream 3_in_1.py failed ({proc.returncode}):\n{proc.stderr}")
        upstream = np.array([r["answer"] for r in
                             json.loads((out / "annotation_3_in_1" / "vqa_result.json").read_text())])

    if len(upstream) != n * UPSTREAM_NUM:
        sys.exit(f"upstream emitted {len(upstream)} scores, expected {n * UPSTREAM_NUM}")

    spatial_set, action_set = _complex_routing(DATASET, args.split)
    ours = np.empty(n)
    branches = {}
    for i, p in enumerate(prompts):
        ours[i], br = _compose_3_in_1(attr[i], spat[i], act[i], p, spatial_set, action_set)
        branches[br] = branches.get(br, 0) + 1

    theirs = upstream[::UPSTREAM_NUM]                        # first slot of each prompt
    # all 10 slots must be identical, or the broadcast assumption is wrong
    spread = float(np.abs(upstream.reshape(n, UPSTREAM_NUM) - theirs[:, None]).max())
    err = np.abs(ours - theirs)
    worst = int(err.argmax())

    print(f"part 1 — identity vs upstream 3_in_1.py | split={args.split} | {n} prompts")
    print(f"  branches taken: {branches}")
    print(f"  within-prompt spread upstream: {spread:.3e} (must be 0)")
    print(f"  max |ours - upstream|: {err.max():.3e}  (tol {args.tol:g})")
    print(f"  worst prompt: {prompts[worst]!r}")
    print(f"    ours={ours[worst]:.12f} upstream={theirs[worst]:.12f}")
    if err.max() > args.tol or spread > 0:
        sys.exit("FAIL — composition does NOT reproduce upstream 3_in_1.py")
    print(f"  PASS — identical to upstream on all {n} prompts")

    print("part 2 — routing coverage (upstream can't be run on train; lists checked directly)")
    for split in ("val", "train"):
        check_routing_coverage(split)
    print("  PASS — every prompt resolves to exactly one branch, no list overlap")


if __name__ == "__main__":
    main()
