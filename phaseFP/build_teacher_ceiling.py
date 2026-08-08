#!/usr/bin/env python3
"""Task T, stage 3 — lay out teacher best-of-K and control selections for the official evaluators.

Emits two "models" of symlinks in the layout `phaseA_generate.py` produces, so the existing
CompBench harness scores them with no special-casing and no image is copied.

Two control definitions are emitted, because they answer different questions and the existing
+0.0717 figure used the first:

  `TeacherRandom`    a random NON-oracle of the K candidates — matches `exp0/build_primary_headroom.py`,
                     so the new val-split number is comparable with the old train-split one.
  `TeacherUniform`   a uniform draw over all K, including the oracle — the unbiased "what you get
                     without selection" baseline, and the correct denominator for an amortization
                     ratio. The non-oracle control slightly inflates the measured headroom.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = {int(p["idx"]): p for p in json.loads(Path(args.prompts).read_text())}
    per: dict[int, list[float]] = {}
    for shard in sorted(Path(args.scores_dir).glob("scores_rank*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            vqa = row.get("vqa")
            if isinstance(vqa, (list, tuple)) and len(vqa) >= args.N:
                per[int(row["idx"])] = [float(x) for x in vqa[: args.N]]

    out = Path(args.out)
    images_root = Path(args.images_root)
    rng = random.Random(args.seed)

    selection = []
    made = defaultdict(int)
    for idx in sorted(per):
        scores = per[idx]
        oracle = max(range(args.N), key=scores.__getitem__)
        non_oracle = [j for j in range(args.N) if j != oracle]
        control = rng.choice(non_oracle)
        uniform = rng.randrange(args.N)
        picks = {"TeacherBest": oracle, "TeacherRandom": control, "TeacherUniform": uniform}
        ok = True
        for label, j in picks.items():
            src = images_root / f"p{idx:05d}" / f"s{args.steps}" / f"cand{j}.png"
            if not src.exists():
                ok = False
                break
        if not ok:
            continue
        for label, j in picks.items():
            src = (images_root / f"p{idx:05d}" / f"s{args.steps}" / f"cand{j}.png").resolve()
            dst_dir = out / "images" / label / f"p{idx:05d}" / f"s{args.steps}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "cand0.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            made[label] += 1
        selection.append({
            "idx": idx,
            "category": prompts[idx].get("category"),
            "prompt": prompts[idx]["prompt"],
            "vqa": scores,
            "oracle_j": oracle,
            "random_j": control,
            "uniform_j": uniform,
            "vqa_oracle": scores[oracle],
            "vqa_random": scores[control],
            "vqa_uniform": scores[uniform],
            "vqa_mean": sum(scores) / len(scores),
        })

    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(selection, indent=1))
    # The evaluators need a prompt manifest restricted to the prompts we actually laid out.
    kept = {r["idx"] for r in selection}
    (out / "prompts.json").write_text(json.dumps(
        [p for i, p in sorted(prompts.items()) if i in kept], indent=1))

    n = len(selection)
    summary = {
        "prompts": n,
        "N": args.N,
        "steps": args.steps,
        "symlinks_per_model": dict(made),
        "in_objective_vqa": {
            "oracle_minus_uniform": sum(r["vqa_oracle"] - r["vqa_uniform"] for r in selection) / n,
            "oracle_minus_random_nonoracle": sum(
                r["vqa_oracle"] - r["vqa_random"] for r in selection) / n,
            "oracle_minus_candidate_mean": sum(
                r["vqa_oracle"] - r["vqa_mean"] for r in selection) / n,
        },
        "note": (
            "These VQAScore gaps are in-objective (selector == scorer) and are reported only as a "
            "sanity check that selection did something. The load-bearing numbers come from the "
            "official CompBench evaluators run over these symlinks."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
