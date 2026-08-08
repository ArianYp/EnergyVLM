#!/usr/bin/env python3
"""Task H — blinded human audit of the counterfactual prompt edits.

The counterfactual arm's whole meaning depends on an unverified premise: that each rule-based edit
really changes exactly one compositional atom and yields a coherent prompt. If the edits are
degenerate (no semantic change, ungrammatical, or changing more than one atom), a null
correct-versus-counterfactual result would be uninterpretable.

This exports a stratified, shuffled, blinded review sheet. The reviewer never sees which prompt is
the original, which arm a record belongs to, or whether the pair reverses — those live only in the
answer key, which is written separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseI.train_preference import load_selection

QUESTIONS = [
    "q1_single_atom: Do the two prompts differ in exactly ONE compositional atom "
    "(one colour, shape, texture, count, spatial relation, depth relation, or verb)? [yes/no]",
    "q2_semantic_change: Is that difference a real semantic change rather than a synonym or "
    "a no-op? [yes/no]",
    "q3_coherent: Is prompt B a coherent, physically plausible request? [yes/no]",
    "q4_which_atom: Which atom differs? [colour/shape/texture/count/spatial/depth/verb/none/other]",
    "q5_notes: free text",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default="phaseFP/fixedpair_101185")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--per_stratum", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite {out_root}")

    records = load_selection(Path(args.cache_root) / "counterfactual")

    # Stratify by edit family AND reversal status: reversals are what the sharp analysis rests on,
    # and they concentrate on near-ties, so they must not be under-sampled.
    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in records:
        key = (str(record.get("edit_family", "unknown")),
               bool(record.get("pair_reverses_under_counterfactual")))
        strata[key].append(record)

    rng = random.Random(args.seed)
    sampled = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: int(r["idx"]))
        sampled.extend(rng.sample(bucket, min(args.per_stratum, len(bucket))))
    rng.shuffle(sampled)

    sheet, key_rows = [], []
    for position, record in enumerate(sampled):
        original = record["prompt"]
        edited = record["label_prompt"]
        # Blind the presentation order too, so a reviewer cannot learn "A is always the original".
        flip = bool(
            hashlib.blake2b(
                f"editaudit:{args.seed}:{record['idx']}".encode(), digest_size=1
            ).digest()[0] & 1
        )
        prompt_a, prompt_b = (edited, original) if flip else (original, edited)
        item_id = f"E{position:04d}"
        sheet.append({
            "item_id": item_id,
            "prompt_A": prompt_a,
            "prompt_B": prompt_b,
            **{q.split(":")[0]: "" for q in QUESTIONS},
        })
        key_rows.append({
            "item_id": item_id,
            "idx": int(record["idx"]),
            "category": record.get("category"),
            "edit_family": record.get("edit_family"),
            "edit_source": record.get("edit_source"),
            "edit_target": record.get("edit_target"),
            "original_is": "B" if flip else "A",
            "pair_reverses_under_counterfactual": bool(
                record.get("pair_reverses_under_counterfactual")
            ),
            "original_pair_margin": record.get("original_pair_margin"),
            "counterfactual_pair_margin": record.get("counterfactual_pair_margin"),
            "pair_a": record.get("pair_a"),
            "pair_b": record.get("pair_b"),
            "seed_base": record.get("seed_base"),
        })

    out_root.mkdir(parents=True)
    (out_root / "review_sheet.json").write_text(json.dumps(sheet, indent=2))
    (out_root / "answer_key.json").write_text(json.dumps(key_rows, indent=2))

    with open(out_root / "review_sheet.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0].keys()))
        writer.writeheader()
        writer.writerows(sheet)

    instructions = [
        "# Blinded counterfactual edit audit",
        "",
        f"{len(sheet)} prompt pairs, stratified by edit family and by whether the fixed image pair "
        "reverses, then shuffled. Presentation order within each pair is randomised, so **A is not "
        "always the original**.",
        "",
        "Fill `review_sheet.csv` (or the JSON). Do not open `answer_key.json` until every row is "
        "answered.",
        "",
        "## Questions",
        "",
        *[f"- **{q}**" for q in QUESTIONS],
        "",
        "## Why this matters",
        "",
        "The counterfactual training arm orients its preference labels using the edited prompt. If "
        "the edits do not change exactly one atom, or are not real semantic changes, then a null "
        "correct-versus-counterfactual result cannot distinguish 'orientation does not matter' from "
        "'the intervention was too weak to matter'. Report the pass rate per edit family, and "
        "report it separately for reversing and non-reversing records.",
        "",
        "## Sampling",
        "",
        f"- {args.per_stratum} records per (edit family x reversal status) stratum, seed {args.seed}",
        "- Strata present: "
        + ", ".join(f"{family}/{'rev' if rev else 'norev'}" for family, rev in sorted(strata)),
    ]
    (out_root / "INSTRUCTIONS.md").write_text("\n".join(instructions) + "\n")

    summary = {
        "items": len(sheet),
        "per_stratum": args.per_stratum,
        "seed": args.seed,
        "strata_sizes": {f"{family}/{'rev' if rev else 'norev'}": len(bucket)
                         for (family, rev), bucket in sorted(strata.items())},
        "sampled_by_edit_family": dict(Counter(k["edit_family"] for k in key_rows)),
        "sampled_reversing": sum(k["pair_reverses_under_counterfactual"] for k in key_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
