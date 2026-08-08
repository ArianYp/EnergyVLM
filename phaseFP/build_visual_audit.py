#!/usr/bin/env python3
"""Task V — non-cherry-picked visual audit of the fixed-pair arms.

The selection rule is mechanical and fixed in advance, because a visual audit whose examples were
chosen after looking at them is worthless. Two populations are emitted:

  `random`   a seeded uniform sample per category — chosen WITHOUT reference to any score, so it
             shows what the arms typically look like, including the boring cases.
  `stratified` per category, the prompts where (correct - M1) is largest, near zero, and most
             negative — i.e. explicit wins, ties and FAILURES. The losses are the point: a grid
             containing only wins would be cherry-picking with extra steps.

Emits a self-contained HTML contact sheet plus a blinded rating sheet in which arm identity is
hidden behind per-row shuffled column labels, so a rater cannot track which column is `correct`.
"""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

PRIMARY = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex"]


def load_scores(scores_dir: Path, label: str, category: str, step: int = 4) -> dict[int, dict]:
    path = scores_dir / f"{label}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(r["idx"]): r for r in payload["per_prompt"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=LABEL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_category_random", type=int, default=2)
    parser.add_argument("--per_category_stratum", type=int, default=1)
    parser.add_argument("--reference", default="M1", help="arm used to define win/tie/loss")
    parser.add_argument("--treatment", default="correct")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, label = spec.partition("=")
        arms[name] = label
    scores_dir = Path(args.scores_dir)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    rng = random.Random(args.seed)
    selected = []
    for category in PRIMARY:
        loaded = {n: load_scores(scores_dir, lab, category, args.step) for n, lab in arms.items()}
        if any(not v for v in loaded.values()):
            continue
        shared = sorted(set.intersection(*(set(v) for v in loaded.values())))
        if not shared:
            continue

        # (1) Uniform sample, chosen with no reference to any score.
        for idx in rng.sample(shared, min(args.per_category_random, len(shared))):
            selected.append({"idx": idx, "category": category, "stratum": "random"})

        # (2) Outcome-stratified: wins, ties and losses of treatment against reference.
        if args.treatment in loaded and args.reference in loaded:
            delta = {
                i: loaded[args.treatment][i]["score"] - loaded[args.reference][i]["score"]
                for i in shared
            }
            ordered = sorted(shared, key=lambda i: delta[i])
            k = args.per_category_stratum
            mid = len(ordered) // 2
            picks = {
                "loss": ordered[:k],
                "tie": ordered[max(0, mid - k // 2): max(0, mid - k // 2) + k],
                "win": ordered[-k:],
            }
            for stratum, idxs in picks.items():
                for idx in idxs:
                    selected.append({
                        "idx": idx, "category": category, "stratum": stratum,
                        "delta_vs_reference": delta[idx],
                    })

    # Deduplicate, keeping the more informative stratum label.
    seen, rows = {}, []
    for row in selected:
        key = (row["category"], row["idx"])
        if key in seen and seen[key]["stratum"] != "random":
            continue
        seen[key] = row
    rows = list(seen.values())
    rows.sort(key=lambda r: (r["category"], r["stratum"], r["idx"]))

    names = list(arms)
    manifest, rating = [], []
    for position, row in enumerate(rows):
        category, idx = row["category"], row["idx"]
        entry = dict(row)
        entry["item_id"] = f"V{position:04d}"
        per_arm = {}
        for name, label in arms.items():
            rec = load_scores(scores_dir, label, category, args.step).get(idx)
            if rec is None:
                continue
            per_arm[name] = {"score": rec["score"], "src": rec["src"], "prompt": rec["prompt"]}
        if len(per_arm) != len(arms):
            continue
        entry["prompt"] = next(iter(per_arm.values()))["prompt"]
        entry["arms"] = per_arm
        # Per-row column order, deterministic but unpredictable, so a rater cannot learn a position.
        order = sorted(
            names,
            key=lambda n: hashlib.blake2b(
                f"{args.seed}:{entry['item_id']}:{n}".encode(), digest_size=8
            ).digest(),
        )
        entry["column_order"] = order
        manifest.append(entry)
        rating.append({
            "item_id": entry["item_id"],
            "prompt": entry["prompt"],
            "category": category,
            **{f"col{c + 1}_best_match": "" for c in range(len(order))},
            "which_column_best": "",
            "notes": "",
        })

    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "key.json").write_text(json.dumps(
        {e["item_id"]: {"column_order": e["column_order"], "stratum": e["stratum"],
                        "scores": {k: v["score"] for k, v in e["arms"].items()}}
         for e in manifest}, indent=1))
    with open(out / "rating_sheet.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rating[0].keys()))
        w.writeheader()
        w.writerows(rating)

    # Contact sheet. Images are referenced by relative path; the HTML lives beside them.
    html = [
        "<!doctype html><meta charset='utf-8'><title>Phase-FP visual audit</title>",
        "<style>body{font:14px/1.4 system-ui;margin:24px;color:#202124}"
        "h1{font-size:20px}h2{font-size:15px;margin:26px 0 6px}"
        "table{border-collapse:collapse;margin-bottom:18px}"
        "td,th{border:1px solid #cbd0d6;padding:4px;text-align:center;vertical-align:top}"
        "img{width:190px;height:190px;object-fit:cover;display:block}"
        ".p{font-size:12px;color:#667085;max-width:780px;margin:2px 0 6px}"
        ".s{font-size:11px;color:#667085}.win{color:#27864f}.loss{color:#b43c31}</style>",
        "<h1>Phase-FP visual audit — non-cherry-picked</h1>",
        f"<p class='p'>Selection rule fixed in advance (seed {args.seed}): "
        f"{args.per_category_random} uniform samples per category chosen without reference to any "
        f"score, plus {args.per_category_stratum} each of the largest win, nearest tie and largest "
        f"<b>loss</b> of <code>{args.treatment}</code> against <code>{args.reference}</code>. "
        "Column order is shuffled per row; see key.json.</p>",
    ]
    by_stratum = defaultdict(list)
    for e in manifest:
        by_stratum[e["stratum"]].append(e)
    for stratum in ("win", "tie", "loss", "random"):
        items = by_stratum.get(stratum, [])
        if not items:
            continue
        html.append(f"<h2>{stratum} ({len(items)} items)</h2>")
        for e in items:
            html.append(f"<div class='p'><b>{e['item_id']}</b> [{e['category']}] "
                        f"{e['prompt']}</div><table><tr>")
            for c, name in enumerate(e["column_order"]):
                html.append(f"<th>column {c + 1}</th>")
            html.append("</tr><tr>")
            for name in e["column_order"]:
                src = os.path.relpath(e["arms"][name]["src"], out)
                html.append(f"<td><img src='{src}' alt='{e['item_id']} {name}'></td>")
            html.append("</tr><tr>")
            for name in e["column_order"]:
                html.append(f"<td class='s'>score {e['arms'][name]['score']:.3f}</td>")
            html.append("</tr></table>")
    (out / "contact_sheet.html").write_text("\n".join(html))

    summary = {
        "items": len(manifest),
        "by_stratum": {k: len(v) for k, v in by_stratum.items()},
        "arms": arms,
        "selection_rule": (
            f"seed {args.seed}; {args.per_category_random} uniform per category (score-blind) plus "
            f"{args.per_category_stratum} each of win/tie/loss of {args.treatment} vs "
            f"{args.reference}; column order shuffled per row"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
