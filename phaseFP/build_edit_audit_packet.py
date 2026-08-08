#!/usr/bin/env python3
"""Task H, stage 3 — assemble the blinded human review packet.

Combines the text sheet with the rendered training pair so one pass answers both questions the
roadmap asks: is the edit valid, and does the frozen pair actually differ in the edited concept.

Blinding is preserved throughout. The reviewer never sees which prompt is the original, which image
is the preferred one, the VQAScore values, or whether the record is in the reversal subset — all of
that lives only in the answer key, which the packet does not link to.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

QUESTIONS = [
    ("q1_single_atom", "Do prompts A and B differ in exactly ONE compositional atom?", "yes / no"),
    ("q2_semantic_change", "Is that a real semantic change (not a synonym or no-op)?", "yes / no"),
    ("q3_coherent", "Is prompt B coherent and physically plausible?", "yes / no"),
    ("q4_pair_differs", "Do images LEFT and RIGHT visibly differ in the edited concept?", "yes / no / unclear"),
    ("q5_which_matches_A", "Which image better matches prompt A?", "left / right / neither"),
    ("q6_notes", "Notes", "free text"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--answer_key", required=True)
    parser.add_argument("--review_sheet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    key = {row["item_id"]: row for row in json.loads(Path(args.answer_key).read_text())}
    sheet = json.loads(Path(args.review_sheet).read_text())
    pairs_root = Path(args.pairs_root)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    items, rows = [], []
    for entry in sheet:
        item_id = entry["item_id"]
        k = key.get(item_id)
        if k is None:
            continue
        idx = int(k["idx"])
        directory = pairs_root / f"p{idx:05d}"
        a_img, b_img = directory / "pair_a.png", directory / "pair_b.png"
        if not (a_img.exists() and b_img.exists()):
            continue
        # Independently shuffle which rendered image is shown left, so image side carries no signal
        # about which candidate was preferred.
        flip = bool(hashlib.blake2b(
            f"pairside:{args.seed}:{item_id}".encode(), digest_size=1).digest()[0] & 1)
        left, right = (b_img, a_img) if flip else (a_img, b_img)
        items.append({
            "item_id": item_id,
            "prompt_A": entry["prompt_A"],
            "prompt_B": entry["prompt_B"],
            "left": os.path.relpath(left, out),
            "right": os.path.relpath(right, out),
            "_left_is_pair_b": flip,
        })
        rows.append({"item_id": item_id, "prompt_A": entry["prompt_A"],
                     "prompt_B": entry["prompt_B"],
                     **{q: "" for q, _, _ in QUESTIONS}})

    with open(out / "review_sheet.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "side_key.json").write_text(json.dumps(
        {i["item_id"]: {"left_is_pair_b": i["_left_is_pair_b"]} for i in items}, indent=1))

    html = [
        "<!doctype html><meta charset='utf-8'><title>Counterfactual edit audit</title>",
        "<style>body{font:15px/1.5 system-ui;margin:28px;color:#202124;max-width:1000px}"
        "h1{font-size:21px}.item{border-top:1px solid #cbd0d6;padding:18px 0}"
        ".pid{font-weight:700}.pr{margin:6px 0;font-size:15px}"
        ".imgs{display:flex;gap:14px;margin-top:10px}"
        "img{width:340px;height:340px;object-fit:cover;border:1px solid #cbd0d6}"
        ".lab{font-size:12px;color:#667085;text-align:center}"
        "code{background:#f2f4f7;padding:1px 5px;border-radius:3px}"
        "ol{font-size:14px;color:#344054}</style>",
        "<h1>Counterfactual edit audit</h1>",
        f"<p>{len(items)} items. Fill <code>review_sheet.csv</code>. "
        "<b>A is not always the original prompt, and the left image is not always the first "
        "candidate</b> — both are shuffled per item. Do not open the key files until finished.</p>",
        "<ol>" + "".join(f"<li><b>{q}</b> — {t} <i>({o})</i></li>" for q, t, o in QUESTIONS)
        + "</ol>",
    ]
    for i in items:
        html.append(
            f"<div class='item'><div class='pid'>{i['item_id']}</div>"
            f"<div class='pr'>A: {i['prompt_A']}</div>"
            f"<div class='pr'>B: {i['prompt_B']}</div>"
            f"<div class='imgs'><div><img src='{i['left']}' alt='left'>"
            f"<div class='lab'>LEFT</div></div>"
            f"<div><img src='{i['right']}' alt='right'>"
            f"<div class='lab'>RIGHT</div></div></div></div>"
        )
    (out / "review.html").write_text("\n".join(html))

    summary = {"items": len(items), "pairs_root": str(pairs_root),
               "questions": [q for q, _, _ in QUESTIONS],
               "blinding": "prompt order and image side both shuffled per item"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
