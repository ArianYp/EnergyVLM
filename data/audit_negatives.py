#!/usr/bin/env python3
"""Audit a negatives file for edits that would confuse the ranking: paraphrases / compatible
descriptions (the photo does not contradict them), attribute words used as nouns or in fixed
phrases, pronoun "one", count edits that break number agreement, verb edits on inanimate subjects.

    python3 data/audit_negatives.py cache/negatives_3k.json [--sample 30]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.build_negatives import ANIMATE, COLOR_WORDS, MATERIALS, QUAL_ANT, SIZE_ANT, _words  # noqa: E402

SIZE = set(SIZE_ANT)
ANT = [{"big", "small"}, {"tall", "short"}, {"long", "short"}, {"wide", "narrow"}, {"large", "small"}, {"little", "big"}, {"huge", "tiny"}]


def checks():
    return {
        "colour in 'black and white' or as a noun / last word": lambda p, n: n["family"] == "color" and (
            re.search(r"black[- ]+and[- ]+white|white[- ]+and[- ]+black", p, re.I) is not None
            or re.search(r"\b" + re.escape(n["source"]) + r"\s*[.,;!]?\s*$", p, re.I) is not None
            or re.search(r"\b(an|a|the|of|some)\s+" + re.escape(n["source"]) + r"\s*([.,;]|$|\s+(and|or|with|on|in)\b)", p, re.I) is not None),
        "'one' as pronoun": lambda p, n: n["family"] == "count" and n["source"].lower() == "one" and re.search(
            r"\b(no|the|this|that|another|each|every|which|any|only|some|other)\s+one\b|\bone\s+(of|another|more|day|way|side|end|hand|time)\b", p, re.I) is not None,
        "count crosses one/many without plural change": lambda p, n: n["family"] == "count" and (n["source"].lower() == "one") != (n["target"].lower() == "one")
            and (n["target"].lower() == "one" or n["prompt"].lower().count("s ") + n["prompt"].lower().endswith("s") <= p.lower().count("s ") + p.lower().endswith("s")),
        "size edit that is not an antonym": lambda p, n: n["family"] == "shape" and (n["source"].lower() in SIZE or n["target"].lower() in SIZE)
            and {n["source"].lower(), n["target"].lower()} not in ANT,
        "shape noun (square/diamond/...) or not attributive": lambda p, n: n["family"] == "shape" and re.search(
            r"\b" + re.escape(n["source"]) + r"\s*([.,;]|$|\s+(and|or|with|of|on|in|at|to|that|is|are)\b)", p, re.I) is not None,
        "texture: material<->quality or non-antonym quality": lambda p, n: n["family"] == "texture" and not (
            (n["source"].lower() in MATERIALS and n["target"].lower() in MATERIALS)
            or (n["source"].lower() in QUAL_ANT and n["target"].lower() in QUAL_ANT[n["source"].lower()])),
        "texture word as noun ('glass of', last word)": lambda p, n: n["family"] == "texture" and re.search(
            r"\b" + re.escape(n["source"]) + r"\s*([.,;]|$|\s+(of|and|or|with|on|in|at|to|that|is|are)\b)", p, re.I) is not None
            and not re.search(r"\bof\s+" + re.escape(n["source"]) + r"\b", p, re.I),
        "verb edit without an animate subject before it": lambda p, n: n["family"] == "verb" and not (
            set(_words(p[:p.lower().find(n["source"].lower())])) & ANIMATE),
        "edit produced an identical or empty prompt": lambda p, n: n["prompt"].strip() == p.strip() or not n["prompt"].strip(),
        "target word already in the caption": lambda p, n: n["target"].lower() in _words(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    d = json.load(open(args.path))
    negs = d["negatives"]
    rows = [(i, v["prompt"], n) for i, v in negs.items() for n in v["negatives"]]
    print(json.dumps(d["summary"], indent=1))
    print(f"{len(rows)} negatives over {len(negs)} captions")
    total_flag = 0
    for name, f in checks().items():
        hits = [(p, n) for _, p, n in rows if f(p, n)]
        total_flag += len(hits)
        print(f"\n## {name}: {len(hits)} ({100 * len(hits) / max(len(rows), 1):.2f}%)")
        for p, n in hits[:args.show]:
            print(f"   {p}\n      ==> [{n['family']}: {n['source']} -> {n['target']}] {n['prompt']}")
    print(f"\nflagged (with overlaps): {total_flag} of {len(rows)}")
    rng = random.Random(1)
    print(f"\n## random sample of {args.sample}")
    for _, p, n in rng.sample(rows, min(args.sample, len(rows))):
        print(f"   ({n['family']:10s} {n['source']} -> {n['target']})\n      {p}\n      {n['prompt']}")


if __name__ == "__main__":
    main()
