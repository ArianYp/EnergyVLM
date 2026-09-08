#!/usr/bin/env python3
"""
Prompt pools for the two pre-registered eval sets that are not T2I-CompBench.

Both emit the same `[{idx, category, prompt}, ...]` shape `exp0/phaseA_generate.py`
consumes via `--prompts_json`, so generation for GenEval2 and for the fidelity set
reuses the already-validated paired-seed generator instead of a second code path.

  geneval2  GenEval2/geneval2_data.jsonl, in file order (800 prompts). The order is
            load-bearing: GenEval2/evaluation.py emits one score list per line of the
            benchmark file with no key, so `idx == line number` is what lets the
            per-prompt scores be joined back. `category` carries the atom skills.

  coco      COCO val2017 captions, one caption per image, for FID/CMMD. The
            reference side is the *real* val2017 images, so taking one caption per
            image keeps generated and reference sets the same size and the same
            underlying scene distribution.

  subset    a stratified subset of an existing pool, for the multi-seed diversity run
            (which costs n_seeds images per prompt, so it runs on fewer prompts).
            **`idx` is preserved**, not renumbered: the generator derives a prompt's
            noise from its `idx`, so keeping it means candidate 0 of the diversity run
            is the very image the alignment eval scored.

    python phaseC/build_eval_pool.py geneval2 --out phaseC/geneval2/prompts.json
    python phaseC/build_eval_pool.py coco --n 5000 --out phaseC/fidelity/prompts.json
    python phaseC/build_eval_pool.py subset --pool phaseC/eval/prompts.json \
        --per_category 50 --out phaseC/diversity/prompts.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def build_geneval2(args) -> list[dict]:
    rows = [json.loads(ln) for ln in
            Path(args.data).read_text().splitlines() if ln.strip()]
    pool = []
    for i, r in enumerate(rows):
        pool.append({"idx": i, "category": "geneval2", "prompt": r["prompt"],
                     "skills": sorted(set(r.get("skills", []))),
                     "atom_count": r.get("atom_count")})
    dup = len(pool) - len({p["prompt"] for p in pool})
    if dup:
        raise SystemExit(f"{dup} duplicate prompts — evaluation.py keys images by prompt "
                         f"string, so duplicates would collide")
    return pool


def build_coco(args) -> list[dict]:
    ann = json.loads(Path(args.captions).read_text())
    # one caption per image, lowest caption id, so the choice is deterministic and
    # independent of the json's internal ordering
    first: dict[int, tuple[int, str]] = {}
    for a in ann["annotations"]:
        cur = first.get(a["image_id"])
        if cur is None or a["id"] < cur[0]:
            first[a["image_id"]] = (a["id"], a["caption"].strip().replace("\n", " "))
    items = [(img_id, cap) for img_id, (_, cap) in sorted(first.items())]
    rng = random.Random(args.seed)
    if args.n and args.n < len(items):
        items = rng.sample(items, args.n)
        items.sort()
    return [{"idx": i, "category": "coco", "prompt": cap, "coco_image_id": img_id}
            for i, (img_id, cap) in enumerate(items)]


def build_subset(args) -> list[dict]:
    pool = json.loads(Path(args.pool).read_text())
    bycat: dict[str, list[dict]] = {}
    for it in pool:
        bycat.setdefault(it["category"], []).append(it)
    rng = random.Random(args.seed)
    out = []
    for cat in sorted(bycat):
        items = sorted(bycat[cat], key=lambda it: it["idx"])
        out += rng.sample(items, min(args.per_category, len(items)))
    return sorted(out, key=lambda it: it["idx"])       # idx preserved, NOT renumbered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="which", required=True)

    g = sub.add_parser("geneval2")
    g.add_argument("--data", default=str(REPO / "GenEval2" / "geneval2_data.jsonl"))
    g.add_argument("--out", default="phaseC/geneval2/prompts.json")
    g.set_defaults(func=build_geneval2)

    c = sub.add_parser("coco")
    c.add_argument("--captions",
                   default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO/annotations/captions_val2017.json")
    c.add_argument("--n", type=int, default=5000, help="0 = all")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="phaseC/fidelity/prompts.json")
    c.set_defaults(func=build_coco)

    b = sub.add_parser("subset")
    b.add_argument("--pool", default="phaseC/eval/prompts.json")
    b.add_argument("--per_category", type=int, default=50)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="phaseC/diversity/prompts.json")
    b.set_defaults(func=build_subset)

    args = ap.parse_args()
    pool = args.func(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1))
    print(f"{args.which}: {len(pool)} prompts -> {out}")
    for p in pool[:3]:
        print(f"  [{p['idx']}] {p['prompt']!r}")


if __name__ == "__main__":
    main()
