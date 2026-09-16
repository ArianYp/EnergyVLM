#!/usr/bin/env python3
"""Structured negatives for the ranking arms (docs/rank/): rule-based, same-object rearrangements.

For every caption of a candidate cache, up to --m negative prompts, each changing exactly ONE span
of the caption while keeping every noun, so the caption's real photograph stays a valid anchor for
all of them (the design note's "O1 non-oracle" setting: the positive's generation should land closer
to the photo than any negative's). One edit per family first, in the order
    spatial > 3d_spatial > count > color > shape/size > texture > verb
then further values of the same families; a second edit of the same span only when nothing else is
left. --families restricts the families (v3 of the campaign = color,texture,verb,shape,count, after
VQAScore showed spatial / 3d_spatial edits produce no contradiction in the student's images), and
--count_min_delta 2 drops small count changes (two -> three is not realised either).

Rules per family (each from an audit of a first version, docs/rank/REVIEW.md):
  color    attributive use only ("a red truck", "a blue and white plate"; not "an orange", not the last
           word); never the fixed phrase "black and white"; targets are basic colours outside the source's
           near-synonym class (grey/silver, tan/beige/brown, gold/yellow)
  count    "one" only as a numeral (not "one of", "another one", "no one", enumerations) with the counted
           noun pluralised; two..eight -> another plural count (never back to one)
  shape    size words only to their antonym (big/small, tall/short, long/short, wide/narrow); shape words
           to a different shape class; attributive use only; no "diamond", no place-name "square"
  texture  materials (wood, metal, glass, plastic, ...) to another material, attributive or after "of";
           surface qualities to their antonym (smooth/rough, glossy/matte, shiny/dull, soft/hard,
           spotted/striped); never a texture noun ("a glass of wine", "toilet paper", "paper reading ...")
  spatial  the relation pairs (left/right, above/below, on top of/under, next to/far from, in front of/
           behind), but not "on top of" a ground / surface noun
  verb     the verb pairs, only when an animate noun precedes the verb ("a bed sitting next to a tree"
           changes nothing in the picture)

    python3 data/build_negatives.py --cache cache/train_3k --m 3 --families color,texture,verb,shape,count \
        --count_min_delta 2 --out cache/negatives_3k.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

FAMILY_ORDER = ["spatial", "3d_spatial", "count", "color", "shape", "texture", "verb"]
FAMILIES = list(FAMILY_ORDER)          # settable from main(): --families
COUNT_MIN_DELTA = 1                    # --count_min_delta: minimum |target - source| for count edits

# relation and verb pairs (the phaseK counterfactual vocabulary of the experimental branch)
SPATIAL_PAIRS = [("on the left of", "on the right of"), ("on the right of", "on the left of"),
                 ("to the left of", "to the right of"), ("to the right of", "to the left of"),
                 ("left of", "right of"), ("right of", "left of"), ("on the top of", "on the bottom of"),
                 ("on the bottom of", "on the top of"), ("on top of", "under"), ("at the top of", "at the bottom of"),
                 ("at the bottom of", "at the top of"), ("above", "below"), ("below", "above"), ("under", "above"),
                 ("beneath", "above"), ("next to", "far from"), ("near", "far from"), ("on side of", "far from"), ("far from", "near")]
SPATIAL3D_PAIRS = [("in front of", "behind"), ("behind", "in front of")]
VERB_PAIRS = [("standing", "sitting"), ("sitting", "standing"), ("walking", "running"), ("running", "walking"),
              ("holding", "dropping"), ("looking at", "looking away from"), ("feeding", "ignoring"),
              ("sleeping", "standing"), ("lying", "standing"), ("flying", "resting"), ("playing with", "ignoring"),
              ("chasing", "running away from"), ("jumping over", "standing beside")]

FUNCTION = {"and", "or", "with", "of", "on", "in", "at", "to", "that", "is", "are", "was", "were", "which", "who",
            "while", "as", "for", "by", "from", "into", "onto", "near", "under", "over", "the", "a", "an", "it",
            "its", "his", "her", "their", "one", "ones", "looking", "sitting", "standing", "but", "than", "then",
            "up", "down", "out", "off", "next", "against", "behind", "above", "below", "beside", "around"}

COLOR_CLASSES = [{"gray", "grey", "silver"}, {"tan", "beige", "brown"}, {"gold", "yellow"}, {"orange"}, {"red"},
                 {"blue"}, {"green"}, {"purple"}, {"pink"}, {"black"}, {"white"}]
COLOR_WORDS = set().union(*COLOR_CLASSES)
COLOR_TARGETS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"]

NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
ONE_PREV = {"no", "the", "this", "that", "another", "each", "every", "which", "any", "only", "some", "other"}
ONE_NEXT = {"of", "another", "more", "day", "way", "side", "end", "hand", "time", "thing", "person's"}
IRREGULAR = {"man": "men", "woman": "women", "child": "children", "person": "people", "foot": "feet", "tooth": "teeth",
             "mouse": "mice", "goose": "geese", "sheep": "sheep", "deer": "deer", "fish": "fish", "moose": "moose",
             "cactus": "cacti", "leaf": "leaves", "knife": "knives", "loaf": "loaves", "wolf": "wolves", "calf": "calves",
             "half": "halves", "shelf": "shelves", "life": "lives", "ox": "oxen"}
ADJ_SKIP = COLOR_WORDS | {"big", "small", "tall", "short", "long", "wide", "narrow", "large", "little", "young", "old",
                          "giant", "huge", "tiny", "cute", "wooden", "metal", "plastic", "glass", "stuffed", "adult",
                          "baby", "red", "male", "female", "elderly", "older", "younger", "lone", "single", "empty",
                          "full", "open", "closed", "new", "vintage", "modern", "fresh", "hot", "cold", "dirty", "clean"}

SIZE_ANT = {"big": ["small"], "small": ["big"], "tall": ["short"], "short": ["tall", "long"], "long": ["short"],
            "wide": ["narrow"], "narrow": ["wide"], "large": ["small"], "little": ["big"], "huge": ["tiny"], "tiny": ["huge"]}
SHAPE_CLASSES = [{"round", "circular", "circle"}, {"square", "cubic"}, {"triangular", "triangle"},
                 {"rectangular", "rectangle", "oblong"}, {"spherical", "sphere"}, {"cylindrical", "cylinder"},
                 {"conical"}, {"oval"}, {"hexagonal"}, {"pentagonal"}, {"pyramidal"}, {"teardrop"}, {"angular"}]
SHAPE_WORDS = set().union(*SHAPE_CLASSES)
SHAPE_TARGETS = ["round", "square", "triangular", "rectangular", "oval", "hexagonal"]
SQUARE_PREV = {"town", "times", "public", "city", "market", "building", "main", "village", "the", "a"}

MATERIALS = {"metal", "metallic", "wood", "wooden", "glass", "rubber", "plastic", "fabric", "stone", "ceramic", "leather",
             "paper", "woolen", "brick", "concrete", "marble", "steel", "iron"}
MAT_TARGETS = ["wooden", "metal", "glass", "plastic", "stone", "leather", "paper", "fabric", "brick"]
MAT_CLASSES = [{"metal", "metallic", "steel", "iron"}, {"wood", "wooden"}, {"glass"}, {"rubber"}, {"plastic"}, {"fabric", "woolen"},
               {"stone", "concrete", "marble", "brick"}, {"ceramic"}, {"leather"}, {"paper"}]
QUAL_ANT = {"smooth": ["rough"], "rough": ["smooth"], "glossy": ["matte"], "matte": ["glossy"], "shiny": ["dull"],
            "dull": ["shiny"], "soft": ["hard"], "hard": ["soft"], "spotted": ["striped"], "striped": ["spotted"],
            "fluffy": ["smooth"], "fuzzy": ["smooth"]}

ANIMATE = {"person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children",
           "kid", "kids", "baby", "babies", "toddler", "guy", "guys", "lady", "ladies", "couple", "group", "crowd",
           "family", "adult", "adults", "player", "players", "skier", "skiers", "surfer", "surfers", "skateboarder",
           "skateboarders", "snowboarder", "snowboarders", "rider", "riders", "biker", "bikers", "cyclist", "cyclists",
           "motorcyclist", "pedestrian", "pedestrians", "passenger", "passengers", "worker", "workers", "chef", "cook",
           "officer", "police", "soldier", "batter", "pitcher", "catcher", "umpire", "teacher", "student", "students",
           "friends", "gentleman", "male", "female", "teenager", "teen", "someone", "somebody", "everyone", "dog",
           "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens", "horse", "horses", "pony", "cow", "cows",
           "cattle", "bull", "sheep", "lamb", "goat", "goats", "pig", "pigs", "elephant", "elephants", "zebra",
           "zebras", "giraffe", "giraffes", "bear", "bears", "bird", "birds", "duck", "ducks", "goose", "geese",
           "pigeon", "pigeons", "seagull", "seagulls", "gull", "owl", "eagle", "parrot", "chicken", "chickens",
           "animal", "animals", "monkey", "monkeys", "deer", "rabbit", "bunny", "squirrel", "lion", "tiger",
           "camel", "donkey", "mule", "penguin", "swan", "flamingo", "hawk", "crow", "bee", "butterfly", "frog",
           "turtle", "lizard", "fish", "dolphin", "whale", "seal", "otter", "kangaroo", "koala", "panda", "fox",
           "wolf", "moose", "buffalo", "bison", "ram", "hen", "rooster", "turkey", "peacock"}

SURFACE = {"field", "fields", "grass", "ground", "floor", "beach", "sand", "snow", "hill", "hills", "road", "street",
           "sidewalk", "lawn", "dirt", "pavement", "water", "ocean", "sea", "lake", "river", "court", "track", "slope",
           "slopes", "mountain", "mountains", "meadow", "pasture", "land", "surface", "roof", "rooftop", "runway",
           "path", "trail", "carpet", "rug", "grassland", "plain", "plains", "terrain", "area", "lot", "deck", "pier"}


def _case_like(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def _next(prompt: str, end: int):
    m = re.match(r"[\s-]*([A-Za-z']+)", prompt[end:])
    return m.group(1).lower() if m else None


def _prev(prompt: str, start: int):
    m = re.search(r"([A-Za-z']+)[\s-]*$", prompt[:start])
    return m.group(1).lower() if m else None


def _attributive(prompt: str, m) -> bool:
    nw = _next(prompt, m.end())
    return nw is not None and nw not in FUNCTION


def _finditer(prompt: str, source: str):
    return re.compile(rf"(?<![\w-]){re.escape(source)}(?![\w-])", re.IGNORECASE).finditer(prompt)


def _class_of(word: str, classes):
    for c in classes:
        if word in c:
            return c
    return {word}


def _rotation(source: str, targets: list[str], present: set[str], classes) -> list[str]:
    cls = _class_of(source, classes)
    start = targets.index(source) + 1 if source in targets else 0
    return [targets[(start + k) % len(targets)] for k in range(len(targets))
            if targets[(start + k) % len(targets)] not in cls and targets[(start + k) % len(targets)] not in present]


def _plural(noun: str) -> str:
    low = noun.lower()
    if low in IRREGULAR:
        p = IRREGULAR[low]
    elif re.search(r"[^aeiou]y$", low):
        p = low[:-1] + "ies"
    elif re.search(r"(s|x|z|ch|sh)$", low):
        p = low + "es"
    else:
        p = low + "s"
    return _case_like(noun, p)


def _edits(prompt: str, family: str) -> list[dict]:
    present = set(_words(prompt))
    out = []
    if family == "color":
        bw = [(m.start(), m.end()) for m in re.finditer(r"\bblack[\s-]+and[\s-]+white\b|\bwhite[\s-]+and[\s-]+black\b", prompt, re.I)]
        for src in sorted(COLOR_WORDS):
            for m in _finditer(prompt, src):
                if any(a <= m.start() and m.end() <= b for a, b in bw):
                    continue
                ok = _attributive(prompt, m)
                if not ok and _next(prompt, m.end()) == "and":
                    m3 = re.match(r"[\s-]*and[\s-]+([A-Za-z]+)[\s-]+([A-Za-z']+)", prompt[m.end():], re.I)
                    ok = bool(m3) and m3.group(1).lower() in COLOR_WORDS and m3.group(2).lower() not in FUNCTION
                if not ok or (src in {"orange", "gold", "silver", "tan"} and not _attributive(prompt, m)):
                    continue
                tg = _rotation(src, COLOR_TARGETS, present, COLOR_CLASSES)
                if tg:
                    out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": tg})
    elif family == "count":
        for src in NUMS:
            for m in _finditer(prompt, src):
                pw, nw = _prev(prompt, m.start()), _next(prompt, m.end())
                if nw is None:
                    continue
                if src == "one":
                    if pw in ONE_PREV or nw in ONE_NEXT or nw in FUNCTION:
                        continue
                    if re.search(r"\b(the other|another|others)\b", prompt, re.I) or len(list(_finditer(prompt, "one"))) > 1:
                        continue
                    m2 = re.match(r"(\s+)([A-Za-z']+)(\s+([A-Za-z']+))?", prompt[m.end():])
                    if not m2:
                        continue
                    noun_start, noun = m.end() + len(m2.group(1)), m2.group(2)
                    if noun.lower() in ADJ_SKIP:
                        if m2.group(4) and m2.group(4).lower() not in FUNCTION:
                            noun_start = m.end() + m2.start(4); noun = m2.group(4)
                        else:
                            continue
                    if noun.lower() in FUNCTION or noun.lower().endswith(("ing", "ed")):
                        continue
                    if noun.lower().endswith("s") and not noun.lower().endswith(("ss", "us", "is")):
                        continue
                    out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": ["two", "three"],
                                "noun_span": (noun_start, noun_start + len(noun)), "noun_plural": _plural(noun)})
                else:
                    tg = [t for t in NUMS[1:] if t != src and t not in present]
                    i = NUMS.index(src); tg = sorted(tg, key=lambda t: (abs(NUMS.index(t) - i) != 1, abs(NUMS.index(t) - i)))
                    out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": tg})
    elif family == "shape":
        for src in sorted(set(SIZE_ANT) | SHAPE_WORDS):
            for m in _finditer(prompt, src):
                if not _attributive(prompt, m):
                    continue
                if src == "square" and _prev(prompt, m.start()) in SQUARE_PREV and _next(prompt, m.end()) in FUNCTION | {"."}:
                    continue
                tg = [t for t in SIZE_ANT[src] if t not in present] if src in SIZE_ANT else _rotation(src, SHAPE_TARGETS, present, SHAPE_CLASSES)
                if tg:
                    out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": tg})
    elif family == "texture":
        for src in sorted(MATERIALS | set(QUAL_ANT)):
            for m in _finditer(prompt, src):
                if src in MATERIALS:
                    if not (_attributive(prompt, m) or _prev(prompt, m.start()) == "of"):
                        continue
                    if _prev(prompt, m.start()) in {"toilet", "tissue", "news", "wrapping", "wax"} or (_next(prompt, m.end()) or "").endswith("ing"):
                        continue
                    tg = _rotation(src, MAT_TARGETS, present, MAT_CLASSES)
                    if _prev(prompt, m.start()) == "of":
                        tg = [t for t in tg if t != "wooden"] + (["wood"] if "wood" not in present and src not in {"wood", "wooden"} else [])
                else:
                    if not _attributive(prompt, m):
                        continue
                    tg = [t for t in QUAL_ANT[src] if t not in present]
                if tg:
                    out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": tg})
    elif family == "verb":
        for src, tgt in VERB_PAIRS:
            if re.search(rf"(?<![\w-]){re.escape(tgt)}(?![\w-])", prompt, re.I):
                continue
            for m in _finditer(prompt, src):
                if not (set(_words(prompt[:m.start()])) & ANIMATE) or (src == "lying" and _next(prompt, m.end()) == "down"):
                    continue
                out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": [tgt]})
    else:
        pairs = SPATIAL_PAIRS if family == "spatial" else SPATIAL3D_PAIRS
        for src, tgt in pairs:
            if re.search(rf"(?<![\w-]){re.escape(tgt)}(?![\w-])", prompt, re.I):
                continue
            for m in _finditer(prompt, src):
                if src in {"on top of", "on the top of", "above", "over"}:
                    m4 = re.match(r"[\s-]*(?:(?:a|an|the|some|his|her|their|its)[\s-]+)?([A-Za-z']+)(?:[\s-]+([A-Za-z']+))?", prompt[m.end():])
                    if m4 and any(w and w.lower() in SURFACE for w in (m4.group(1), m4.group(2))):
                        continue
                out.append({"source": m.group(0), "start": m.start(), "end": m.end(), "targets": [tgt]})
    out.sort(key=lambda d: (d["start"], -d["end"]))
    return out


def _apply(prompt: str, e: dict, target: str) -> str:
    rep = _case_like(e["source"], target)
    new = prompt[:e["start"]] + rep + prompt[e["end"]:]
    if "noun_span" in e:
        a, b = e["noun_span"]
        shift = len(rep) - (e["end"] - e["start"])
        new = new[:a + shift] + e["noun_plural"] + new[b + shift:]
    art = re.search(r"(?<![\w-])(a|an)([\s-]+)$", new[:e["start"]], re.I)
    if art:
        want = "an" if rep[:1].lower() in "aeiou" else "a"
        if art.group(1).lower() != want:
            new = new[:art.start(1)] + _case_like(art.group(1), want) + new[art.end(1):]
    return new


def negatives_for(prompt: str, m: int, distinct_spans: bool = True) -> list[dict]:
    combos = {}
    for f in FAMILY_ORDER:
        if f not in FAMILIES:
            combos[f] = []
            continue
        combos[f] = [(e, t) for e in _edits(prompt, f) for t in e["targets"]
                     if not (f == "count" and e["source"].lower() in NUMS and t in NUMS
                             and abs(NUMS.index(t) - NUMS.index(e["source"].lower())) < COUNT_MIN_DELTA)]
        spans = {}
        for e, t in combos[f]:
            spans.setdefault(e["start"], []).append((e, t))
        rr = []
        for k in range(max((len(v) for v in spans.values()), default=0)):
            for s in sorted(spans):
                if k < len(spans[s]):
                    rr.append(spans[s][k])
        combos[f] = rr
    negs, seen, used = [], {prompt}, set()

    def take(e, t, f):
        new = _apply(prompt, e, t)
        if new not in seen:
            seen.add(new); used.add((e["start"], e["end"]))
            negs.append({"prompt": new, "family": f, "source": e["source"], "target": t})

    for pass_no in (0, 1):
        r = 0
        while len(negs) < m and any(r < len(c) for c in combos.values()):
            for f in FAMILY_ORDER:
                if r < len(combos[f]) and len(negs) < m:
                    e, t = combos[f][r]
                    if pass_no == 0 and distinct_spans and (e["start"], e["end"]) in used:
                        continue
                    take(e, t, f)
            r += 1
        if not distinct_spans:
            break
    return negs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="candidate cache dir (selection_rank*.jsonl with idx and prompt)")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--families", default=",".join(FAMILY_ORDER))
    ap.add_argument("--count_min_delta", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global FAMILIES, COUNT_MIN_DELTA
    FAMILIES = [f for f in args.families.split(",") if f]; COUNT_MIN_DELTA = args.count_min_delta
    recs = {}
    for f in sorted(glob.glob(f"{args.cache}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln); recs[int(r["idx"])] = r["prompt"]
    out, n_neg, fam = {}, Counter(), Counter()
    for idx in sorted(recs):
        negs = negatives_for(recs[idx], args.m)
        n_neg[len(negs)] += 1
        for n in negs:
            fam[n["family"]] += 1
        if negs:
            out[str(idx)] = {"prompt": recs[idx], "negatives": negs}
    summary = {"captions": len(recs), "with_negatives": len(out), "coverage": len(out) / max(len(recs), 1), "m": args.m,
               "families_used": FAMILIES, "count_min_delta": COUNT_MIN_DELTA,
               "negatives_per_caption": {str(k): v for k, v in sorted(n_neg.items())}, "families": dict(fam.most_common()), "cache": args.cache}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "negatives": out}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
