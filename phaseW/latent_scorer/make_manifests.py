#!/usr/bin/env python3
"""Manifests for the latent-scorer study.

  train / val : captions of the 118k cache whose prompt is NOT in the 3k pool (24,000 drawn with a
                fixed seed; 22,000 train, 2,000 val = held-out captions and photographs)
  test        : the 3k pool itself (the captions the selection-rule arms train on), so the scorer
                that ranks those candidates never saw them

Each manifest line is the original cache record plus a `split` field; the generator re-rolls the
candidates from `seed_base` and stores the terminal latents with the DINO embeddings.
"""
import json
import random
from pathlib import Path

excl = set()
test = []
for f in sorted(Path("phaseN/coco_selection_dinopatch").glob("selection_rank*.jsonl")):
    for ln in f.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln); excl.add(r["prompt"]); r["split"] = "test"; test.append(r)
pool = []
for f in sorted(Path("phaseN/coco_selection_118k").glob("selection_rank*.jsonl")):
    for ln in f.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["prompt"] not in excl and "dino_patch_cos" in r and "endpoint_vqa" in r:
                pool.append(r)
pool.sort(key=lambda r: r["idx"])
random.Random(20260907).shuffle(pool)
sel = pool[:24000]
for i, r in enumerate(sel):
    r["split"] = "train" if i < 22000 else "val"
out = Path("phaseW/latent_scorer"); out.mkdir(exist_ok=True)
with open(out / "manifest.jsonl", "w") as fh:
    for r in sel + test:
        fh.write(json.dumps(r) + "\n")
print(f"train {sum(r['split']=='train' for r in sel)} val {sum(r['split']=='val' for r in sel)} test {len(test)} "
      f"(118k pool after exclusion: {len(pool)}; excluded prompts {len(excl)})")
