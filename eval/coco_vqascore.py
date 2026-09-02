#!/usr/bin/env python3
"""In-domain alignment: VQAScore(image, caption) on held-out COCO captions.

CompBench and GenEval2 are out-of-domain prompts for a COCO-trained student. This scores the
fidelity pool (COCO val2017 captions, images already generated for FID/CMMD) with VQAScore, giving
the in-domain half. Needs third_party/t2v_metrics importable.

    python eval/coco_vqascore.py --images out/fidelity/images/dino_patch_s0_s4 --steps 4 \
        --prompts pools/eval/fidelity_prompts.json --out out/coco_vqa/dino_patch_s0.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="{images}/p{idx:05d}/s{steps}/cand0.png")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--image_name", default="cand0.png")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--t2v_dir", default=str(ROOT / "third_party" / "t2v_metrics"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.t2v_dir)
    import t2v_metrics
    vqa = t2v_metrics.VQAScore(model=args.vqa_model, device="cuda:0")

    pool = json.loads(Path(args.prompts).read_text())
    items, missing = [], []
    for it in pool:
        p = Path(args.images) / f"p{it['idx']:05d}" / f"s{args.steps}" / args.image_name
        (items if p.exists() else missing).append((it["idx"], it["prompt"], str(p)))
    if missing:
        sys.exit(f"{len(missing)} of {len(pool)} images missing under {args.images}")
    rows = []
    for n, (idx, prompt, path) in enumerate(items):
        rows.append({"idx": idx, "prompt": prompt, "score": float(vqa(images=[path], texts=[prompt]).item())})
        if n % 500 == 0:
            print(f"[coco-vqa] {n}/{len(items)}", flush=True)
    mean = sum(r["score"] for r in rows) / len(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"images": args.images, "steps": args.steps, "vqa_model": args.vqa_model,
               "n": len(rows), "mean": mean, "per_prompt": rows}, open(args.out, "w"), indent=1)
    print(f"in-domain VQAScore mean {mean:.4f} over {len(rows)} captions -> {args.out}")


if __name__ == "__main__":
    main()
