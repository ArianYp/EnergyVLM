#!/usr/bin/env python3
"""In-domain alignment: VQAScore(image, caption) on held-out COCO captions.

Every alignment number in the project so far is out-of-domain (T2I-CompBench / GenEval2 prompts for
a COCO-trained student). This scores the fidelity pools -- 5,000 COCO val2017 captions, images
already generated for FID/CMMD -- with the same VQAScore model used for selection, giving the
in-domain half of the comparison. Per-prompt scores are kept for paired contrasts.

    python3 phaseW/score_coco_vqa.py --images phaseT/fidelity/images/T_B4_s0 --steps 4 \
        --prompts phaseT/fidelity/prompts.json --out phaseW/coco_vqa/T_B4_s0.json
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="{images}/p{idx:05d}/s{steps}/cand0.png")
    ap.add_argument("--prompts", default="phaseT/fidelity/prompts.json")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--image_name", default="cand0.png")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "t2v_metrics")); sys.path.insert(0, str(ROOT / "exp0"))
    import _t2v_compat  # noqa: F401
    import t2v_metrics
    hub_cache = os.path.join(os.environ.get("HF_HOME", str(ROOT / "cache/huggingface")), "hub")
    vqa = t2v_metrics.VQAScore(model=args.vqa_model, device="cuda:0", cache_dir=hub_cache)

    pool = json.loads(Path(args.prompts).read_text())
    items, missing = [], 0
    for it in pool:
        p = Path(args.images) / f"p{it['idx']:05d}" / f"s{args.steps}" / args.image_name
        if p.exists():
            items.append((it["idx"], it["prompt"], str(p)))
        else:
            missing += 1
    if missing:
        sys.exit(f"{missing} of {len(pool)} images missing under {args.images}")
    rows = []
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        # one caption per image: score the diagonal of the (images x texts) matrix
        for idx, prompt, path in chunk:
            s = float(vqa(images=[path], texts=[prompt]).item())
            rows.append({"idx": idx, "prompt": prompt, "score": s})
        if (i // args.batch) % 20 == 0:
            print(f"[coco-vqa] {i + len(chunk)}/{len(items)}", flush=True)
    mean = sum(r["score"] for r in rows) / len(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"images": args.images, "steps": args.steps, "vqa_model": args.vqa_model, "n": len(rows),
               "mean": mean, "per_prompt": rows}, open(args.out, "w"), indent=1)
    print(f"{Path(args.images).name}: in-domain VQAScore mean {mean:.4f} over {len(rows)} COCO captions -> {args.out}")


if __name__ == "__main__":
    main()
