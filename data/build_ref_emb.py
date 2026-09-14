#!/usr/bin/env python3
"""Reference-photograph DINO embeddings for the reward arms at ANY pool size:
{caption idx -> DINOv2-B patch-mean unit vector (fp16)} for every record of a candidate cache, with
the offline scorer's conventions (photo squashed to height x height with bicubic, HF processor:
resize 256, centre crop 224, ImageNet statistics; CLS dropped; L2-normalised).

data/build_reward_refs.py writes the same vectors for the 3k pool from the latent shards (it also
writes the replay set); this script needs only the cache and is how the 118k-pool references of the
paper's arm were built. On captions shared with an existing file the two agree to fp16 precision
(--check prints the cosine).

    python3 data/build_ref_emb.py --cache cache/train --out cache/reward/ref_emb_118k.pt --check cache/reward/ref_emb.pt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="data/build_candidates.py output (selection_rank*.jsonl with the reference paths)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--check", default=None, help="an existing ref_emb file; shared captions are compared")
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device("cuda")
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(args.dino_id)
    recs = {}
    for f in sorted(glob.glob(f"{args.cache}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                recs[int(r["idx"])] = r["reference"]
    ids = sorted(recs)
    out = {}
    with torch.no_grad():
        for i in range(0, len(ids), args.batch):
            chunk = ids[i:i + args.batch]
            imgs = []
            for k in chunk:
                p = Path(str(recs[k]))
                p = p if p.suffix else p.with_suffix(".jpg")
                imgs.append(Image.open(p).convert("RGB").resize((args.height, args.height), Image.BICUBIC))
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            h = dino(pixel_values=px).last_hidden_state[:, 1:].float()
            e = F.normalize(h.mean(1), dim=-1).cpu().to(torch.float16)
            for k, v in zip(chunk, e):
                out[k] = v
            if (i // args.batch) % 100 == 0:
                print(f"{i}/{len(ids)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("wrote", args.out, len(out))
    if args.check and Path(args.check).exists():
        ref = torch.load(args.check, map_location="cpu", weights_only=False)
        common = [k for k in ref if k in out]
        if common:
            cos = torch.stack([(ref[k].float() * out[k].float()).sum() for k in common])
            print(f"check vs {args.check}: {len(common)} shared captions, cosine min {cos.min():.5f} mean {cos.mean():.5f}")


if __name__ == "__main__":
    main()
