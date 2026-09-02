#!/usr/bin/env python3
"""FID, CMMD, precision and recall of generated sets against real COCO val2017 photographs.

  FID        Inception-pool3 Frechet distance via clean-fid (mode="clean"); the same resizer is
             applied to both sides. Reported with a split-half check (FID on two disjoint halves of
             the generated set) since FID is strongly sample-size biased and has no cheap bootstrap.
  CMMD       CLIP-ViT-L/14 MMD^2, sigma=10, x1000, with a bootstrap CI over generated images.
  precision  improved precision / recall (Kynkaanniemi et al. 2019, k=3) on the CLIP features.
  / recall   Recall falls under diversity collapse; precision saturates near 1 in 768-d and is a
             coarse sanity check only.

Real images are center-cropped to square so square generations are not penalised for aspect.

    python eval/fidelity.py --gen_root out/fidelity/images --models naive_s0,dino_patch_s0,base \
        --steps 4,28 --coco_dir /path/to/COCO/val2017 --out out/fidelity_report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.cmmd import clip_embed, cmmd  # noqa: E402


def square_crop(img_np):
    h, w = img_np.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img_np[top:top + s, left:left + s]


def inception_feats(files, model, square: bool, batch_size=128, num_workers=8):
    from cleanfid import fid as cfid
    return cfid.get_files_features([str(f) for f in files], model, mode="clean", verbose=False,
                                   batch_size=batch_size, num_workers=num_workers,
                                   custom_image_tranform=square_crop if square else None)


def fid_from_feats(a, b):
    from cleanfid.fid import frechet_distance
    return float(frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)))


def precision_recall(real: torch.Tensor, fake: torch.Tensor, k: int = 3):
    def radii(x):
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).values[:, -1]
    real = torch.nn.functional.normalize(real, dim=1)
    fake = torch.nn.functional.normalize(fake, dim=1)
    r_real, r_fake = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    return float((d <= r_real[None, :]).any(1).float().mean()), float((d.T <= r_fake[None, :]).any(1).float().mean())


def cmmd_boot_ci(gen, ref, rng, reps, sigma=10.0, scale=1000.0):
    """Bootstrap over generated images with one cached kernel (MMD^2 is a mean of kernel entries)."""
    def k(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma * sigma))
    K_gg, K_gr, k_rr = k(gen, gen), k(gen, ref), k(ref, ref).mean()
    n = gen.shape[0]
    vals = np.empty(reps)
    for b in range(reps):
        idx = torch.as_tensor(rng.integers(0, n, n), device=gen.device)
        vals[b] = float((K_gg[idx][:, idx].mean() + k_rr - 2 * K_gr[idx].mean()) * scale)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True, help="{gen_root}/{model}/p*/s{steps}/cand0.png")
    ap.add_argument("--models", required=True, help="comma-separated model labels")
    ap.add_argument("--steps", default="4")
    ap.add_argument("--image_name", default="cand0.png")
    ap.add_argument("--coco_dir", required=True, help="COCO val2017 image directory")
    ap.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--n_ref", type=int, default=5000, help="0 = all real images")
    ap.add_argument("--min_gen", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown report; a .json is written beside it")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    rng = np.random.default_rng(args.seed)
    steps = [int(x) for x in args.steps.split(",")]

    coco = sorted(Path(args.coco_dir).glob("*.jpg"))
    if args.n_ref and args.n_ref < len(coco):
        coco = sorted(random.Random(args.seed).sample(coco, args.n_ref))
    print(f"reference: {len(coco)} real images (center-cropped square)", flush=True)

    from cleanfid.fid import build_feature_extractor
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    incep = build_feature_extractor("clean", device=torch.device(device), use_dataparallel=False)
    proc = CLIPImageProcessor.from_pretrained(args.clip_id)
    clip = CLIPVisionModelWithProjection.from_pretrained(args.clip_id, torch_dtype=torch.float16).to(device).eval()
    ref_i = inception_feats(coco, incep, True)
    ref_c = clip_embed([str(p) for p in coco], clip, proc, device).to(device)

    root = Path(args.gen_root)
    res = {}
    for m in args.models.split(","):
        for s in steps:
            files = sorted((root / m).glob(f"p*/s{s}/{args.image_name}"))
            if len(files) < args.min_gen:
                print(f"  skip {m}@{s}: {len(files)} images < --min_gen {args.min_gen}", flush=True)
                continue
            gi = inception_feats(files, incep, False)
            gc = clip_embed([str(f) for f in files], clip, proc, device).to(device)
            n = len(files)
            half = rng.permutation(n)
            fid = fid_from_feats(gi, ref_i)
            fid_h = [fid_from_feats(gi[half[: n // 2]], ref_i), fid_from_feats(gi[half[n // 2:]], ref_i)]
            cm = cmmd(gc, ref_c)
            cm_lo, cm_hi = cmmd_boot_ci(gc, ref_c, rng, args.boot)
            prec, rec = precision_recall(ref_c.float(), gc.float())
            res[f"{m}@{s}"] = {"model": m, "steps": s, "n": n, "fid": fid, "fid_split_half": fid_h,
                               "cmmd": cm, "cmmd_ci": [cm_lo, cm_hi], "precision": prec, "recall": rec}
            print(f"  {m}@{s}: FID={fid:.2f} (halves {fid_h[0]:.2f}/{fid_h[1]:.2f})  CMMD={cm:.2f} "
                  f"[{cm_lo:.2f},{cm_hi:.2f}]  P={prec:.3f} R={rec:.3f}", flush=True)
    if not res:
        sys.exit("no generated set reached --min_gen")

    md = [f"# Fidelity vs {len(coco)} real COCO val2017 images\n\n",
          "| model@steps | n | FID | FID split-half | CMMD | CMMD 95% CI | precision | recall |\n",
          "|---|--:|--:|--:|--:|---|--:|--:|\n"]
    for k, v in res.items():
        md.append(f"| {k} | {v['n']} | {v['fid']:.2f} | {v['fid_split_half'][0]:.2f} / {v['fid_split_half'][1]:.2f} | "
                  f"{v['cmmd']:.2f} | [{v['cmmd_ci'][0]:.2f}, {v['cmmd_ci'][1]:.2f}] | "
                  f"{v['precision']:.3f} | {v['recall']:.3f} |\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(md))
    out.with_suffix(".json").write_text(json.dumps({"reference_n": len(coco), "results": res}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
