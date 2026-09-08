#!/usr/bin/env python3
"""Does the reference photograph's aspect handling leak into DINO selection?

The cache squashes the COCO photo to 512x512 (aspect-distorting resize) before DINO, while the
candidates are natively square. Re-score the same cached candidates against the same photographs
under three reference preprocessings and compare the selections:

    squash        resize the whole photo to 512x512                    (what the cache did)
    crop_resize   center-crop to square, then resize to 512
    resize_crop   resize shortest side to 512, then center-crop 512

Reports, per mode: top-1 agreement with the cached manifest, mean within-prompt Spearman with the
cached scores, and % of oracle VQAScore headroom recovered (the selection-quality proxy). If the
modes disagree materially on which candidate wins, aspect handling is part of the selector.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr


def prep(ref: Image.Image, mode: str, size: int) -> Image.Image:
    w, h = ref.size
    if mode == "squash":
        return ref.resize((size, size), Image.BICUBIC)
    if mode == "crop_resize":
        s = min(w, h)
        return ref.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    if mode == "resize_crop":
        r = size / min(w, h)
        im = ref.resize((max(size, round(w * r)), max(size, round(h * r))), Image.BICUBIC)
        w2, h2 = im.size
        return im.crop(((w2 - size) // 2, (h2 - size) // 2, (w2 - size) // 2 + size, (h2 - size) // 2 + size))
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="phaseN/coco_selection_dinopatch")
    ap.add_argument("--cache_dir", default="phaseT/energy_cache")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", default="phaseW/aspect_check.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    MODES = ("squash", "crop_resize", "resize_crop")

    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoImageProcessor, AutoModel
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval()
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(args.dino_id)

    R = {}
    for f in sorted(glob.glob(str(Path(args.records) / "selection_rank*.jsonl"))):
        for line in open(f):
            r = json.loads(line); R[r["idx"]] = r
    data = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(str(Path(args.cache_dir) / "latents_shard*.pt")))]
    idx = [i for d in data for i in d["idx"]]
    cand = torch.cat([d["cand"] for d in data], 0)
    P, N = len(idx), int(data[0]["N"])
    print(f"[aspect] {P} captions x N={N}", flush=True)

    @torch.no_grad()
    def decode(z):
        img = pipe.vae.decode(z.to(device, torch.bfloat16) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor).sample
        img = ((img.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
        return [Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in img]

    @torch.no_grad()
    def patch_feats(ims):
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device)
        h = dino(pixel_values=px).last_hidden_state
        return F.normalize(h[:, 1:].float().mean(1), dim=-1)

    S = {m: np.zeros((P, N)) for m in MODES}
    aspects = []
    for k in range(P):
        rec = R[idx[k]]
        ref = Image.open(rec["reference"]).convert("RGB")
        aspects.append(max(ref.size) / min(ref.size))
        c = patch_feats(decode(cand[k]))
        for m in MODES:
            r = patch_feats([prep(ref, m, args.height)])
            S[m][k] = (c * r).sum(-1).cpu().numpy()
        if (k + 1) % 500 == 0:
            print(f"[aspect] {k+1}/{P}", flush=True)

    cached = np.array([R[i]["dino_patch_cos"] for i in idx])
    VQ = np.array([R[i]["endpoint_vqa"] for i in idx])
    rnd = np.array([R[i]["random_idx"] for i in idx])
    v_rand = np.mean([VQ[k, rnd[k]] for k in range(P)]); v_orac = VQ.max(1).mean()
    aspects = np.array(aspects)
    out = {"n": P, "aspect_ratio_mean": float(aspects.mean()), "aspect_ratio_p90": float(np.percentile(aspects, 90))}
    print(f"\n  reference aspect ratio: mean {aspects.mean():.2f}, 90th pct {np.percentile(aspects, 90):.2f}")
    print(f"\n  {'mode':12s} {'top1 = cached':>14s} {'top1 = squash':>14s} {'rho vs cached':>14s} {'% VQA headroom':>15s}")
    sq = S["squash"]
    print(f"  {'cached':12s} {'-':>14s} {'-':>14s} {'-':>14s} "
          f"{100*(np.mean([VQ[k, cached[k].argmax()] for k in range(P)])-v_rand)/(v_orac-v_rand):14.1f}%")
    for m in MODES:
        s = S[m]
        agree_cached = float(np.mean(s.argmax(1) == cached.argmax(1)))
        agree_sq = float(np.mean(s.argmax(1) == sq.argmax(1)))
        rho = float(np.nanmean([spearmanr(s[k], cached[k])[0] for k in range(P)]))
        hr = 100 * (np.mean([VQ[k, s[k].argmax()] for k in range(P)]) - v_rand) / (v_orac - v_rand)
        out[m] = {"top1_agree_cached": agree_cached, "top1_agree_squash": agree_sq, "rho_vs_cached": rho, "headroom_pct": hr}
        print(f"  {m:12s} {agree_cached:14.3f} {agree_sq:14.3f} {rho:14.3f} {hr:14.1f}%")
    # does disagreement concentrate on non-square photos?
    wide = aspects > 1.3
    d = S["crop_resize"].argmax(1) != sq.argmax(1)
    out["disagree_rate_wide"] = float(d[wide].mean()); out["disagree_rate_square"] = float(d[~wide].mean())
    print(f"\n  crop_resize vs squash disagreement: {100*d[wide].mean():.1f}% on photos with aspect > 1.3 "
          f"vs {100*d[~wide].mean():.1f}% on near-square photos")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
