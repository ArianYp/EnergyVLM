#!/usr/bin/env python3
"""Held-out DINO similarity of a student checkpoint (the monitor of the reward study, report
Section 11.7). The same validation captions (from the latent manifest; never in the training pool)
for every checkpoint, two quantities, both scored offline (8-bit image, HF processor, DINOv2 patch
mean) against the caption's reference photograph:

  (a) supervised clean estimates: the caption's argmax teacher candidate is rolled out (K steps,
      guidance w); the student's x0-hat from the two least-noisy supervised inputs (states K-3 and
      K-2, Delta = 1) is decoded and scored;
  (b) complete 4-step samples at w = 1 (two seeds per caption).

    python3 eval/heldout_dino.py --ckpt checkpoints/dino_patch_3k_s0/checkpoint_avg_last5.pt \
        --manifest cache/latents/manifest.jsonl --out out/heldout/dino_patch_s0@avg_last5.json
    python3 eval/heldout_compare.py --dir out/heldout          # seed statistics and paired contrasts
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import candidate_noise, encode_prompt, rollout, vae_decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt, or 'base' for the undistilled transformer")
    ap.add_argument("--manifest", default="cache/latents/manifest.jsonl", help="data/build_latent_manifest.py output (split, prompt, reference, seed_base, dino_patch_cos)")
    ap.add_argument("--n_captions", type=int, default=16)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--select_seed", type=int, default=12345, help="which validation captions (fixed across checkpoints)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda")
    recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
    val = [r for r in recs if r["split"] == "val"]
    rng = np.random.default_rng(args.select_seed); val = [val[i] for i in rng.choice(len(val), size=args.n_captions, replace=False)]

    from diffusers import StableDiffusion3Pipeline
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    teacher = copy.deepcopy(pipe.transformer)
    student = pipe.transformer
    if args.ckpt != "base":
        ck = torch.load(args.ckpt, map_location="cpu", mmap=False, weights_only=False)
        student.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["model"].items()}, strict=True)
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval(); proc = AutoImageProcessor.from_pretrained(args.dino_id)
    lat_c = student.config.in_channels; H = args.height // pipe.vae_scale_factor

    @torch.no_grad()
    def embed_lat(lat):
        u8 = ((vae_decode(pipe.vae, lat) + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
        px = proc(images=[Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in u8], return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(dino(pixel_values=px).last_hidden_state[:, 1:].float().mean(1), dim=-1)

    @torch.no_grad()
    def embed_img(path):
        p = Path(str(path)); p = p if p.suffix else p.with_suffix(".jpg")
        img = Image.open(p).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
        px = proc(images=[img], return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(dino(pixel_values=px).last_hidden_state[:, 1:].float().mean(1), dim=-1)[0]

    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)
    out = {"ckpt": args.ckpt, "captions": []}
    sup = [args.K - 3, args.K - 2]                                    # the two least-noisy supervised inputs (Delta = 1)
    with torch.no_grad():
        for r in val:
            emb, pooled = encode_prompt(pipe, r["prompt"], device)
            e_ref = embed_img(r["reference"])
            j = int(np.argmax(r["dino_patch_cos"]))
            z0 = candidate_noise(r["seed_base"], j, (1, lat_c, H, H), device)
            zs, sig = rollout(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, args.K, args.cfg, device, keep_states=True)
            ts = pipe.scheduler.timesteps.to(device)
            x0s = []
            for s in sup:
                with torch.autocast("cuda", torch.bfloat16):
                    v = student(hidden_states=zs[s], timestep=ts[s].expand(1), encoder_hidden_states=emb, pooled_projections=pooled, return_dict=False)[0]
                x0s.append(zs[s].float() - sig[s] * v.float())
            sc_x0 = (embed_lat(torch.cat(x0s, 0)) * e_ref[None]).sum(-1).cpu().tolist()
            zsam = torch.randn(args.n_seeds, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(int(r["idx"]) * 7 + 1))
            zK = rollout(student, pipe.scheduler, zsam, emb, pooled, neg_emb, neg_pool, 4, 1.0, device)
            sc_sam = (embed_lat(zK) * e_ref[None]).sum(-1).cpu().tolist()
            out["captions"].append({"idx": int(r["idx"]), "x0_dino": sc_x0, "sample_dino": sc_sam, "teacher_argmax_dino": float(max(r["dino_patch_cos"]))})
    out["summary"] = {"x0_dino_mean": float(np.mean([np.mean(c["x0_dino"]) for c in out["captions"]])),
                      "sample_dino_mean": float(np.mean([np.mean(c["sample_dino"]) for c in out["captions"]])),
                      "teacher_argmax_dino_mean": float(np.mean([c["teacher_argmax_dino"] for c in out["captions"]]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)
    print("[heldout] " + json.dumps(out["summary"]) + f" ({args.ckpt})", flush=True)


if __name__ == "__main__":
    main()
