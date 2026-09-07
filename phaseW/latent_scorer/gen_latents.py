#!/usr/bin/env python3
"""Terminal latents + DINO embeddings for the latent-scorer study, one shard of the manifest.

For every record: the four candidates are re-rolled from `seed_base + j` with the frozen 8-step
CFG-7 teacher (the trainer's rollout, so these are the trajectories the students train on), the
terminal latent z_K is kept in bf16, each decoded candidate and the reference photograph are
embedded with DINOv2-B mean-pooled patches (the offline scorer's preprocessing), and the cosine is
recomputed and checked against the cached `dino_patch_cos` (bf16 drift is ~2e-3).

    python3 phaseW/latent_scorer/gen_latents.py --manifest phaseW/latent_scorer/manifest.jsonl \
        --shard 0 --nshard 10 --out phaseW/latent_scorer/shards/shard0.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def rollout(teacher, scheduler, z0, emb, pooled, neg_emb, neg_pool, K, cfg, device):
    N = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sigmas = scheduler.sigmas.to(device, torch.float32)
    ts = scheduler.timesteps.to(device)
    e = torch.cat([neg_emb.repeat(N, 1, 1), emb.repeat(N, 1, 1)], 0)
    p = torch.cat([neg_pool.repeat(N, 1), pooled.repeat(N, 1)], 0)
    z = z0
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v_all = teacher(hidden_states=torch.cat([z, z], 0), timestep=ts[k].expand(2 * N),
                            encoder_hidden_states=e, pooled_projections=p, return_dict=False)[0]
        v_u, v_c = v_all.chunk(2, 0)
        v = v_u + cfg * (v_c - v_u)
        z = (z.float() + (sigmas[k + 1] - sigmas[k]) * v.float()).to(torch.bfloat16)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda")
    recs = [json.loads(ln) for ln in open(args.manifest) if ln.strip()]
    recs = [r for i, r in enumerate(recs) if i % args.nshard == args.shard]
    print(f"[gen] shard {args.shard}/{args.nshard}: {len(recs)} records", flush=True)

    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
        m.to(dtype=torch.bfloat16).eval()        # the pipeline ships mixed dtypes; the trainer casts too
        for p in m.parameters():
            p.requires_grad = False
    teacher = pipe.transformer
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(args.dino_id)
    lat_c = teacher.config.in_channels
    H = args.height // pipe.vae_scale_factor
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
                                                     do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)

    @torch.no_grad()
    def embed(images):
        px = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
        h = dino(pixel_values=px).last_hidden_state
        return F.normalize(h[:, 1:].float().mean(1), dim=-1)

    out = {k: [] for k in ("idx", "seed_base", "split", "prompt", "reference", "z", "e_cand", "e_ref", "cos_new", "cos_cached", "vqa", "random_idx")}
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(recs):
            N = int(r["N"])
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[r["prompt"]], prompt_2=[r["prompt"]], prompt_3=[r["prompt"]],
                                                   do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            z0 = torch.cat([torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16,
                                        generator=torch.Generator(device=device).manual_seed(int(r["seed_base"]) + j)) for j in range(N)], 0)
            zK = rollout(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, args.K, args.cfg, device)
            lat = (zK.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            img = pipe.vae.decode(lat, return_dict=False)[0].clamp(-1, 1).float()
            u8 = ((img + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
            pil = [Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in u8]
            e_c = embed(pil)
            pref = Path(str(r["reference"])); pref = pref if pref.suffix else pref.with_suffix(".jpg")
            ref = Image.open(pref).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
            e_r = embed([ref])[0]
            cos = (e_c * e_r[None]).sum(-1)
            out["idx"].append(int(r["idx"])); out["seed_base"].append(int(r["seed_base"])); out["split"].append(r["split"])
            out["prompt"].append(r["prompt"]); out["reference"].append(str(r["reference"]))
            out["z"].append(zK.cpu()); out["e_cand"].append(e_c.half().cpu()); out["e_ref"].append(e_r.half().cpu())
            out["cos_new"].append(cos.cpu()); out["cos_cached"].append(torch.tensor(r["dino_patch_cos"], dtype=torch.float32))
            out["vqa"].append(torch.tensor(r.get("endpoint_vqa", [float("nan")] * N), dtype=torch.float32))
            out["random_idx"].append(int(r.get("random_idx", -1)))
            if (i + 1) % 100 == 0:
                d = torch.stack(out["cos_new"]) - torch.stack(out["cos_cached"])
                print(f"[gen] {i+1}/{len(recs)} {(time.time()-t0)/(i+1):.2f}s/rec | recomputed vs cached dino_patch_cos: "
                      f"max|diff| {d.abs().max():.4f} argmax agree {(torch.stack(out['cos_new']).argmax(1) == torch.stack(out['cos_cached']).argmax(1)).float().mean():.3f}", flush=True)
    res = {"idx": torch.tensor(out["idx"]), "seed_base": torch.tensor(out["seed_base"]), "split": out["split"],
           "prompt": out["prompt"], "reference": out["reference"], "random_idx": torch.tensor(out["random_idx"]),
           "z": torch.stack(out["z"]), "e_cand": torch.stack(out["e_cand"]), "e_ref": torch.stack(out["e_ref"]),
           "cos_new": torch.stack(out["cos_new"]), "cos_cached": torch.stack(out["cos_cached"]), "vqa": torch.stack(out["vqa"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, args.out)
    d = (res["cos_new"] - res["cos_cached"]).abs()
    print(f"[gen] wrote {args.out}: {len(res['idx'])} records, z {tuple(res['z'].shape)} {res['z'].dtype}; "
          f"recomputed-vs-cached max|diff| {d.max():.4f} mean {d.mean():.5f}, argmax agreement "
          f"{(res['cos_new'].argmax(1) == res['cos_cached'].argmax(1)).float().mean():.4f}; GEN_SHARD_OK", flush=True)


if __name__ == "__main__":
    main()
