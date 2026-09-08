#!/usr/bin/env python3
"""Held-out DINO similarity of a student checkpoint, on two kinds of outputs, for the reward study
(reviewer requirement 5). Same 16 validation captions (never in the 3k pool) for every checkpoint.

  (a) supervised clean estimates: the caption's argmax teacher candidate is rolled out (8 steps,
      w=7); the student's x0-hat from the two least-noisy supervised inputs (states 5 and 6, Delta=1)
      is decoded and scored with the OFFLINE PIL-based DINO scorer against the reference photo;
  (b) complete 4-step samples at w=1 (two seeds per caption), decoded and scored the same way.

    python3 phaseW/latent_scorer/heldout_dino.py --ckpt <checkpoint.pt|base> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phaseW.latent_scorer.gen_latents import rollout as rollout_teacher  # noqa: E402


def rollout_student(model, scheduler, z0, emb, pooled, K, device):
    N = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sig = scheduler.sigmas.to(device, torch.float32); ts = scheduler.timesteps.to(device)
    z = z0
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v = model(hidden_states=z, timestep=ts[k].expand(N), encoder_hidden_states=emb.repeat(N, 1, 1),
                      pooled_projections=pooled.repeat(N, 1), return_dict=False)[0]
        z = (z.float() + (sig[k + 1] - sig[k]) * v.float()).to(torch.bfloat16)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="phaseW/latent_scorer/manifest.jsonl")
    ap.add_argument("--n_captions", type=int, default=16)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda")
    recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
    val = [r for r in recs if r["split"] == "val"]
    rng = np.random.default_rng(12345); val = [val[i] for i in rng.choice(len(val), size=args.n_captions, replace=False)]

    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    import copy
    teacher = copy.deepcopy(pipe.transformer)
    student = pipe.transformer
    if args.ckpt != "base":
        ck = torch.load(args.ckpt, map_location="cpu", mmap=False, weights_only=False)
        student.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["model"].items()}, strict=True)
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval(); proc = AutoImageProcessor.from_pretrained(args.dino_id)
    lat_c = student.config.in_channels; H = args.height // pipe.vae_scale_factor

    @torch.no_grad()
    def embed_lat(lat):
        z = (lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        img = pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1).float()
        u8 = ((img + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
        px = proc(images=[Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in u8], return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(dino(pixel_values=px).last_hidden_state[:, 1:].float().mean(1), dim=-1)

    @torch.no_grad()
    def embed_img(path):
        p = Path(str(path)); p = p if p.suffix else p.with_suffix(".jpg")
        img = Image.open(p).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
        px = proc(images=[img], return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(dino(pixel_values=px).last_hidden_state[:, 1:].float().mean(1), dim=-1)[0]

    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    out = {"ckpt": args.ckpt, "captions": []}
    with torch.no_grad():
        for r in val:
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[r["prompt"]], prompt_2=[r["prompt"]], prompt_3=[r["prompt"]], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            e_ref = embed_img(r["reference"])
            j = int(np.argmax(r["dino_patch_cos"]))
            z0 = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(int(r["seed_base"]) + j))
            # teacher trajectory states (keep all): re-implement the 8-step rollout keeping states
            pipe.scheduler.set_timesteps(8, device=device); sig = pipe.scheduler.sigmas.to(device, torch.float32); ts = pipe.scheduler.timesteps.to(device)
            e2 = torch.cat([neg_emb, emb], 0); p2 = torch.cat([neg_pool, pooled], 0); z = z0; zs = [z]
            for k in range(8):
                with torch.autocast("cuda", torch.bfloat16):
                    v_all = teacher(hidden_states=torch.cat([z, z], 0), timestep=ts[k].expand(2), encoder_hidden_states=e2, pooled_projections=p2, return_dict=False)[0]
                vu, vc = v_all.chunk(2, 0); v = vu + 7.0 * (vc - vu)
                z = (z.float() + (sig[k + 1] - sig[k]) * v.float()).to(torch.bfloat16); zs.append(z)
            x0s = []
            for s_idx in (5, 6):                                  # the two least-noisy supervised inputs (Delta = 1 from k = 6, 7)
                with torch.autocast("cuda", torch.bfloat16):
                    v = student(hidden_states=zs[s_idx], timestep=ts[s_idx].expand(1), encoder_hidden_states=emb, pooled_projections=pooled, return_dict=False)[0]
                x0s.append(zs[s_idx].float() - sig[s_idx] * v.float())
            sc_x0 = (embed_lat(torch.cat(x0s, 0)) * e_ref[None]).sum(-1).cpu().tolist()
            zsam = torch.randn(args.n_seeds, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(int(r["idx"]) * 7 + 1))
            zK = rollout_student(student, pipe.scheduler, zsam, emb, pooled, 4, device)
            sc_sam = (embed_lat(zK) * e_ref[None]).sum(-1).cpu().tolist()
            out["captions"].append({"idx": int(r["idx"]), "x0_dino": sc_x0, "sample_dino": sc_sam, "teacher_argmax_dino": float(max(r["dino_patch_cos"]))})
    out["summary"] = {"x0_dino_mean": float(np.mean([np.mean(c["x0_dino"]) for c in out["captions"]])),
                      "sample_dino_mean": float(np.mean([np.mean(c["sample_dino"]) for c in out["captions"]])),
                      "teacher_argmax_dino_mean": float(np.mean([c["teacher_argmax_dino"] for c in out["captions"]]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)
    print("[heldout] " + json.dumps(out["summary"]) + f" ({args.ckpt})", flush=True)


if __name__ == "__main__":
    main()
