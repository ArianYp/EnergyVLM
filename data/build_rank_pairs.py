#!/usr/bin/env python3
"""Prompt-aware SELECTION, stage 1 (docs/rank/): teacher images of each caption and its negatives.

For every caption of the training cache that has negatives, and for --heldout_n captions of the
large cache outside it, the frozen 8-step / guidance-7 TEACHER (the generator of the candidate
cache) samples the positive prompt and each negative from ONE shared noise (seed_base + --seed_offset,
disjoint from the candidates' seed_base + j) and the offline DINOv2 patch-mean scorer embeds the
decoded images. Only embeddings are stored (positive, negatives, photo): stage 2 (train/rank_head.py)
trains the ranking head on true DINO features of teacher images, the distribution of the candidates
it re-scores in stage 3 (data/rescore_cache.py).

    python3 data/build_rank_pairs.py --cache cache/train_3k --heldout_cache cache/train --out_dir cache/rank_pairs
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data.build_negatives as BN  # noqa: E402
from data.build_negatives import negatives_for  # noqa: E402


def load_cache(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln); out[int(r["idx"])] = r
    return out


@torch.no_grad()
def teacher_rollout_multi(teacher, scheduler, z0, emb, pooled, neg_emb, neg_pool, K, cfg, device):
    """CFG Euler rollout of m DIFFERENT prompts from the same noise: z0 [m,C,H,W], emb [m,L,D]."""
    m = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sig = scheduler.sigmas.to(device, torch.float32); ts = scheduler.timesteps.to(device)
    e = torch.cat([neg_emb.repeat(m, 1, 1), emb], 0); p = torch.cat([neg_pool.repeat(m, 1), pooled], 0)
    z = z0
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v_all = teacher(hidden_states=torch.cat([z, z], 0), timestep=ts[k].expand(2 * m),
                            encoder_hidden_states=e, pooled_projections=p, return_dict=False)[0]
        v_u, v_c = v_all.chunk(2, 0)
        z = (z.float() + (sig[k + 1] - sig[k]) * (v_u + cfg * (v_c - v_u)).float()).to(torch.bfloat16)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train_3k")
    ap.add_argument("--heldout_cache", default="cache/train")
    ap.add_argument("--heldout_n", type=int, default=300)
    ap.add_argument("--ref", default="cache/reward/ref_emb.pt")
    ap.add_argument("--heldout_ref", default="cache/reward/ref_emb_118k.pt")
    ap.add_argument("--proj_manifest", default="cache/latents/manifest.jsonl")
    ap.add_argument("--families", default="color,texture,verb,shape,count")
    ap.add_argument("--count_min_delta", type=int, default=2)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out_dir", default="cache/rank_pairs")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed_offset", type=int, default=977)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    BN.FAMILIES = args.families.split(","); BN.COUNT_MIN_DELTA = args.count_min_delta
    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train = load_cache(args.cache); held_all = load_cache(args.heldout_cache)
    seen_p = {r["prompt"] for r in train.values()}; seen_r = {r["reference"] for r in train.values()}
    if Path(args.proj_manifest).exists():
        for ln in open(args.proj_manifest):
            if ln.strip():
                r = json.loads(ln); seen_p.add(r["prompt"]); seen_r.add(r["reference"])
    held_ids = sorted(i for i, r in held_all.items() if r["prompt"] not in seen_p and r["reference"] not in seen_r)
    random.Random(0).shuffle(held_ids)
    splits = {"train": [(i, train[i]) for i in sorted(train)], "heldout": [(i, held_all[i]) for i in held_ids]}
    refs = {"train": torch.load(args.ref, map_location="cpu", weights_only=False),
            "heldout": torch.load(args.heldout_ref, map_location="cpu", weights_only=False)}

    from diffusers import StableDiffusion3Pipeline
    from train.distill import PatchScorer
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for mdl in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        mdl.to(dtype=torch.bfloat16).eval()
    teacher = pipe.transformer.eval()
    scorer = PatchScorer(args.dino_id, device)
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    lat_c = teacher.config.in_channels; H = args.height // pipe.vae_scale_factor
    for split, items in splits.items():
        recs = []; t0 = time.time(); n_target = args.heldout_n if split == "heldout" else len(items)
        for idx, r in items:
            if (args.limit and len(recs) >= args.limit) or len(recs) >= n_target:
                break
            negs = negatives_for(r["prompt"], args.m)
            if not negs or (split == "heldout" and len(negs) < 2):
                continue
            prompts = [r["prompt"]] + [n["prompt"] for n in negs]
            with torch.no_grad():
                emb, _, pooled, _ = pipe.encode_prompt(prompt=prompts, prompt_2=prompts, prompt_3=prompts, do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
                g = torch.Generator(device=device).manual_seed(int(r["seed_base"]) + args.seed_offset)
                z0 = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=g).expand(len(prompts), -1, -1, -1).contiguous()
                zK = teacher_rollout_multi(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, args.K, args.cfg, device)
                e = scorer.embed_latents(pipe.vae, zK).cpu()
            recs.append({"idx": int(idx), "prompt": r["prompt"], "negatives": negs, "e_pos": e[0], "e_negs": e[1:], "u": refs[split][int(idx)].float()})
            if len(recs) % 100 == 0:
                print(f"[{split}] {len(recs)} captions {time.time() - t0:.0f}s", flush=True)
        torch.save(recs, out_dir / f"pairs_{split}.pt")
        print(f"[{split}] wrote {len(recs)} captions, {sum(len(x['negatives']) for x in recs)} negatives -> {out_dir / f'pairs_{split}.pt'}", flush=True)


if __name__ == "__main__":
    main()
