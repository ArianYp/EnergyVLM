#!/usr/bin/env python3
"""Teacher candidates for the benchmark-prompt pool (docs/bench/): N candidates per T2I-CompBench++
TRAIN prompt from the frozen teacher (K Euler steps, guidance cfg), decoded to PNG in the evaluation
image layout so eval/compbench.py can score every candidate with the official evaluator of its
category:

    <out_root>/images/<label>/p{idx:05d}/s{K}/cand{j}.png

Candidate j of prompt idx starts from manual_seed(seed + idx*1000 + j): exactly the noise the trainer
re-rolls (seed_base = seed + idx*1000, common.sampling.candidate_noise), so the scored image IS the
trajectory the student is later trained on. Rank-sharded by idx % WORLD_SIZE; resumable (a prompt
with all N files present is skipped).

The same script generates the REFERENCE candidates of the reward arm (docs/bench/): the teacher at
its own documented setting, --K 40 --cfg 4.5 --N 8 --label bench_ref_k40cfg45.

    CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 python data/build_bench_candidates.py --N 16 --K 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import candidate_noise, decode_and_save, encode_prompt, rollout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pools/bench_train/compbench_prompts.json")
    ap.add_argument("--out_root", default="cache/bench")
    ap.add_argument("--label", default="bench_teacher_k10")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local); device = torch.device(f"cuda:{local}")
    pool = json.load(open(args.pool))
    mine = [r for r in pool if int(r["idx"]) % world == rank]
    print(f"[r{rank}/{world}] {len(mine)} prompts, N={args.N} K={args.K} cfg={args.cfg} {args.height}px seed={args.seed}", flush=True)
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)
    lat_c = pipe.transformer.config.in_channels; H = args.height // pipe.vae_scale_factor
    done = 0
    for n, r in enumerate(mine):
        idx, prompt = int(r["idx"]), r["prompt"]
        pdir = Path(args.out_root) / "images" / args.label / f"p{idx:05d}" / f"s{args.K}"
        if all((pdir / f"cand{j}.png").exists() for j in range(args.N)):
            continue
        with torch.no_grad():
            emb, pooled = encode_prompt(pipe, prompt, device)
            seed_base = args.seed + idx * 1000
            z0 = torch.cat([candidate_noise(seed_base, j, (1, lat_c, H, H), device) for j in range(args.N)], 0)
            lat = rollout(pipe.transformer, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, args.K, args.cfg, device)
            decode_and_save(pipe.vae, lat, pdir)
        done += 1
        if done % 20 == 0:
            print(f"[r{rank}] {n + 1}/{len(mine)} prompts", flush=True)
    print(f"[r{rank}] done ({done} generated, {len(mine) - done} skipped)", flush=True)


if __name__ == "__main__":
    main()
