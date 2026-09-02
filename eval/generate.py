#!/usr/bin/env python3
"""Sample one image per prompt from the base model or a distilled student, at fixed step counts.

The initial noise of prompt `idx` is seeded by `--seed + idx`, so every model and step count is
evaluated on identical noise. Students distilled against a guided teacher have the guidance
absorbed into their conditional prediction and must be sampled with `--cfg 1`; the base model is
sampled with its native guidance (`--cfg 7`).

`--out_root` and `--prompts_json` accept comma-separated lists of equal length so one process can
serve several prompt pools after a single model load.

Output: {out_root}/images/{label}/p{idx:05d}/s{steps}/cand{j}.png
Sharded with RANK / WORLD_SIZE (prompt idx % world), resumable.
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
from common.sampling import decode_and_save, encode_prompt, rollout  # noqa: E402

SEED_STRIDE = 1_000_000   # keeps candidate j > 0 disjoint from every prompt's candidate 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--prompts_json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", required=True, help="'base' or a checkpoint .pt")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--steps_list", default="4")
    ap.add_argument("--n_seeds", type=int, default=1, help="images per prompt (cand0..candN-1)")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    roots = [Path(x) for x in args.out_root.split(",") if x]
    jsons = [x for x in args.prompts_json.split(",") if x]
    if len(roots) != len(jsons):
        raise SystemExit("--out_root and --prompts_json must have the same number of entries")
    pools = [(r, json.loads(Path(j).read_text())) for r, j in zip(roots, jsons)]
    steps_list = [int(x) for x in args.steps_list.split(",")]
    if rank == 0:
        for r, pool in pools:
            r.mkdir(parents=True, exist_ok=True)
            pf = r / "prompts.json"
            if not pf.exists():
                pf.write_text(json.dumps(pool, indent=1))
            print(f"[r0] label={args.label} ckpt={args.checkpoint} cfg={args.cfg} | {r}: "
                  f"{len(pool)} prompts x steps {steps_list}", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    if args.checkpoint != "base":
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
        assert not unexpected and not missing, (unexpected[:3], missing[:3])
        pipe.transformer.to(dtype=torch.bfloat16).eval()
        del ck

    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)
    lat_c = pipe.transformer.config.in_channels
    h_lat = args.height // pipe.vae_scale_factor

    for out_root, pool in pools:
        mine = [it for it in pool if it["idx"] % world == rank]
        for n, item in enumerate(mine):
            pidx, prompt = item["idx"], item["prompt"]
            pdir = out_root / "images" / args.label / f"p{pidx:05d}"
            todo = [(s, j) for s in steps_list for j in range(args.n_seeds)
                    if not (pdir / f"s{s}" / f"cand{j}.png").exists()]
            if not todo:
                continue
            with torch.no_grad():
                emb, pooled = encode_prompt(pipe, prompt, device)
            z0 = {}
            for j in sorted({j for _, j in todo}):
                g = torch.Generator(device=device).manual_seed(args.seed + pidx + j * SEED_STRIDE)
                z0[j] = torch.randn(1, lat_c, h_lat, h_lat, device=device, dtype=torch.bfloat16, generator=g)
            for s, j in todo:
                lat = rollout(pipe.transformer, pipe.scheduler, z0[j].clone(), emb, pooled,
                              neg_emb, neg_pool, s, args.cfg, device)
                decode_and_save(pipe.vae, lat, pdir / f"s{s}", name=f"cand{j}")
            if n % 20 == 0:
                print(f"[r{rank}] {n + 1}/{len(mine)} prompts", flush=True)
        print(f"[r{rank}] done label={args.label} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
