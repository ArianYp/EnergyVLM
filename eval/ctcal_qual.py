#!/usr/bin/env python3
"""Qualitative comparison on the EXACT prompts of CTCal's figures (docs/bench/ctcal_prompts.json): our
4-step students side by side with the 28-step guided teacher, same prompt, same noise per column.

Models (one pipeline load, checkpoints swapped in place):
  teacher-28      frozen SD3.5-M, 28 steps, cfg 7            (what our students are distilled from)
  COCO ours       scored selection + projector reward, K=10   (the COCO-caption recipe)
  bench random    benchmark prompts, random candidate         (in-domain control)
  bench argmax    benchmark prompts, evaluator-argmax         (the GORS-style arm; our best CompBench)
Writes one contact sheet per seed. Checkpoint paths default to the group-readable ones of
docs/CHECKPOINTS.md; pass --ckpt_ours / --ckpt_random / --ckpt_argmax to use others.

    python eval/ctcal_qual.py --seeds 0,1            (one GPU, scripts/ctcal_qual.lsf)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import decode_and_save, encode_prompt, rollout  # noqa: E402

CKPT_ROOT = "/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "docs" / "bench" / "ctcal_prompts.json"))
    ap.add_argument("--out", default="out/ctcal_qual")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--ckpt_ours", default=f"{CKPT_ROOT}/phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472/checkpoint_avg_last5.pt")
    ap.add_argument("--ckpt_random", default=f"{CKPT_ROOT}/phaseW_B2_bench-k10-hp1-acc4_s0_155627/checkpoint_avg_last5.pt")
    ap.add_argument("--ckpt_argmax", default=f"{CKPT_ROOT}/phaseW_CD_bench_hard_bench-k10-hp1-acc4_s0_155631/checkpoint_avg_last5.pt")
    args = ap.parse_args()
    models = [
        ("SD3.5-M teacher (28 steps, cfg 7)", "base", 28, 7.0),
        ("ours, COCO captions (4 steps)", args.ckpt_ours, 4, 1.0),
        ("bench prompts, random pick (4 steps)", args.ckpt_random, 4, 1.0),
        ("bench prompts, evaluator-argmax (4 steps)", args.ckpt_argmax, 4, 1.0),
    ]
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = json.load(open(args.prompts))
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    base_sd = {k: v.clone() for k, v in pipe.transformer.state_dict().items()}
    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)
    lat_c = pipe.transformer.config.in_channels; H = args.height // pipe.vae_scale_factor
    for name, ckpt, steps, cfg in models:
        tag = _tag(name)
        if ckpt == "base":
            pipe.transformer.load_state_dict(base_sd)
        else:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            miss, unexp = pipe.transformer.load_state_dict(ck["model"], strict=False)
            assert not miss and not unexp, (miss[:3], unexp[:3])
            pipe.transformer.to(dtype=torch.bfloat16).eval(); del ck
        print(f"[gen] {name}  steps={steps} cfg={cfg}", flush=True)
        for r in prompts:
            with torch.no_grad():
                emb, pooled = encode_prompt(pipe, r["prompt"], device)
                for sd in seeds:
                    f = out / "images" / f"{tag}__p{r['idx']:02d}__s{sd}.png"
                    if f.exists():
                        continue
                    z = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16,
                                    generator=torch.Generator(device=device).manual_seed(sd * 100000 + r["idx"]))
                    lat = rollout(pipe.transformer, pipe.scheduler, z, emb, pooled, neg_emb, neg_pool, steps, cfg, device)
                    decode_and_save(pipe.vae, lat, f.parent, name=f.stem)
    sheet(out, prompts, seeds, models)


def _tag(name: str) -> str:
    return name.split("(")[0].strip().replace(" ", "_").replace(",", "")


def sheet(out: Path, prompts, seeds, models, T=248, pad=6, cap=64) -> None:
    font = ImageFont.load_default()
    tags = [_tag(m[0]) for m in models]
    for sd in seeds:
        rows = list(prompts)
        W = pad + len(models) * (T + pad); Hh = pad + 30 + len(rows) * (T + cap + pad)
        im = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(im)
        dr.text((pad, pad), f"CTCal figure prompts, verbatim — our models, seed {sd}, 512 px", fill="#000", font=font)
        for c, (name, _, steps, cfg) in enumerate(models):
            dr.multiline_text((pad + c * (T + pad), pad + 14), "\n".join(textwrap.wrap(name, 34)[:2]), fill="#333", font=font)
        for i, r in enumerate(rows):
            y = pad + 30 + i * (T + cap + pad)
            for c, tag in enumerate(tags):
                x = pad + c * (T + pad)
                f = out / "images" / f"{tag}__p{r['idx']:02d}__s{sd}.png"
                if f.exists():
                    im.paste(Image.open(f).convert("RGB").resize((T, T)), (x, y))
                else:
                    dr.rectangle([x, y, x + T, y + T], outline="#ccc")
            dr.multiline_text((pad + 2, y + T + 2), "\n".join(textwrap.wrap(f'[{r["category"]}] "{r["prompt"]}"  — {r["src"]}', 96)[:3]), fill="#000", font=font)
        p = out / f"sheet_seed{sd}.jpg"; im.save(p, quality=90); print("wrote", p)


if __name__ == "__main__":
    main()
