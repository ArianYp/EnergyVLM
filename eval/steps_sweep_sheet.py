#!/usr/bin/env python3
"""Visual step sweep for a distilled student: the same prompt and noise at 1..28 denoising steps
(guidance 1, the scheduler grid at each count). The source of docs/figs/steps_sweep.jpg; the measured
numbers behind it are in docs/CHECKPOINTS.md.

    python eval/steps_sweep_sheet.py --ckpt checkpoints/<run>/checkpoint_avg_last5.pt   (one GPU)
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import decode_and_save, encode_prompt, rollout  # noqa: E402

STEPS = [1, 2, 4, 6, 8, 16, 28]
PROMPTS = ["a white piano and a black bench", "a horse on the left of a car",
           "a metallic desk lamp and a leather jacket", "A wizard stirs a bubbling cauldron and a cat sits on the table"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/"
                                      "phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472/checkpoint_avg_last5.pt")
    ap.add_argument("--out", default="out/steps_sweep")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = pipe.transformer.load_state_dict(ck["model"], strict=False); assert not miss and not unexp; del ck
    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)
        lat_c = pipe.transformer.config.in_channels; H = args.height // pipe.vae_scale_factor
        for pi, pr in enumerate(PROMPTS):
            emb, pooled = encode_prompt(pipe, pr, device)
            for s in STEPS:
                f = out / "img" / f"p{pi}_s{s}.png"
                if f.exists():
                    continue
                z = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(pi))
                decode_and_save(pipe.vae, rollout(pipe.transformer, pipe.scheduler, z, emb, pooled, neg_emb, neg_pool, s, 1.0, device), f.parent, name=f.stem)
    T, pad, cap = 232, 6, 40; font = ImageFont.load_default()
    im = Image.new("RGB", (pad + len(STEPS) * (T + pad), pad + 16 + len(PROMPTS) * (T + cap + pad)), "white"); dr = ImageDraw.Draw(im)
    dr.text((pad, pad), "same student, same noise, N denoising steps (guidance 1.0, scheduler grid)", fill="#000", font=font)
    for c, s in enumerate(STEPS):
        dr.text((pad + c * (T + pad), pad + 16 - 10), f"{s} steps", fill="#333", font=font)
    for r, pr in enumerate(PROMPTS):
        y = pad + 16 + r * (T + cap + pad)
        for c, s in enumerate(STEPS):
            f = out / "img" / f"p{r}_s{s}.png"
            if f.exists():
                im.paste(Image.open(f).convert("RGB").resize((T, T)), (pad + c * (T + pad), y))
        dr.multiline_text((pad + 2, y + T + 2), "\n".join(textwrap.wrap(pr, 110)[:2]), fill="#000", font=font)
    im.save(out / "sheet.jpg", quality=90); print("wrote", out / "sheet.jpg")


if __name__ == "__main__":
    main()
