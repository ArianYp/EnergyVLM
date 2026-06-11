
#!/usr/bin/env python3
"""
DDP image generation for an Arshia-trained MMDiT checkpoint.

Samples COCO val captions across `accelerate`'s worker pool and writes:
  {output_dir}/images/prompt_{global_idx:05d}.png
  {output_dir}/prompt_to_image.json
  {output_dir}/generation_config.json

Designed to be consumed by `score_dit.py` (CLIPScore / VQAScore / DA-Score).

Example (4 GPUs):
  python -m accelerate.commands.launch --num_processes=4 generate_dit.py \
    --ckpt .../training_checkpoints/dit-xl2-vanilla-coco/checkpoints/0400000.pt \
    --coco-captions /lustre/.../COCO/annotations/captions_val2017.json \
    --output-dir logs/eval_dit_xl \
    --num-prompts 1000 --cfg-scale 4.0 --num-steps 50 --batch-size 16 --use-ema
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from mmdit import MMDiT
from samplers_t2i import euler_sampler


def load_coco_captions(path, num, seed):
    with open(path) as f:
        ann = json.load(f)
    caps = [a["caption"].strip() for a in ann["annotations"] if a["caption"].strip()]
    rng = random.Random(seed)
    if num is None or num <= 0 or num >= len(caps):
        caps = caps[:]
        rng.shuffle(caps)
        return caps
    return rng.sample(caps, num)


def build_model(ckpt_obj, latent_size, device):
    ckpt_args = ckpt_obj.get("args") if isinstance(ckpt_obj, dict) else None
    depth = getattr(ckpt_args, "depth", 24) if ckpt_args is not None else 24
    hidden_size = getattr(ckpt_args, "hidden_size", None) if ckpt_args is not None else None
    num_heads = getattr(ckpt_args, "num_heads", None) if ckpt_args is not None else None
    encoder_depth = getattr(ckpt_args, "encoder_depth", 8) if ckpt_args is not None else 8
    model = MMDiT(
        input_size=latent_size,
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads,
        z_dims=[0],
        encoder_depth=encoder_depth,
    ).to(device)
    return model, {"depth": depth, "hidden_size": hidden_size, "num_heads": num_heads, "encoder_depth": encoder_depth}


def load_weights(model, ckpt_obj, use_ema):
    if isinstance(ckpt_obj, dict) and use_ema and "ema" in ckpt_obj:
        sd, key = ckpt_obj["ema"], "ema"
    elif isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        sd, key = ckpt_obj["model"], "model"
    else:
        sd, key = ckpt_obj, "raw"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return key, len(missing), len(unexpected)


def encode_text(prompts, tokenizer, text_model, device, max_length=77):
    tokens = tokenizer(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return text_model(**tokens).last_hidden_state


@torch.no_grad()
def sample_batch(model, vae, prompt_ctx, null_ctx, latent_size,
                 num_steps, cfg_scale, path_type, device, seed,
                 latents_scale=0.18215):
    B = prompt_ctx.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    xT = torch.randn((B, 4, latent_size, latent_size), device=device, generator=g)
    latents = euler_sampler(
        model, xT, prompt_ctx, y_null=null_ctx,
        num_steps=num_steps, cfg_scale=cfg_scale,
        guidance_low=0., guidance_high=1.,
        path_type=path_type, heun=False,
    ).to(torch.float32)
    images = vae.decode(latents / latents_scale).sample
    images = ((images + 1) / 2).clamp(0, 1).mul(255).add(0.5).clamp(0, 255).to("cpu", torch.uint8)
    return [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--coco-captions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=1000, help="0 or negative = all")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--path-type", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-text-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    output_dir = Path(args.output_dir)
    img_dir = output_dir / "images"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()

    latent_size = args.resolution // 8

    # Deterministic shared prompt list (same seed => same sample + same shuffle on all ranks).
    prompts = load_coco_captions(args.coco_captions, args.num_prompts, args.seed)
    if accelerator.is_main_process:
        print(f"[rank 0] Will generate {len(prompts)} images -> {img_dir}")

    # Shard prompts across ranks by stride; each rank gets a disjoint subset.
    my_indices = list(range(rank, len(prompts), world_size))
    my_prompts = [prompts[i] for i in my_indices]

    # Load DiT weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model, arch = build_model(ckpt, latent_size, device)
    key, n_missing, n_unexpected = load_weights(model, ckpt, args.use_ema)
    if accelerator.is_main_process:
        print(f"[rank 0] loaded DiT: arch={arch}, state_dict key='{key}', missing={n_missing}, unexpected={n_unexpected}")
    model.eval()
    del ckpt  # free host RAM after dispatch to GPU

    # Shared auxiliaries: stagger downloads, then everyone reads from HF cache.
    with accelerator.main_process_first():
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_text_id)
        text_model = CLIPTextModel.from_pretrained(args.clip_text_id).to(device).eval()
    null_ctx_once = encode_text([""], tokenizer, text_model, device)

    pbar = tqdm(range(0, len(my_prompts), args.batch_size),
                desc=f"rank {rank}", disable=not accelerator.is_local_main_process)
    for start in pbar:
        batch_prompts = my_prompts[start:start + args.batch_size]
        batch_indices = my_indices[start:start + args.batch_size]
        ctx = encode_text(batch_prompts, tokenizer, text_model, device)
        null_ctx = null_ctx_once.expand(ctx.shape[0], -1, -1).contiguous()
        batch_seed = args.seed + batch_indices[0]  # deterministic w.r.t. global idx
        imgs = sample_batch(
            model, vae, ctx, null_ctx, latent_size,
            args.num_steps, args.cfg_scale, args.path_type, device, seed=batch_seed,
        )
        for idx, img in zip(batch_indices, imgs):
            img.save(img_dir / f"prompt_{idx:05d}.png")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Filename convention is deterministic (global idx), so we reconstruct the mapping
        # on rank 0 without gathering anything from other ranks.
        prompt_to_image = {
            prompts[i]: str(img_dir / f"prompt_{i:05d}.png") for i in range(len(prompts))
        }
        with open(output_dir / "prompt_to_image.json", "w") as f:
            json.dump(prompt_to_image, f, indent=2)
        with open(output_dir / "generation_config.json", "w") as f:
            json.dump({
                "ckpt": str(args.ckpt),
                "coco_captions": str(args.coco_captions),
                "num_prompts": len(prompts),
                "cfg_scale": args.cfg_scale,
                "num_steps": args.num_steps,
                "path_type": args.path_type,
                "resolution": args.resolution,
                "batch_size": args.batch_size,
                "use_ema": args.use_ema,
                "seed": args.seed,
                "clip_text_id": args.clip_text_id,
                "arch": arch,
            }, f, indent=2)
        print(f"[rank 0] wrote {output_dir / 'prompt_to_image.json'}")


if __name__ == "__main__":
    main()
