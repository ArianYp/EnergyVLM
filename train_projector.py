#!/usr/bin/env python3
"""
Train the projector g_phi to align DiT internal representations with DINOv2.

Methodology:
  - Generate images with PixArt (DiT) for text prompts
  - At final denoising step T, extract H_T^(L) from last DiT block
  - Pool: h_T = Pool(H_T^(L))
  - Get DINOv2 CLS: z_DINO = f(y) for generated image y
  - Project: z_DiT = g_phi(h_T)
  - Minimize E = 1 - cos(z_DiT, z_DINO)

Usage:
  python train_projector.py --num_steps 5000 --batch_size 4 --lr 1e-4
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import wandb
import torch.nn.functional as F
from diffusers import PixArtAlphaPipeline
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from projector import DiT2DINOProjector


# --- DiT hidden state extraction ---

def _make_hook(capture_dict: dict, is_final_step: list):
    """Create a forward hook that captures output only at the final denoising step."""
    def hook(module, input, output):
        if is_final_step[0]:
            # output may be tensor or tuple (hidden_states,); BasicTransformerBlock returns tensor
            hidden = output[0] if isinstance(output, tuple) else output
            # shape: [batch, seq_len, inner_dim]
            capture_dict["hidden"] = hidden.detach()
    return hook


def extract_dit_hidden_at_final_step(pipe, latents, prompt_embeds, prompt_attention_mask, 
                                     added_cond_kwargs, device, guidance_scale=1.0):
    """
    Run the denoising loop and capture hidden states from the last DiT block at the final step.
    
    Returns:
        h_T: Pooled hidden states [B, inner_dim]
        image: Generated image (PIL or tensor) for DINOv2
    """
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    image_processor = pipe.image_processor
    
    # Prepare
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([pipe.encode_prompt("", 1, 1, device)[0], prompt_embeds], dim=0)
        # Simplified: use empty negative for now; for proper CFG we'd need full encode_prompt
        # For projector training, guidance_scale=1 is often sufficient
        do_classifier_free_guidance = False
    
    timesteps = scheduler.timesteps
    capture_dict = {}
    is_final_step = [False]
    
    # Register hook on last transformer block (before norm_out, proj_out)
    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final_step))
    
    try:
        for i, t in enumerate(timesteps):
            is_final_step[0] = (i == len(timesteps) - 1)
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            
            if not torch.is_tensor(t):
                t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            else:
                t_tensor = t.unsqueeze(0).to(device) if t.dim() == 0 else t
            t_tensor = t_tensor.expand(latent_model_input.shape[0])
            
            with torch.no_grad():
                noise_pred = transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=t_tensor,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # PixArt flow matching: transformer outputs 2x channels, take first half
                latent_channels = pipe.transformer.config.in_channels
                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
            
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode to image (VAE output in [-1, 1], convert to [0, 1])
        with torch.no_grad():
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)  # [B, 3, H, W] in [0, 1]
        
    finally:
        handle.remove()
    
    if "hidden" not in capture_dict:
        raise RuntimeError("Failed to capture DiT hidden states at final step")
    
    H_T = capture_dict["hidden"]  # [B, seq_len, inner_dim]
    h_T = H_T.mean(dim=1)  # mean pooling: [B, inner_dim]
    
    return h_T, image


def load_coco_captions(coco_path: str) -> list[str]:
    """Load captions from COCO annotations. Returns list of caption strings."""
    path = Path(coco_path)
    if path.is_dir():
        # Try standard locations
        for ann_file in ["annotations/captions_train2017.json", "annotations/captions_train2014.json", "captions_train2017.json"]:
            candidate = path / ann_file
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No COCO captions JSON found under {coco_path}")
    with open(path) as f:
        ann = json.load(f)
    return [a["caption"] for a in ann["annotations"]]


def get_prompts(num_prompts: int, caption_pool: list[str] | None, seed: int) -> list[str]:
    """Sample prompts from caption pool (COCO) or fallback to built-in prompts."""
    pool = caption_pool if caption_pool else []
    if pool:
        rng = random.Random(seed)
        indices = [rng.randint(0, len(pool) - 1) for _ in range(num_prompts)]
        return [pool[i] for i in indices]
    # Fallback built-in prompts
    base_prompts = [
        "A photo of a cat sitting on a wooden table",
        "A landscape with mountains and a lake at sunset",
        "A bowl of fresh fruit on a kitchen counter",
        "A person walking a dog in the park",
        "A vintage car parked on a street",
        "A cozy living room with a fireplace",
        "An astronaut floating in space",
        "A delicious pizza with various toppings",
        "A butterfly on a colorful flower",
        "A modern office with large windows",
        "A sailboat on calm blue water",
        "A plate of sushi with chopsticks",
        "A snow-covered forest in winter",
        "A cup of coffee with latte art",
        "A bookshelf filled with old books",
    ]
    rng = random.Random(seed)
    return [base_prompts[rng.randint(0, len(base_prompts) - 1)] for _ in range(num_prompts)]


def main():
    parser = argparse.ArgumentParser(description="Train DiT-to-DINOv2 projector")
    parser.add_argument("--output_dir", type=str, default="checkpoints/projector")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-XL-2-512x512",
                        help="PixArt model: XL-2-512x512 (lower VRAM) or XL-2-1024-MS")
    parser.add_argument("--dinov2_id", type=str, default="facebook/dinov2-base")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--images_dir", type=str, default="generated_images", help="Directory to save generated images")
    parser.add_argument("--coco_path", type=str, default="/data/gpfs/datasets/COCO",
                        help="Path to COCO dataset root (uses annotations/captions_train2017.json). Set to '' to use built-in prompts.")
    parser.add_argument("--wandb_project", type=str, default="energy-based-dit", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    # Load COCO captions if path provided
    caption_pool = None
    if args.coco_path and Path(args.coco_path).exists():
        caption_pool = load_coco_captions(args.coco_path)
        print(f"Loaded {len(caption_pool)} COCO captions from {args.coco_path}")
    else:
        print("Using built-in prompts (COCO path not set or not found)")
    
    # --- Load models ---
    print("Loading PixArt pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # Use pipeline's default scheduler (preserves prediction_type for flow matching)
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    
    # PixArt config
    dit_dim = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim  # 1152
    
    print("Loading DINOv2...")
    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id)
    dinov2_model.eval()
    dinov2_model = dinov2_model.to(device)
    dinov2_dim = dinov2_model.config.hidden_size  # 768 for base
    
    # Projector
    projector = DiT2DINOProjector(dit_dim=dit_dim, dinov2_dim=dinov2_dim).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    
    # DINOv2 preprocessing: resize to 224, ImageNet normalize
    # image from VAE is [B, 3, H, W] in [0, 1]
    
    # --- W&B ---
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    # --- Training loop ---
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(range(args.num_steps), desc="Training projector")
    
    for step in pbar:
        projector.train()
        optimizer.zero_grad()
        
        # Get prompts for this batch (from COCO or built-in)
        prompts = get_prompts(args.batch_size, caption_pool, seed=args.seed + step)
        
        # Encode prompts
        with torch.no_grad():
            (
                prompt_embeds,
                prompt_attention_mask,
                _,
                _,
            ) = pipe.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
                device=device,
                clean_caption=True,
            )
        
        # Prepare latents
        latent_channels = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            args.batch_size,
            latent_channels,
            args.height,
            args.width,
            prompt_embeds.dtype,
            device,
            generator=torch.Generator(device).manual_seed(args.seed + step + 1),
        )
        
        # Micro-conditions for PixArt
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if pipe.transformer.config.sample_size == 128:
            resolution = torch.tensor([[args.height, args.width]], dtype=prompt_embeds.dtype, device=device).repeat(args.batch_size, 1)
            aspect_ratio = torch.tensor([[float(args.height / args.width)]], dtype=prompt_embeds.dtype, device=device).repeat(args.batch_size, 1)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        
        # Set timesteps
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        
        # Extract h_T and generate image
        h_T, image_tensor = extract_dit_hidden_at_final_step(
            pipe,
            latents,
            prompt_embeds,
            prompt_attention_mask,
            added_cond_kwargs,
            device,
            guidance_scale=1.0,
        )
        
        # DINOv2: convert to PIL, use processor for resize+normalize
        pil_images = [
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            for img in image_tensor
        ]
        # Save generated images (every save_every steps to limit disk usage)
        if (step + 1) % args.save_every == 0:
            for i, pil_img in enumerate(pil_images):
                pil_img.save(images_dir / f"step{step+1:06d}_img{i}.png")
        dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
        pixel_values = dinov2_inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            dinov2_outputs = dinov2_model(pixel_values=pixel_values)
            z_DINO = dinov2_outputs.last_hidden_state[:, 0, :]  # CLS token
            z_DINO = F.normalize(z_DINO, p=2, dim=-1)
        
        z_DiT = projector(h_T.float().to(device))  # h_T is float16 from DiT; projector expects float32
        loss = 1.0 - (z_DiT * z_DINO).sum(dim=-1).mean()  # E = 1 - cos
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        
        if not args.no_wandb:
            wandb.log({"train/loss": loss_val, "step": step + 1})
        
        if (step + 1) % args.log_every == 0:
            print(f"Step {step+1}/{args.num_steps} | Loss: {loss_val:.4f}")
        
        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"projector_step{step+1}.pt")
            torch.save(projector.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            if not args.no_wandb:
                wandb.log({"checkpoint_step": step + 1})
    
    # Final save
    torch.save(projector.state_dict(), os.path.join(args.output_dir, "projector_final.pt"))
    print("Training complete.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
