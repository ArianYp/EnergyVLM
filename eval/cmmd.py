"""CMMD (Jayasumana et al., CVPR 2024), reproducing the reference implementation exactly.

Reference: google-research/google-research/cmmd (JAX) and its PyTorch port sayakpaul/cmmd-pytorch.
    images      sorted file list, RGB, CENTER-CROP to square, bicubic resize to 336x336, [0, 1]
    model       CLIP ViT-L/14 @ 336px  (openai/clip-vit-large-patch14-336), CLIP mean/std
    embeddings  image projection, L2-normalised to unit length
    distance    MMD^2 with Gaussian RBF kernel, sigma = 10, biased (V-statistic) estimator, x 1000
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CLIP_ID = "openai/clip-vit-large-patch14-336"
RESOLUTION = 336
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
SIGMA, SCALE = 10.0, 1000.0


def load_clip(device):
    from transformers import CLIPVisionModelWithProjection
    return CLIPVisionModelWithProjection.from_pretrained(CLIP_ID).to(device).eval()


def preprocess(path) -> torch.Tensor:
    """PIL -> RGB -> center crop to square -> bicubic 336x336 -> [0,1] float tensor [3,H,W]."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((RESOLUTION, RESOLUTION), resample=Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def clip_embed(paths, model, device, bs: int = 32) -> torch.Tensor:
    """Unit-normalised CLIP image embeddings, [N, 768], on CPU."""
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(p) for p in paths[i:i + bs]])
        x = ((x - MEAN) / STD).to(device)
        e = model(pixel_values=x).image_embeds.float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out, 0)


def cmmd(x: torch.Tensor, y: torch.Tensor, sigma: float = SIGMA, scale: float = SCALE) -> float:
    """Biased MMD^2 estimate (diagonal terms included), scaled by 1000, on unit-norm embeddings."""
    gamma = 1.0 / (2 * sigma ** 2)

    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))
    return float((k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()).item() * scale)
