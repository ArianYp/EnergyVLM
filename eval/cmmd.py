"""CLIP Maximum Mean Discrepancy (CMMD): Gaussian-RBF MMD^2 on CLIP image embeddings."""
from __future__ import annotations

import torch
from PIL import Image


@torch.no_grad()
def clip_embed(paths, model, proc, device, bs: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        embs.append(model(pixel_values=px).image_embeds.float().cpu())
    return torch.cat(embs, 0)


def cmmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0, scale: float = 1000.0) -> float:
    def k(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma * sigma))
    return float((k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()).item() * scale)
