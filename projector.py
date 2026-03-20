"""
Projector g_phi: Maps DiT hidden representation h_T to DINOv2 feature space.

Per the methodology:
  h_T = Pool(H_T^(L))  -- pooled output of last DiT layer at final denoising step
  z_DiT = g_phi(h_T)  -- projected DiT representation
  z_DINO = f(y)       -- DINOv2 CLS token of generated image y

The projector is trained to align z_DiT with z_DINO via cosine similarity.
"""

import torch
import torch.nn as nn


class DiT2DINOProjector(nn.Module):
    """
    Projects pooled DiT hidden states to DINOv2 feature space.
    
    Args:
        dit_dim: Dimension of DiT hidden states (PixArt: 1152 = 16*72)
        dinov2_dim: Dimension of DINOv2 CLS token (base: 768)
        hidden_dim: Optional hidden dimension for MLP (default: 2x dit_dim)
    """
    def __init__(
        self,
        dit_dim: int = 1152,
        dinov2_dim: int = 768,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * dit_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dit_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dinov2_dim),
        )
        self.dinov2_dim = dinov2_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Pooled DiT hidden states [B, dit_dim]
        Returns:
            z: Projected features [B, dinov2_dim], L2-normalized for cosine similarity
        """
        z = self.mlp(h)
        return nn.functional.normalize(z, p=2, dim=-1)
