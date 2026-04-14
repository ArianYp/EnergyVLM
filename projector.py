"""
Projector g_phi: Maps mean-pooled DiT hidden states to DINOv2 feature space.

  h_T    = H_T.mean(dim=1)  -- mean-pool over spatial tokens from teacher's last block
  z_DiT  = g_phi(h_T)       -- projected representation (raw, unnormalised)
  z_DINO = f(y)[:, 0, :]    -- DINOv2 CLS token of teacher-generated image y

Alignment loss: L_align = 1 - cosine(normalize(z_DiT), normalize(z_DINO))
L2 normalisation is done in the training script, not here, following REPA convention.

Why mean-pool instead of per-token alignment (as in REPA)?
  REPA operates at 256×256 where the DiT's 16×16 = 256 spatial tokens exactly match
  DINOv2's 16×16 = 256 patch tokens, enabling direct token-to-token correspondence.
  SD3.5 at 512×512 produces 32×32 = 1024 DiT tokens vs DINOv2's 256 patch tokens —
  no direct spatial correspondence is possible, so global mean-pooling is the right
  aggregation strategy.

Architecture follows REPA's build_mlp:
  - projector_dim=2048 (fixed, independent of input width — REPA default)
  - SiLU activations (REPA uses SiLU, not GELU)
  - 3 layers: input → projector_dim → projector_dim → z_dim
"""

import torch
import torch.nn as nn


class DiT2DINOProjector(nn.Module):
    """
    Projects mean-pooled DiT hidden states into DINOv2 feature space.

    Args:
        dit_dim:       Dimension of input DiT hidden states (e.g. 1536 for SD3.5).
        dinov2_dim:    Dimension of DINOv2 output (768 for ViT-B, 1024 for ViT-L).
        projector_dim: Hidden dimension of the MLP. Fixed at 2048 following REPA
                       (independent of dit_dim, unlike the previous 2×dit_dim rule).
    """

    def __init__(
        self,
        dit_dim: int = 1536,
        dinov2_dim: int = 768,
        projector_dim: int = 2048,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            # Normalise raw teacher hidden states before projection.
            # h_T comes from the last transformer block with no guaranteed scale;
            # without this, the first linear layer sees wildly varying input magnitudes
            # that destabilise early projector training.  elementwise_affine=False
            # keeps the normalisation parameter-free (no learnable scale/bias) so it
            # cannot be undone, matching standard practice in representation learning.
            nn.LayerNorm(dit_dim, elementwise_affine=False),
            nn.Linear(dit_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, dinov2_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Mean-pooled DiT hidden states [B, dit_dim]
        Returns:
            z: Projected features [B, dinov2_dim], NOT L2-normalised.
               Caller is responsible for normalisation before cosine similarity.
        """
        return self.mlp(h)
