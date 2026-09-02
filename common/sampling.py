"""Frozen-teacher Euler rollout and VAE decode for SD3.5 (rectified flow).

The same rollout is used to build the candidate cache, to re-roll the selected candidate during
training, and to sample students at evaluation time, so a candidate scored offline is exactly the
trajectory the student is later trained on.

Euler step on the sigma grid:  z_{k+1} = z_k + (sigma_{k+1} - sigma_k) * v_k,
with classifier-free guidance  v = v_uncond + cfg * (v_cond - v_uncond)  when cfg > 1.
"""
from __future__ import annotations

from pathlib import Path

import torch


@torch.no_grad()
def rollout(transformer, scheduler, z, prompt_embeds, pooled, neg_embeds, neg_pooled,
            num_steps: int, cfg: float, device, keep_states: bool = False):
    """Roll z (bf16, [B,C,H,W]) for `num_steps` Euler steps.

    Returns the endpoint latent, or (states, sigmas) with states[k] the latent after k steps when
    `keep_states` is set. With cfg <= 1 a single conditional forward is used per step.
    """
    B = z.shape[0]
    scheduler.set_timesteps(num_steps, device=device)
    sigmas = scheduler.sigmas.to(device, dtype=torch.float32)
    timesteps = scheduler.timesteps.to(device)
    use_cfg = cfg > 1.0
    if use_cfg:
        emb = torch.cat([neg_embeds.repeat(B, 1, 1), prompt_embeds.repeat(B, 1, 1)], dim=0)
        pool = torch.cat([neg_pooled.repeat(B, 1), pooled.repeat(B, 1)], dim=0)
    states = [z]
    for k in range(num_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if use_cfg:
                v_all = transformer(hidden_states=torch.cat([z, z], dim=0),
                                    timestep=timesteps[k].expand(2 * B),
                                    encoder_hidden_states=emb, pooled_projections=pool,
                                    return_dict=False)[0]
                v_u, v_c = v_all.chunk(2, dim=0)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = transformer(hidden_states=z, timestep=timesteps[k].expand(B),
                                encoder_hidden_states=prompt_embeds.repeat(B, 1, 1),
                                pooled_projections=pooled.repeat(B, 1),
                                return_dict=False)[0]
        z = (z.float() + (sigmas[k + 1] - sigmas[k]) * v.float()).to(torch.bfloat16)
        if keep_states:
            states.append(z)
    return (states, sigmas) if keep_states else z


def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """SD3.5 VAE decode -> image tensor in [-1, 1], float32."""
    z = (latents.to(vae.dtype) / vae.config.scaling_factor) + vae.config.shift_factor
    img = vae.decode(z, return_dict=False)[0]
    return img.clamp(-1.0, 1.0).to(torch.float32)


@torch.no_grad()
def decode_and_save(vae, latents: torch.Tensor, out_dir: Path, name: str | None = None) -> None:
    """Decode a latent batch to out_dir/cand{j}.png (or out_dir/{name}.png for a single image)."""
    from torchvision.utils import save_image
    img = (vae_decode(vae, latents) + 1) / 2
    out_dir.mkdir(parents=True, exist_ok=True)
    if name is not None:
        assert img.shape[0] == 1, f"name= requires a single-image batch, got {img.shape[0]}"
        save_image(img[0].cpu(), out_dir / f"{name}.png")
        return
    for j in range(img.shape[0]):
        save_image(img[j].cpu(), out_dir / f"cand{j}.png")


def encode_prompt(pipe, text: str, device):
    """(sequence embeddings, pooled embeddings) for one prompt, using all three SD3.5 text encoders."""
    emb, _, pooled, _ = pipe.encode_prompt(prompt=[text], prompt_2=[text], prompt_3=[text],
                                           do_classifier_free_guidance=False, device=device,
                                           num_images_per_prompt=1)
    return emb, pooled


def candidate_noise(seed_base: int, j: int, shape, device):
    """Initial noise of candidate j of a caption; the trainer re-rolls from the same seed."""
    g = torch.Generator(device=device).manual_seed(int(seed_base) + int(j))
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)
