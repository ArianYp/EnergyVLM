#!/usr/bin/env python3
"""One correct way to build the SD3.5-Medium pipeline. Use this instead of hand-rolling it.

Why this file exists
--------------------
`StableDiffusion3Pipeline.from_pretrained(..., torch_dtype=torch.bfloat16)` does **not** cast every
submodule. CLIP's `text_projection` stays fp16, and the first call to `encode_prompt` dies with

    RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != c10::BFloat16

`exp0/phaseA_generate.py` has carried the per-component cast (and a comment about it) since the
start, but the fix lives in that file rather than anywhere importable, so every new script has to
rediscover it. It cost jobs **89604** and **89591** on the same day. Hence this helper.
"""
from __future__ import annotations

import torch


def load_sd35(model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda",
              checkpoint=None, dtype=torch.bfloat16):
    """Pipeline in `dtype` throughout, optionally with student weights loaded.

    checkpoint: None / "base" for the frozen teacher, else a path to a .pt holding ck["model"].
    Returns (pipe, transformer). The transformer is eval() with requires_grad_(False).
    """
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    # the whole point of this helper — see module docstring
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2,
              pipe.text_encoder_3):
        m.to(dtype=dtype)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    if checkpoint and checkpoint != "base":
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
        assert not missing, f"missing keys: {list(missing)[:5]}"
        pipe.transformer.to(dtype=dtype).eval()
        del ck

    return pipe, pipe.transformer


def sigmas_and_timesteps(pipe, num_steps, device):
    """The (sigmas, timesteps) the samplers in this repo use. sigmas is fp32, length num_steps+1."""
    pipe.scheduler.set_timesteps(num_steps, device=device)
    return (pipe.scheduler.sigmas.to(device, torch.float32),
            pipe.scheduler.timesteps.to(device))
