"""Deterministic, paired, order-independent noise for the PIE-Bench comparison.

Every random tensor is a pure function of a content tuple

    (evaluation_seed, record_id, step_index, n_avg_index, purpose)

hashed with SHA-256. Nothing depends on which model runs first, on batch size, GPU count,
distributed rank, dataset order, resuming, or which other records were skipped -- the two models
therefore see bitwise-identical noise for a record, which is what makes the comparison paired.

Python's built-in hash() is salted per process (PYTHONHASHSEED) and must never be used here.
"""
from __future__ import annotations

import hashlib

import torch

SEED_BITS = 1 << 63


def derive_seed(evaluation_seed: int, record_id: str, step_index: int,
                n_avg_index: int, purpose: str) -> int:
    """Stable 63-bit seed from the content tuple. SHA-256, not hash()."""
    key = f"{int(evaluation_seed)}|{record_id}|{int(step_index)}|{int(n_avg_index)}|{purpose}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_BITS


def derive_noise(shape, device, evaluation_seed: int, record_id: str, step_index: int,
                 n_avg_index: int, purpose: str = "flowedit", dtype=torch.float32) -> torch.Tensor:
    """One Gaussian tensor from a local generator seeded by derive_seed()."""
    seed = derive_seed(evaluation_seed, record_id, step_index, n_avg_index, purpose)
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(tuple(shape), device=device, dtype=dtype, generator=gen), seed


def build_noise_bank(total_steps: int, n_avg: int, shape, device, evaluation_seed: int,
                     record_id: str, purpose: str = "flowedit"):
    """bank[i][a] for EVERY interval of the grid, plus the seeds used, for the metadata record.

    The bank covers all intervals rather than only the active ones, so the noise at interval i does
    not move when n_start changes.
    """
    bank, seeds = [], []
    for i in range(total_steps):
        row, srow = [], []
        for a in range(n_avg):
            t, s = derive_noise(shape, device, evaluation_seed, record_id, i, a, purpose)
            row.append(t); srow.append(s)
        bank.append(row); seeds.append(srow)
    return bank, seeds


def vae_seed(evaluation_seed: int, record_id: str) -> int:
    """Seed for the VAE posterior draw: its own purpose, disjoint from the FlowEdit stream."""
    return derive_seed(evaluation_seed, record_id, -1, -1, "vae_posterior")
