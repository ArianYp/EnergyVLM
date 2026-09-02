"""torchrun-aware process setup. Single-process when WORLD_SIZE is unset."""
from __future__ import annotations

import os
from datetime import timedelta

import torch


def setup_distributed(default_gpu: int = 0, nccl_timeout_min: int = 30):
    """Returns (rank, world_size, local_rank, device, is_main)."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        n_visible = torch.cuda.device_count()
        assert n_visible >= 1, f"rank {rank}: no visible GPU"
        # Either every rank sees all GPUs (pin to LOCAL_RANK) or each rank was given exactly one
        # via CUDA_VISIBLE_DEVICES (pin to cuda:0).
        if n_visible == 1:
            cuda_idx = 0
        elif local_rank < n_visible:
            cuda_idx = local_rank
        else:
            raise RuntimeError(f"rank {rank}: LOCAL_RANK={local_rank} but only {n_visible} GPUs visible")
        torch.cuda.set_device(cuda_idx)
        device = torch.device(f"cuda:{cuda_idx}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=nccl_timeout_min))
        return rank, world_size, local_rank, device, rank == 0
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(f"cuda:{default_gpu}")
    torch.cuda.set_device(device)
    return 0, 1, default_gpu, device, True


def barrier() -> None:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])


def teardown() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
