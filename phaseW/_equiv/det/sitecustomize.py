"""Deterministic-kernel hook for trainer equivalence tests.

Put this directory first on PYTHONPATH (and export CUBLAS_WORKSPACE_CONFIG=:4096:8) for BOTH
trainers under comparison. Flash / memory-efficient attention backward is nondeterministic, which
shows up as ~1e-6 parameter differences between two runs of the SAME code; with these settings an
original-vs-original control is bit-exact, so any remaining difference is a code difference.
"""
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
