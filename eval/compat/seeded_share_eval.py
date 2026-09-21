#!/usr/bin/env python3
"""Run the vendored, UNMODIFIED `Share_eval.py` with the RNG seeded (eval/sharecot_nonspatial.py).

Share_eval.py samples at temperature 0.2 / top_p 0.7 and seeds nothing, so two runs on the same
images give different scores. We do not want to edit the official file (the whole point is that the
numbers are theirs), so we exec it verbatim after seeding the global RNGs. Everything else -- argv,
cwd, the module's own argparse -- is exactly what a direct `python Share_eval.py ...` invocation
would see.

Env:
  SHARECOT_SEED    int, default 0
  SHARECOT_SCRIPT  path to Share_eval.py (default: third_party/T2I-CompBench/MLLM_eval/ShareGPT4V-CoT_eval/Share_eval.py)
"""
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

SEED = int(os.environ.get("SHARECOT_SEED", "0"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

script = os.environ.get(
    "SHARECOT_SCRIPT",
    str(Path(__file__).resolve().parents[2] / "third_party" / "T2I-CompBench" / "MLLM_eval"
        / "ShareGPT4V-CoT_eval" / "Share_eval.py"),
)
sys.argv = [script] + sys.argv[1:]
print(f"[seeded] seed={SEED} exec {script}", flush=True)
runpy.run_path(script, run_name="__main__")
