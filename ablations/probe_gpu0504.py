#!/usr/bin/env python3
"""Health probe for a single LSF GPU node (written for farm-gpu0504).

Exercises exactly what our 4-GPU training jobs need:
  1. all four GPUs are visible and are the expected model,
  2. ~60 GB of ballast per GPU is really allocatable (plus a transient push to
     ~78 GB, which is the per-GPU footprint of the preference/REPA trainers),
  3. NCCL init + a correct all-reduce on every iteration,
  4. a sustained bf16 matmul burn so thermal/clock/ECC problems have time to show.

Prints a per-rank TFLOP/s + peak-memory line and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import torch
import torch.distributed as dist

GIB = 1024 ** 3


def log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def allocate_ballast(target_gib: float, device: torch.device, chunk_gib: float = 4.0):
    """Allocate ~target_gib of device memory in chunks; returns the chunk list."""
    chunks = []
    allocated = 0.0
    while allocated < target_gib:
        this = min(chunk_gib, target_gib - allocated)
        nbytes = int(this * GIB)
        chunks.append(torch.empty(nbytes, dtype=torch.uint8, device=device))
        allocated += this
    # Touch the memory so it is genuinely backed, not just reserved.
    for c in chunks:
        c.fill_(1)
    torch.cuda.synchronize(device)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ballast_gib", type=float, default=60.0,
                    help="persistent per-GPU ballast held during the burn")
    ap.add_argument("--headroom_gib", type=float, default=78.0,
                    help="transient per-GPU peak to prove the training footprint fits")
    ap.add_argument("--burn_seconds", type=float, default=60.0)
    ap.add_argument("--matmul_n", type=int, default=16384)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        print("PROBE_RANK_FAIL: torch.cuda.is_available() is False", flush=True)
        return 2
    if torch.cuda.device_count() < world:
        print(f"PROBE_RANK_FAIL: only {torch.cuda.device_count()} CUDA devices visible, "
              f"need {world}", flush=True)
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        timeout=dt.timedelta(minutes=10),
    )

    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / GIB
    log(rank, f"device={props.name} sm={props.major}.{props.minor} "
              f"total_mem={total_gib:.1f} GiB pci_bus={getattr(props, 'pci_bus_id', 'n/a')}")

    failures: list[str] = []

    # ------------------------------------------------------------------ memory
    torch.cuda.reset_peak_memory_stats(device)
    try:
        head = allocate_ballast(args.headroom_gib, device)
        peak_headroom = torch.cuda.max_memory_allocated(device) / GIB
        log(rank, f"headroom allocation OK: {args.headroom_gib:.0f} GiB "
                  f"(peak_allocated={peak_headroom:.1f} GiB)")
        del head
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        failures.append(f"headroom_alloc_{args.headroom_gib:.0f}GiB: {exc}")
        log(rank, f"headroom allocation FAILED: {exc}")
        peak_headroom = float("nan")

    ballast = None
    try:
        ballast = allocate_ballast(args.ballast_gib, device)
        log(rank, f"ballast allocation OK: {args.ballast_gib:.0f} GiB held during burn")
    except RuntimeError as exc:
        failures.append(f"ballast_alloc_{args.ballast_gib:.0f}GiB: {exc}")
        log(rank, f"ballast allocation FAILED: {exc}")

    # ------------------------------------------------------- NCCL correctness
    expected = world * (world - 1) / 2.0
    probe = torch.full((1024, 1024), float(rank), dtype=torch.float32, device=device)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    err = (probe - expected).abs().max().item()
    if err != 0.0:
        failures.append(f"allreduce_init_mismatch max_abs_err={err} expected={expected}")
    log(rank, f"initial all_reduce: expected={expected} got={probe[0, 0].item()} "
              f"max_abs_err={err}")

    # ------------------------------------------------------------------- burn
    n = args.matmul_n
    a = torch.randn(n, n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=device)
    c = torch.empty(n, n, dtype=torch.bfloat16, device=device)
    red = torch.full((4 * 1024 * 1024,), float(rank), dtype=torch.float32, device=device)

    # warmup
    for _ in range(3):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(device)

    iters = 0
    bad_reductions = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.burn_seconds:
        for _ in range(4):
            torch.matmul(a, b, out=c)
        red.fill_(float(rank))
        dist.all_reduce(red, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        if (red - expected).abs().max().item() != 0.0:
            bad_reductions += 1
        iters += 4
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    if bad_reductions:
        failures.append(f"allreduce_mismatch_in_{bad_reductions}_iterations")

    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12
    peak_gib = torch.cuda.max_memory_allocated(device) / GIB

    log(rank, f"burn done: matmuls={iters} n={n} elapsed={elapsed:.1f}s "
              f"achieved={tflops:.1f} TFLOP/s (bf16) peak_alloc={peak_gib:.1f} GiB "
              f"bad_allreduce={bad_reductions}")

    del ballast, a, b, c
    torch.cuda.empty_cache()

    # ----------------------------------------------------------- gather + verdict
    payload = json.dumps({
        "rank": rank,
        "device": props.name,
        "total_mem_gib": round(total_gib, 1),
        "peak_headroom_gib": None if peak_headroom != peak_headroom else round(peak_headroom, 1),
        "peak_alloc_gib": round(peak_gib, 1),
        "tflops": round(tflops, 1),
        "matmuls": iters,
        "bad_allreduce": bad_reductions,
        "failures": failures,
    })
    gathered: list[str | None] = [None] * world
    dist.all_gather_object(gathered, payload)

    rc = 0
    if rank == 0:
        rows = [json.loads(g) for g in gathered]
        print("\n=== PER-RANK RESULTS ===", flush=True)
        for r in rows:
            print(f"  rank {r['rank']}: {r['device']} | {r['tflops']:.1f} TFLOP/s bf16 | "
                  f"peak_alloc {r['peak_alloc_gib']} GiB | "
                  f"headroom_peak {r['peak_headroom_gib']} GiB | "
                  f"bad_allreduce {r['bad_allreduce']} | failures {r['failures']}",
                  flush=True)
        all_failures = [f"rank{r['rank']}:{f}" for r in rows for f in r["failures"]]
        tfs = [r["tflops"] for r in rows]
        print(f"\nAGGREGATE: min={min(tfs):.1f} max={max(tfs):.1f} "
              f"mean={sum(tfs)/len(tfs):.1f} TFLOP/s bf16 across {world} GPUs", flush=True)
        # A healthy H200 does >600 TFLOP/s bf16 dense; flag anything grossly below.
        slow = [r["rank"] for r in rows if r["tflops"] < 300]
        if slow:
            all_failures.append(f"slow_gpus_below_300_TFLOPs:{slow}")
        if all_failures:
            print("PROBE_RESULT: FAIL " + "; ".join(all_failures), flush=True)
            rc = 1
        else:
            print("PROBE_RESULT: PASS", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
