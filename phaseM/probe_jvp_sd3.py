#!/usr/bin/env python3
"""Feasibility gate for an sCM/rCM baseline on SD3.5-M: can we JVP through MMDiT?

The continuous-time consistency loss needs the tangent d v_theta/dt, a forward-mode derivative
through the whole denoiser. rCM does not take a monolithic JVP — it carries tangents module by
module and supplies a custom Triton flash-attention JVP kernel, because PyTorch's fused attention
backends have no forward-mode rule. Before committing to eitherpaths we need to know what SD3.5-M's
MMDiT actually supports.

Tests, in increasing order of desirability:
  1. torch.func.jvp with the default SDPA backend            (expected to fail)
  2. torch.func.jvp with the MATH backend forced             (expected to work, slowly)
  3. numerical correctness of the tangent vs a finite difference
  4. time and peak memory at the real training resolution, with and without backward

Prints a machine-readable verdict so the LSF wrapper can gate on it.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import torch


def build(model_id: str, device, dtype):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    net = pipe.transformer.to(device=device, dtype=dtype).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cfg = net.config
    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    return net, cfg


def make_inputs(cfg, device, dtype, height, batch, text_len):
    lat = height // 8
    x = torch.randn(batch, cfg.in_channels, lat, lat, device=device, dtype=dtype)
    t = torch.full((batch,), 500.0, device=device, dtype=dtype)
    enc = torch.randn(batch, text_len, cfg.joint_attention_dim, device=device, dtype=dtype)
    pool = torch.randn(batch, cfg.pooled_projection_dim, device=device, dtype=dtype)
    return x, t, enc, pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--text_len", type=int, default=333)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    verdict = {"config": vars(args), "tests": {}}

    net, cfg = build(args.model_id, device, dtype)
    x, t, enc, pool = make_inputs(cfg, device, dtype, args.height, args.batch, args.text_len)
    seq = (args.height // 16) ** 2 + args.text_len
    verdict["approx_seq_len"] = seq
    print(f"latent {args.height // 8}^2, approx joint seq len {seq}", flush=True)

    def fwd(x_, t_):
        return net(hidden_states=x_, timestep=t_, encoder_hidden_states=enc,
                   pooled_projections=pool, return_dict=False)[0]

    # Tangent directions: d x_t/dt is the teacher velocity (any fixed direction serves the probe).
    vx = torch.randn_like(x)
    vt = torch.ones_like(t)

    # ---- 1. default SDPA backend --------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            out, tan = torch.func.jvp(fwd, (x, t), (vx, vt))
        verdict["tests"]["default_backend"] = {
            "ok": True, "tangent_finite": bool(torch.isfinite(tan).all().item()),
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    except Exception as exc:
        verdict["tests"]["default_backend"] = {
            "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    print("1. default backend:", verdict["tests"]["default_backend"], flush=True)

    # ---- 2. MATH backend forced ---------------------------------------------------------------
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            out, tan = torch.func.jvp(fwd, (x, t), (vx, vt))
        torch.cuda.synchronize()
        verdict["tests"]["math_backend"] = {
            "ok": True,
            "tangent_finite": bool(torch.isfinite(tan).all().item()),
            "tangent_absmean": float(tan.float().abs().mean().item()),
            "seconds": time.time() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    except Exception as exc:
        verdict["tests"]["math_backend"] = {
            "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "trace": traceback.format_exc()[-600:]}
    print("2. math backend:", {k: v for k, v in verdict["tests"]["math_backend"].items()
                               if k != "trace"}, flush=True)

    # ---- 3. numerical check against a finite difference ---------------------------------------
    if verdict["tests"].get("math_backend", {}).get("ok"):
        try:
            eps = 1e-3
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                f0 = fwd(x, t)
                f1 = fwd(x + eps * vx, t + eps * vt)
            fd = (f1 - f0) / eps
            num = (tan - fd).float().norm().item()
            den = fd.float().norm().item() + 1e-12
            verdict["tests"]["finite_difference"] = {
                "rel_error": num / den, "fd_norm": den, "jvp_norm": tan.float().norm().item()}
        except Exception as exc:
            verdict["tests"]["finite_difference"] = {"error": str(exc)[:200]}
        print("3. finite difference:", verdict["tests"]["finite_difference"], flush=True)

    # ---- 4. trainable pass: JVP forward + backward on theta ------------------------------------
    if verdict["tests"].get("math_backend", {}).get("ok"):
        for p in net.parameters():
            p.requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            with sdpa_kernel(SDPBackend.MATH):
                out, tan = torch.func.jvp(fwd, (x, t), (vx, vt))
                # Stand-in for the rCM rectified-flow sCM loss:
                #   g = -(v_sg - v_teacher) - warmup * t * tangent ; loss = ||v - v_sg - g||^2
                g = -(out.detach() - torch.zeros_like(out)) - 1.0 * tan
                g = g / (g.double().norm() + 0.1)
                loss = ((out - out.detach() - g.to(out.dtype)) ** 2).sum()
            loss.backward()
            torch.cuda.synchronize()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
            verdict["tests"]["jvp_backward"] = {
                "ok": True, "seconds": time.time() - t0,
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                "grad_norm_finite": bool(torch.isfinite(torch.tensor(gn)).item()),
                "grad_norm": gn}
        except Exception as exc:
            verdict["tests"]["jvp_backward"] = {
                "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        print("4. jvp+backward:", {k: v for k, v in verdict["tests"]["jvp_backward"].items()}, flush=True)

    feasible = (verdict["tests"].get("math_backend", {}).get("ok")
                and verdict["tests"].get("jvp_backward", {}).get("ok")
                and verdict["tests"].get("finite_difference", {}).get("rel_error", 9) < 0.05)
    verdict["FEASIBLE"] = bool(feasible)
    print(f"\nPROBE_VERDICT: {'FEASIBLE' if feasible else 'NOT_FEASIBLE'}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(verdict, fh, indent=2)


if __name__ == "__main__":
    main()
