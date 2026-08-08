#!/usr/bin/env python3
"""JVP probe v2 — decide whether the tangent is CORRECT, not merely computable.

Probe v1 showed the math SDPA backend gives a finite tangent at full resolution (0.41 s, 10.2 GB)
but flagged a 28% mismatch against a finite difference. Three defects in that check, all mine:

  * forward difference (O(eps) truncation) on a strongly nonlinear 2.5B network, single eps;
  * the timestep tangent was 1.0 while SD3 timesteps live on a 0-1000 scale, so the t-direction
    contributed ~1e-6 of the perturbation and was effectively untested;
  * bf16 run died on a dtype mismatch before reaching the check.

v2 fixes all three: central differences over an eps sweep (error must FALL as eps shrinks if the
tangent is right), x-only and t-only directions tested separately, and the timestep tangent scaled
so we differentiate with respect to the rectified-flow sigma the loss actually uses.

It also measures the structure rCM really uses — JVP under no_grad as a constant target, plus one
ordinary forward/backward — which is far cheaper than backpropagating through the JVP.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def build(model_id, device, dtype):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    net = pipe.transformer.to(device=device, dtype=dtype).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cfg = net.config
    del pipe
    return net, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--text_len", type=int, default=333)
    ap.add_argument("--num_train_ts", type=float, default=1000.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float32           # JVP numerics are done in fp32; bf16 is not worth the risk
    net, cfg = build(args.model_id, device, dtype)
    lat = args.height // 8
    torch.manual_seed(0)
    x = torch.randn(1, cfg.in_channels, lat, lat, device=device, dtype=dtype)
    enc = torch.randn(1, args.text_len, cfg.joint_attention_dim, device=device, dtype=dtype)
    pool = torch.randn(1, cfg.pooled_projection_dim, device=device, dtype=dtype)
    v_teacher = torch.randn_like(x)          # stands in for the PF-ODE direction dx/dsigma

    # Parameterise by rectified-flow sigma in [0,1]; the network consumes sigma * num_train_ts.
    def fwd_sigma(x_, s_):
        return net(hidden_states=x_, timestep=s_ * args.num_train_ts, encoder_hidden_states=enc,
                   pooled_projections=pool, return_dict=False)[0]

    sigma = torch.full((1,), args.sigma, device=device, dtype=dtype)
    report = {"config": vars(args), "seq_len": (args.height // 16) ** 2 + args.text_len}

    def jvp_of(dx, ds):
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            return torch.func.jvp(fwd_sigma, (x, sigma), (dx, ds))

    def central(dx, ds, eps):
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            fp = fwd_sigma(x + eps * dx, sigma + eps * ds)
            fm = fwd_sigma(x - eps * dx, sigma - eps * ds)
        return (fp - fm) / (2 * eps)

    # Three directions: x only, sigma only, and the combination the sCM tangent actually needs.
    zero_x, zero_s = torch.zeros_like(x), torch.zeros_like(sigma)
    dirs = {
        "x_only": (v_teacher, zero_s),
        "sigma_only": (zero_x, torch.ones_like(sigma)),
        "combined": (v_teacher, torch.ones_like(sigma)),
    }
    report["directions"] = {}
    for name, (dx, ds) in dirs.items():
        t0 = time.time()
        _, tan = jvp_of(dx, ds)
        dt_jvp = time.time() - t0
        entry = {"jvp_norm": tan.float().norm().item(), "jvp_seconds": dt_jvp, "eps_sweep": {}}
        for eps in (1e-1, 1e-2, 1e-3):
            fd = central(dx, ds, eps)
            rel = ((tan - fd).float().norm() / (fd.float().norm() + 1e-12)).item()
            entry["eps_sweep"][f"{eps:g}"] = {"rel_error": rel, "fd_norm": fd.float().norm().item()}
        errs = [entry["eps_sweep"][k]["rel_error"] for k in ("0.1", "0.01", "0.001")]
        # A correct tangent: central-difference error falls as eps shrinks, then may rise again from
        # float cancellation. Converging is the signal; a flat curve means the tangent is wrong.
        entry["converges"] = bool(errs[1] < errs[0])
        entry["best_rel_error"] = min(errs)
        report["directions"][name] = entry
        print(f"{name}: jvp_norm={entry['jvp_norm']:.4f} errs={['%.4f'%e for e in errs]} "
              f"converges={entry['converges']}", flush=True)

    # Real training-step structure: JVP as a no_grad constant + one ordinary forward/backward.
    for p in net.parameters():
        p.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        _, tangent = torch.func.jvp(fwd_sigma, (x, sigma), (v_teacher, torch.ones_like(sigma)))
    v_theta = fwd_sigma(x, sigma)                      # ordinary forward, fused kernels allowed
    v_sg = v_theta.detach()
    g = -(v_sg - v_teacher) - 1.0 * args.sigma * tangent
    g = (g.double() / (g.double().norm() + 0.1)).to(v_theta.dtype)
    loss = ((v_theta - v_sg - g) ** 2).sum()
    loss.backward()
    torch.cuda.synchronize()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
    report["rcm_step"] = {
        "seconds": time.time() - t0,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.item()),
        "grad_norm": gn,
        "finite": bool(torch.isfinite(torch.tensor(gn)).item() and torch.isfinite(loss).item()),
    }
    print("rcm_step:", report["rcm_step"], flush=True)

    ok = (report["directions"]["combined"]["converges"]
          and report["directions"]["combined"]["best_rel_error"] < 0.05
          and report["rcm_step"]["finite"])
    report["FEASIBLE"] = bool(ok)
    print(f"\nPROBE2_VERDICT: {'FEASIBLE' if ok else 'NOT_FEASIBLE'}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
