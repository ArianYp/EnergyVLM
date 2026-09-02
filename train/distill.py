#!/usr/bin/env python3
"""Consistency distillation of SD3.5 on selected teacher trajectories.

Each update takes one caption from the candidate cache, picks ONE of its N cached candidates with
the chosen selector, re-rolls that candidate's K-step guided teacher trajectory from its seed, and
trains the student to predict the teacher's clean-latent estimate from a noisier state:

    v_k    = (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k)        teacher velocity on segment k
    x0_k   = z_k - sigma_k * v_k                              teacher's clean-latent estimate
    x0_hat = z_{k-d} - sigma_{k-d} * v_theta(z_{k-d}, c)      student, d steps noisier, no guidance
    loss   = mean_k  pseudo_huber(x0_hat - sg[x0_k])

over the supervised window of k. The student runs a single conditional forward, so guidance is
absorbed and it is sampled with cfg 1.

Selectors (the only difference between the two arms; everything else is identical):
    random       the cached uniform draw `random_idx`
    dino_patch   argmax of `dino_patch_cos`: DINOv2 mean-patch cosine to the caption's photograph

Multi-GPU under torchrun: a DistributedSampler shows every caption once per global epoch, so
num_steps = epochs * n_captions / world_size.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.distributed import barrier, setup_distributed, teardown  # noqa: E402
from common.sampling import candidate_noise, encode_prompt, rollout  # noqa: E402

SELECTORS = {"random": None, "dino_patch": "dino_patch_cos"}


class Records(Dataset):
    def __init__(self, records):
        self.items = records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def select(rec: dict, selector: str) -> int:
    field = SELECTORS[selector]
    if field is None:
        return int(rec["random_idx"])
    s = rec[field]
    return int(max(range(len(s)), key=lambda j: s[j]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True, choices=sorted(SELECTORS))
    ap.add_argument("--cache_dir", required=True, help="data/build_candidates.py output")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--num_steps", type=int, default=6000)
    ap.add_argument("--num_warmup_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=8, help="teacher steps")
    ap.add_argument("--cfg", type=float, default=7.0, help="teacher guidance")
    ap.add_argument("--window", default="0.4,0.9", help="supervised fraction of the trajectory")
    ap.add_argument("--delta_min", type=int, default=1)
    ap.add_argument("--delta_max", type=int, default=3)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default=None, help="log to Weights & Biases if set")
    ap.add_argument("--wandb_run_name", default=None)
    args = ap.parse_args()

    rank, world, local_rank, device, is_main = setup_distributed(0)
    torch.manual_seed(args.seed + rank * 1009)
    random.seed(args.seed + rank * 1009)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))
    barrier()

    recs = []
    for f in sorted(Path(args.cache_dir).glob("selection_rank*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                recs.append(json.loads(ln))
    recs.sort(key=lambda r: r["idx"])
    if not recs:
        raise SystemExit(f"no records under {args.cache_dir}")
    if is_main:
        print(f"[r{rank}] selector={args.selector} | {len(recs)} captions", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    student = pipe.transformer
    student.to(dtype=torch.float32).train()
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    teacher = copy.deepcopy(student).to(dtype=torch.bfloat16).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_module = student_ddp = student
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        student_ddp = DDP(student, device_ids=[device.index], find_unused_parameters=False)
        student_module = student_ddp.module

    opt = torch.optim.AdamW([p for p in student_module.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, args.num_warmup_steps)))

    lat_c = student_module.config.in_channels
    h_lat = args.height // pipe.vae_scale_factor
    huber_c = 0.00054 * (lat_c * h_lat * h_lat) ** 0.5
    K = args.K
    lo, hi = (float(x) for x in args.window.split(","))
    score_idxs = list(range(max(1, round(lo * K)), min(K - 1, round(hi * K)) + 1))

    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)

    sampler = DistributedSampler(Records(recs), num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(Records(recs), batch_size=1, sampler=sampler, shuffle=(sampler is None),
                        drop_last=True, collate_fn=lambda b: b[0])
    data_iter = iter(loader)
    epoch = 0

    if is_main and args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config={**vars(args), "n_captions": len(recs), "world_size": world})
    if is_main:
        h = hashlib.sha256()
        for r in recs:
            h.update(f"{int(r['idx'])}:{select(r, args.selector)}\n".encode())
        print(f"[selection] window={args.window} K={K} supervised_k={score_idxs} "
              f"caption->candidate sha256={h.hexdigest()[:16]}", flush=True)

    pbar = tqdm(total=args.num_steps, disable=not is_main, desc=args.selector)
    gstep = 0
    while gstep < args.num_steps:
        try:
            rec = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            rec = next(data_iter)
        sel = select(rec, args.selector)

        # re-roll the selected teacher trajectory from its seed
        with torch.no_grad():
            emb, pooled = encode_prompt(pipe, rec["prompt"], device)
            z0 = candidate_noise(rec["seed_base"], sel, (1, lat_c, h_lat, h_lat), device)
            states, sigmas = rollout(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool,
                                     K, args.cfg, device, keep_states=True)
            z = [s.float() for s in states]

        opt.zero_grad(set_to_none=True)
        deltas = {k: random.randint(args.delta_min, args.delta_max) for k in score_idxs}
        stu_idx = [k - deltas[k] for k in score_idxs]
        n_w = len(score_idxs)
        z_in = torch.cat([z[s] for s in stu_idx], 0).to(torch.bfloat16)
        t_in = torch.cat([pipe.scheduler.timesteps.to(device)[s].reshape(1) for s in stu_idx], 0)
        with torch.autocast("cuda", torch.bfloat16):
            v_stu = student_ddp(hidden_states=z_in, timestep=t_in,
                                encoder_hidden_states=emb.repeat(n_w, 1, 1),
                                pooled_projections=pooled.repeat(n_w, 1), return_dict=False)[0]
        sig_stu = torch.stack([sigmas[s] for s in stu_idx]).view(-1, 1, 1, 1)
        x_hat = torch.cat([z[s] for s in stu_idx], 0) - sig_stu * v_stu.float()
        x_tea = []
        for k in score_idxs:
            v_k = (z[k + 1] - z[k]) / (sigmas[k + 1] - sigmas[k])
            x_tea.append(z[k] - sigmas[k] * v_k)
        x_tea = torch.cat(x_tea, 0).detach()
        sq = (x_hat - x_tea).pow(2).sum(dim=(1, 2, 3))
        loss = (torch.sqrt(sq + huber_c * huber_c) - huber_c).mean()

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student_module.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        gstep += 1
        pbar.update(1)

        if is_main and gstep % args.log_every == 0:
            lv, g = float(loss.detach().item()), float(gn)
            pbar.set_postfix({"loss": f"{lv:.3g}", "g": f"{g:.2g}"})
            if args.wandb_project:
                import wandb
                wandb.log({"train/loss": lv, "train/grad_norm": g, "train/lr": sched.get_last_lr()[0]}, step=gstep)
        if gstep % args.save_every == 0 and gstep != args.num_steps:
            if is_main:
                torch.save({"model": student_module.state_dict(), "step": gstep, "selector": args.selector},
                           Path(args.output_dir) / f"checkpoint_step{gstep}.pt")
            barrier()

    if is_main:
        torch.save({"model": student_module.state_dict(), "step": gstep, "selector": args.selector},
                   Path(args.output_dir) / "checkpoint_final.pt")
        if args.wandb_project:
            import wandb
            wandb.finish()
    teardown()


if __name__ == "__main__":
    main()
