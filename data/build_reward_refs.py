#!/usr/bin/env python3
"""Inputs of the reward arms (train/distill.py --reward_mode proj|rgb), from the latent shards of
data/build_latents.py:

  ref_emb.pt   {caption idx -> DINOv2 patch-mean embedding of its reference photograph} for every
               caption of the TEST split (the pool the reward arms train on); the reward is the
               cosine between the student's decoded / projected clean estimate and this vector
  replay.pt    2,000 TRAIN-split captions (200 per shard, a fixed draw): terminal latents of their
               four candidates, the candidates' DINO embeddings, the reference embedding and the RGB
               scores. Used to verify the differentiable RGB path against the offline scorer at
               start-up, and as replay data when the projector is refreshed during training.

    python3 data/build_reward_refs.py --shards cache/latents --out_dir cache/reward
"""
from __future__ import annotations

import argparse
import gc
import glob
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="cache/latents")
    ap.add_argument("--out_dir", default="cache/reward")
    ap.add_argument("--per_shard", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ref = {}; rz, re_, rr, rc = [], [], [], []
    rng = np.random.default_rng(args.seed)
    for f in sorted(glob.glob(str(Path(args.shards) / "shard*.pt"))):
        p = torch.load(f, map_location="cpu", weights_only=False)
        for i, sp in enumerate(p["split"]):
            if sp == "test":
                ref[int(p["idx"][i])] = p["e_ref"][i].clone()
        tr = [i for i, sp in enumerate(p["split"]) if sp == "train"]
        pick = np.sort(rng.choice(tr, size=args.per_shard, replace=False))
        rz.append(p["z"][pick].clone()); re_.append(p["e_cand"][pick].clone()); rr.append(p["e_ref"][pick].clone()); rc.append(p["cos_new"][pick].clone())
        del p; gc.collect()
    torch.save(ref, out / "ref_emb.pt")
    torch.save({"z": torch.cat(rz), "e_cand": torch.cat(re_), "e_ref": torch.cat(rr), "cos": torch.cat(rc)}, out / "replay.pt")
    print(f"reference embeddings for {len(ref)} test captions -> {out / 'ref_emb.pt'}; replay of {sum(x.shape[0] for x in rz)} train captions -> {out / 'replay.pt'}")


if __name__ == "__main__":
    main()
