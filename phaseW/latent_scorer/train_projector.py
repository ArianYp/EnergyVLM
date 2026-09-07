#!/usr/bin/env python3
"""Latent scorer: a small projector from the terminal VAE latent of a candidate to the DINOv2
mean-patch embedding space, so candidates can be ranked against the reference photograph without
a VAE decode or a DINO pass on the candidate:

    S_hat_j = cos( P(z_K^{(j)}),  u_pat(x_ref) )          u_pat(x_ref): DINO embedding of the photo

Trained on latents of held-out 118k captions (never the 3k pool) with
    L = mean_j [1 - cos(P(z_j), u_pat(I_j))]  +  lambda * KL( softmax(S/tau) || softmax(S_hat/tau) )
where S is the RGB DINO score of the same candidates; the KL preserves the within-caption ranking
at the temperature the selection arms use. Validation on held-out captions: embedding cosine,
top-1 agreement with the RGB argmax, within-caption Spearman, selection regret in RGB-score units
and the VQAScore headroom recovered against the best-of-4 oracle. The best epoch by regret is
kept, and the 3k pool's candidates are scored into a new cache directory with `latent_cos`.

    python3 phaseW/latent_scorer/train_projector.py --shards phaseW/latent_scorer/shards \
        --out_dir phaseW/latent_scorer/projector --cache_out phaseN/coco_selection_dinopatch_latent
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class LatentProjector(nn.Module):
    """z [B,16,64,64] -> unit vector in R^768 (DINOv2-B patch-mean space). ~6.8M parameters."""

    def __init__(self, c_in=16, d_out=768, width=(64, 128, 256, 512, 512)):
        super().__init__()
        layers, c = [], c_in
        for i, w in enumerate(width):
            layers += [nn.Conv2d(c, w, 3, stride=1 if i == 0 else 2, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
                       nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            c = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c, 1024), nn.SiLU(), nn.Linear(1024, d_out))

    def forward(self, z):
        h = self.body(z).mean(dim=(2, 3))
        return F.normalize(self.head(h), dim=-1)


def load_shards(pattern):
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(pattern))]
    out = {}
    for k in ("z", "e_cand", "e_ref", "cos_new", "cos_cached", "vqa", "idx", "random_idx"):
        out[k] = torch.cat([p[k] for p in parts], 0)
    out["split"] = sum([p["split"] for p in parts], [])
    out["prompt"] = sum([p["prompt"] for p in parts], [])
    return out


def metrics(S_hat, S, vqa, random_idx):
    """S_hat, S: [n,4] latent and RGB scores; vqa [n,4]; random_idx [n]."""
    j_lat, j_rgb = S_hat.argmax(1), S.argmax(1)
    n = S.shape[0]
    top1 = (j_lat == j_rgb).float().mean().item()
    regret = (S.gather(1, j_rgb[:, None]) - S.gather(1, j_lat[:, None])).mean().item()
    rho = float(np.mean([stats.spearmanr(S_hat[i].numpy(), S[i].numpy())[0] for i in range(n)]))
    out = {"top1_agree": top1, "regret_rgb": regret, "spearman": rho}
    if torch.isfinite(vqa).all():
        v_rand = vqa.gather(1, random_idx.clamp_min(0)[:, None]).mean().item() if (random_idx >= 0).all() else vqa.mean().item()
        v_or = vqa.max(1).values.mean().item()
        v_lat = vqa.gather(1, j_lat[:, None]).mean().item(); v_rgb = vqa.gather(1, j_rgb[:, None]).mean().item()
        out.update({"vqa_random": v_rand, "vqa_oracle": v_or, "vqa_latent_pick": v_lat, "vqa_rgb_pick": v_rgb,
                    "headroom_latent": (v_lat - v_rand) / (v_or - v_rand), "headroom_rgb": (v_rgb - v_rand) / (v_or - v_rand),
                    "top1_oracle_latent": (j_lat == vqa.argmax(1)).float().mean().item(),
                    "top1_oracle_rgb": (j_rgb == vqa.argmax(1)).float().mean().item()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="phaseW/latent_scorer/shards")
    ap.add_argument("--out_dir", default="phaseW/latent_scorer/projector")
    ap.add_argument("--cache_in", default="phaseN/coco_selection_dinopatch")
    ap.add_argument("--cache_out", default="phaseN/coco_selection_dinopatch_latent")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32, help="captions per batch (x4 candidates)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default=None, help="log per-epoch losses, validation and test metrics, and the projector artifact")
    ap.add_argument("--wandb_run_name", default="latent-projector")
    args = ap.parse_args()
    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    D = load_shards(f"{args.shards}/shard*.pt")
    split = np.array(D["split"])
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print(f"[proj] train {len(tr)} val {len(va)} test {len(te)} captions; recomputed-vs-cached RGB score max|diff| "
          f"{(D['cos_new'] - D['cos_cached']).abs().max():.4f}", flush=True)
    S_rgb = D["cos_new"]                      # the RGB score of the same latents (recomputed)

    model = LatentProjector().to(device)
    print(f"[proj] parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def scores(idx, bs=64):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                z = D["z"][b].to(device).float().flatten(0, 1)
                e = model(z).float().view(len(b), 4, -1)
                out.append((e * D["e_ref"][b].to(device).float()[:, None]).sum(-1).cpu())
        model.train(); return torch.cat(out, 0)

    def embed_cos(idx, bs=64):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                z = D["z"][b].to(device).float().flatten(0, 1)
                e = model(z).float()
                out.append((e * D["e_cand"][b].to(device).float().flatten(0, 1)).sum(-1).cpu())
        model.train(); return torch.cat(out, 0).mean().item()

    best, log = None, []
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(tr); tot = {"emb": 0.0, "kl": 0.0, "n": 0}
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            z = D["z"][b].to(device).float().flatten(0, 1)                # [B*4,16,64,64]
            e_t = D["e_cand"][b].to(device).float().flatten(0, 1)          # [B*4,768]
            e_r = D["e_ref"][b].to(device).float()                        # [B,768]
            S = S_rgb[b].to(device)                                       # [B,4]
            with torch.autocast("cuda", torch.bfloat16):
                e = model(z).float()
            l_emb = (1 - (e * e_t).sum(-1)).mean()
            S_hat = (e.view(len(b), 4, -1) * e_r[:, None]).sum(-1)
            l_kl = F.kl_div(F.log_softmax(S_hat / args.tau, -1), F.softmax(S / args.tau, -1), reduction="batchmean")
            loss = l_emb + args.lam * l_kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot["emb"] += float(l_emb) * len(b); tot["kl"] += float(l_kl) * len(b); tot["n"] += len(b)
        m = metrics(scores(va), S_rgb[va], D["vqa"][va], D["random_idx"][va]); m["emb_cos_val"] = embed_cos(va)
        m.update({"epoch": ep + 1, "loss_emb": tot["emb"] / tot["n"], "loss_kl": tot["kl"] / tot["n"]})
        log.append(m)
        if wb is not None:
            wb.log({f"proj/{k}": v for k, v in m.items() if k != "epoch"}, step=ep + 1)
        print(f"[proj] ep {ep+1:2d} loss emb {m['loss_emb']:.4f} kl {m['loss_kl']:.4f} | val emb-cos {m['emb_cos_val']:.4f} "
              f"top1 {m['top1_agree']:.3f} spearman {m['spearman']:.3f} regret {m['regret_rgb']:.4f} "
              f"headroom latent {m.get('headroom_latent', float('nan')):.3f} (rgb {m.get('headroom_rgb', float('nan')):.3f}) ({time.time()-t0:.0f}s)", flush=True)
        if best is None or m["regret_rgb"] < best["regret_rgb"]:
            best = dict(m)
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "args": vars(args)}, Path(args.out_dir) / "projector.pt")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"log": log, "best_val": best}, open(Path(args.out_dir) / "train_log.json", "w"), indent=1)
    # reload the best epoch and score the test pool
    model.load_state_dict(torch.load(Path(args.out_dir) / "projector.pt", map_location=device, weights_only=False)["model"])
    S_te = scores(te)
    mt = metrics(S_te, S_rgb[te], D["vqa"][te], D["random_idx"][te]); mt["emb_cos_test"] = embed_cos(te)
    # also against the CACHED RGB score (the one the dino_patch arm used)
    mc = metrics(S_te, D["cos_cached"][te], D["vqa"][te], D["random_idx"][te])
    print(f"[proj] TEST (3k pool, best epoch {best['epoch']}): " + json.dumps({k: round(v, 4) for k, v in mt.items()}), flush=True)
    print(f"[proj] TEST vs cached RGB score: top1 {mc['top1_agree']:.3f} regret {mc['regret_rgb']:.4f} headroom latent {mc.get('headroom_latent', float('nan')):.3f}", flush=True)
    json.dump({"test": mt, "test_vs_cached": mc, "best_val": best}, open(Path(args.out_dir) / "test_metrics.json", "w"), indent=1)
    if wb is not None:
        import wandb
        wb.summary.update({f"test/{k}": v for k, v in mt.items()}); wb.summary.update({f"test_vs_cached/{k}": v for k, v in mc.items()}); wb.summary.update({f"best_val/{k}": v for k, v in best.items()})
        art = wandb.Artifact("latent-projector", type="model", metadata={"best_epoch": best["epoch"], **{f"test_{k}": v for k, v in mt.items()}})
        art.add_file(str(Path(args.out_dir) / "projector.pt")); art.add_file(str(Path(args.out_dir) / "train_log.json")); art.add_file(str(Path(args.out_dir) / "test_metrics.json")); wb.log_artifact(art)
    # write the 3k cache with the latent score
    lat = {int(D["idx"][i]): S_te[k].tolist() for k, i in enumerate(te)}
    Path(args.cache_out).mkdir(parents=True, exist_ok=True)
    n_w = 0
    for f in sorted(Path(args.cache_in).glob("selection_rank*.jsonl")):
        with open(Path(args.cache_out) / f.name, "w") as fh:
            for ln in f.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln); s = lat[int(r["idx"])]
                r["latent_cos"] = s; r["latent_argmax_idx"] = int(np.argmax(s)); n_w += 1
                fh.write(json.dumps(r) + "\n")
    for extra in ("cache_meta.json",):
        if (Path(args.cache_in) / extra).is_file():
            (Path(args.cache_out) / extra).write_text((Path(args.cache_in) / extra).read_text())
    print(f"[proj] wrote {args.cache_out}: {n_w} records with latent_cos; PROJECTOR_OK", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
