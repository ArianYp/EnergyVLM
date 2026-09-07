#!/usr/bin/env python3
"""Does the frozen latent projector still rank a STUDENT's outputs like the RGB scorer? (Reviewer's
precondition before using it as a reward inside training.)

For held-out validation captions (the projector's val split, never the 3k pool), a student is
sampled at 4 steps, cfg 1, with N seeds per caption. Each sample's terminal latent is scored by the
projector (no decode) and, after decoding, by the RGB DINO patch scorer against the same reference
photograph. Reported: within-caption top-1 agreement, Spearman, regret in RGB units, and the
RGB score of the projector's pick vs the RGB pick vs a random pick. Then the projector is
fine-tuned on the student latents of HALF the captions (cosine to the decoded DINO embedding) and
re-measured on the other half: the offline version of "update the projector as the student changes".

    python3 phaseW/latent_scorer/student_regret.py --ckpt <student.pt> --n_captions 500 --n_seeds 4
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_projector import LatentProjector, load_shards  # noqa: E402


def rollout_student(model, scheduler, z0, emb, pooled, K, device):
    """cfg-1 Euler rollout (one conditional pass per step), the student's inference sampler."""
    N = z0.shape[0]
    scheduler.set_timesteps(K, device=device)
    sig = scheduler.sigmas.to(device, torch.float32); ts = scheduler.timesteps.to(device)
    z = z0
    for k in range(K):
        with torch.autocast("cuda", torch.bfloat16):
            v = model(hidden_states=z, timestep=ts[k].expand(N), encoder_hidden_states=emb.repeat(N, 1, 1),
                      pooled_projections=pooled.repeat(N, 1), return_dict=False)[0]
        z = (z.float() + (sig[k + 1] - sig[k]) * v.float()).to(torch.bfloat16)
    return z


def agreement(S_hat, S):
    j_l, j_r = S_hat.argmax(1), S.argmax(1)
    n = S.shape[0]
    rho = float(np.mean([stats.spearmanr(S_hat[i].numpy(), S[i].numpy())[0] for i in range(n)]))
    rnd = torch.randint(0, S.shape[1], (n,))
    return {"top1_agree": float((j_l == j_r).float().mean()), "spearman": rho,
            "regret_rgb": float((S.gather(1, j_r[:, None]) - S.gather(1, j_l[:, None])).mean()),
            "rgb_of_proj_pick": float(S.gather(1, j_l[:, None]).mean()), "rgb_of_rgb_pick": float(S.gather(1, j_r[:, None]).mean()),
            "rgb_of_random_pick": float(S.gather(1, rnd[:, None]).mean()), "rgb_mean": float(S.mean()),
            "within_std": float(S.std(1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student checkpoint (or 'teacher8' for the 8-step cfg-7 teacher as a control)")
    ap.add_argument("--projector", default="phaseW/latent_scorer/projector/projector.pt")
    ap.add_argument("--shards", default="phaseW/latent_scorer/shards")
    ap.add_argument("--n_captions", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ft_epochs", type=int, default=8)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.manual_seed(args.seed)
    D = load_shards(f"{args.shards}/shard*.pt")
    split = np.array(D["split"]); va = np.where(split == "val")[0]
    rng = np.random.default_rng(args.seed); va = rng.permutation(va)[:args.n_captions]
    print(f"[regret] {len(va)} validation captions x {args.n_seeds} seeds, student {args.ckpt}", flush=True)

    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad = False
    model = pipe.transformer
    if args.ckpt != "teacher8":
        ck = torch.load(args.ckpt, map_location="cpu", mmap=False, weights_only=False)
        model.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["model"].items()}, strict=True)
        print(f"[regret] loaded student (step {ck.get('step')})", flush=True)
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval(); proc = AutoImageProcessor.from_pretrained(args.dino_id)
    proj = LatentProjector().to(device); proj.load_state_dict(torch.load(args.projector, map_location=device, weights_only=False)["model"]); proj.eval()
    lat_c = model.config.in_channels; H = args.height // pipe.vae_scale_factor
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)

    @torch.no_grad()
    def embed(images):
        px = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(dino(pixel_values=px).last_hidden_state[:, 1:].float().mean(1), dim=-1)

    Z, E, R = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for ci, i in enumerate(va):
            prompt = D["prompt"][i]
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            z0 = torch.randn(args.n_seeds, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(int(D["idx"][i]) * 7 + 1))
            if args.ckpt == "teacher8":
                from gen_latents import rollout
                zK = rollout(model, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, 8, 7.0, device)
            else:
                zK = rollout_student(model, pipe.scheduler, z0, emb, pooled, args.steps, device)
            lat = (zK.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            img = pipe.vae.decode(lat, return_dict=False)[0].clamp(-1, 1).float()
            u8 = ((img + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
            e_c = embed([Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in u8])
            Z.append(zK.cpu()); E.append(e_c.half().cpu()); R.append(D["e_ref"][i])
            if (ci + 1) % 100 == 0:
                print(f"[regret] {ci+1}/{len(va)} ({time.time()-t0:.0f}s)", flush=True)
    Z = torch.stack(Z); E = torch.stack(E); Rf = torch.stack(R).float()        # [n,N,16,64,64], [n,N,768], [n,768]
    S_rgb = (E.float() * Rf[:, None]).sum(-1)

    def proj_scores(P, idx):
        P.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for a in range(0, len(idx), 32):
                b = idx[a:a + 32]; z = Z[b].to(device).float().flatten(0, 1)
                e = P(z).float().view(len(b), args.n_seeds, -1)
                out.append((e * Rf[b].to(device)[:, None]).sum(-1).cpu())
        return torch.cat(out, 0)

    allidx = np.arange(len(va))
    res = {"ckpt": args.ckpt, "n_captions": len(va), "n_seeds": args.n_seeds, "frozen": agreement(proj_scores(proj, allidx), S_rgb)}
    res["frozen"]["emb_cos"] = float(sum((proj(Z[b].to(device).float().flatten(0, 1)).float().cpu() * E[b].float().flatten(0, 1)).sum(-1).mean() for b in [allidx[a:a + 32] for a in range(0, len(allidx), 32)]) / len(range(0, len(allidx), 32)))
    print(f"[regret] FROZEN projector on student latents: {json.dumps({k: round(v, 4) for k, v in res['frozen'].items()})}", flush=True)
    # fine-tune on half the captions' student latents, test on the other half (offline "update as the student changes")
    half = len(va) // 2; tr, te = allidx[:half], allidx[half:]
    res["frozen_on_test_half"] = agreement(proj_scores(proj, te), S_rgb[te])
    P2 = copy.deepcopy(proj).train()
    opt = torch.optim.AdamW(P2.parameters(), lr=args.ft_lr, weight_decay=0.01)
    for ep in range(args.ft_epochs):
        perm = rng.permutation(tr); tot = 0.0
        for a in range(0, len(perm), 16):
            b = perm[a:a + 16]; z = Z[b].to(device).float().flatten(0, 1); e_t = E[b].to(device).float().flatten(0, 1); e_r = Rf[b].to(device)
            with torch.autocast("cuda", torch.bfloat16):
                e = P2(z).float()
            l_emb = (1 - (e * e_t).sum(-1)).mean()
            S_hat = (e.view(len(b), args.n_seeds, -1) * e_r[:, None]).sum(-1); S = S_rgb[b].to(device)
            l_kl = F.kl_div(F.log_softmax(S_hat / 0.04, -1), F.softmax(S / 0.04, -1), reduction="batchmean")
            loss = l_emb + l_kl; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P2.parameters(), 1.0); opt.step(); tot += float(loss) * len(b)
        m = agreement(proj_scores(P2, te), S_rgb[te])
        print(f"[regret] fine-tune ep {ep+1} loss {tot/len(tr):.4f} | test-half top1 {m['top1_agree']:.3f} spearman {m['spearman']:.3f} regret {m['regret_rgb']:.4f}", flush=True)
    res["finetuned_on_test_half"] = agreement(proj_scores(P2, te), S_rgb[te])
    # and does the fine-tuned projector still rank TEACHER candidates? (forgetting check on the val shards)
    tv = np.where(split == "val")[0][:500]
    def teacher_scores(P):
        P.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for a in range(0, len(tv), 32):
                b = tv[a:a + 32]; z = D["z"][b].to(device).float().flatten(0, 1)
                out.append((P(z).float().view(len(b), 4, -1) * D["e_ref"][b].to(device).float()[:, None]).sum(-1).cpu())
        return torch.cat(out, 0)
    res["teacher_candidates_frozen"] = agreement(teacher_scores(proj), D["cos_new"][tv])
    res["teacher_candidates_finetuned"] = agreement(teacher_scores(P2), D["cos_new"][tv])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    torch.save({"model": P2.state_dict(), "finetuned_on": args.ckpt}, str(Path(args.out).with_suffix(".projector_ft.pt")))
    print("[regret] SUMMARY " + json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}), flush=True)
    print("STUDENT_REGRET_OK", flush=True)


if __name__ == "__main__":
    main()
