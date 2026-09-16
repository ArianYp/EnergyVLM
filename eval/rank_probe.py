#!/usr/bin/env python3
"""Held-out ranking probe for the prompt-aware ranking arms (docs/rank/).

On --n COCO captions of the 118k cache that are NOT in the 3k training pool (caption and photo both
unseen), build the same rule-based negatives as training, sample the caption and its negatives with
a student (4 Euler steps, guidance 1, one shared noise per caption seeded by its idx) and score every
generation against the caption's real photograph in four spaces:

    rgb        cos( DINO(decoded image), u_ref )                 the true offline scorer
    latent     cos( P(z_K), u_ref )                              the latent projector (decode-free)
    latent_g   cos( g(P(z_K)), g(u_ref) )                        the projector through the trained head
    rgb_g      cos( g(DINO(decoded)), g(u_ref) )                 the true features through the head

Reports the pairwise accuracy (positive outranks a negative), overall and per edit family, and the
mean margin, per space. `rgb` answers the design's premise (does raw DINO already order the positive
first?); `latent_g` is what the ranking loss optimised; `rgb_g` says whether the head's ordering
carries over to true features. Per-caption records are saved so models can be compared paired.

    python3 eval/rank_probe.py --ckpt checkpoints/.../checkpoint_avg_last5.pt \
        --head checkpoints/.../rank_head_final.pt --label B1_s0 --out out/rank_probe/B1_s0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.build_negatives import negatives_for  # noqa: E402
from train.rank_utils import RankHead, rollout_schedule, student_rollout  # noqa: E402
from train.latent_scorer import LatentProjector  # noqa: E402
from train.distill import PatchScorer  # noqa: E402


def load_cache(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                out[int(r["idx"])] = (r["prompt"], r["reference"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student checkpoint ({'model': state_dict}) or 'base'")
    ap.add_argument("--head", default=None, help="rank_head_*.pt of the run (optional; identity head if absent)")
    ap.add_argument("--proj", default="checkpoints/latent_scorer/projector.pt")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/train")
    ap.add_argument("--exclude", default="cache/train_3k")
    ap.add_argument("--ref", default="cache/reward/ref_emb_118k.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--min_negs", type=int, default=2)
    ap.add_argument("--proj_manifest", default="cache/latents/manifest.jsonl",
                    help="the projector's pretraining captions (train/val/test shards); excluded from the held-out set (review finding)")
    ap.add_argument("--proj_from_head", type=int, default=1, help="1 = use the REFRESHED projector saved in the head file when present (key 'proj')")
    ap.add_argument("--negatives_version", default="v2", help="label only: negatives come from build_negatives.py's current rules")
    ap.add_argument("--seed", type=int, default=0, help="which held-out captions (the noise is seeded by caption idx)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--score_model", default="facebook/dinov2-base")
    args = ap.parse_args()
    device = torch.device("cuda")

    # held-out captions with at least --min_negs negatives, a fixed draw
    all118, pool3k = load_cache(args.cache), load_cache(args.exclude)
    seen_p = {p for p, _ in pool3k.values()}; seen_r = {r for _, r in pool3k.values()}
    # also exclude the projector's own pretraining / validation captions (27k manifest, by text and
    # photo) and the replay set it is refreshed on (no ids stored: matched by reference embedding)
    n_proj = n_replay = 0
    if args.proj_manifest and Path(args.proj_manifest).exists():
        for ln in open(args.proj_manifest):
            if ln.strip():
                r = json.loads(ln); seen_p.add(r["prompt"]); seen_r.add(r["reference"]); n_proj += 1
    ref_emb = torch.load(args.ref, map_location="cpu", weights_only=False)
    replay_path = Path("cache/reward/replay.pt")
    if replay_path.exists():
        rp = torch.load(replay_path, map_location="cpu", weights_only=False, mmap=True)
        ids = sorted(all118); E = torch.stack([ref_emb[i].float() for i in ids])
        E = torch.nn.functional.normalize(E, dim=-1); R = torch.nn.functional.normalize(rp["e_ref"].float(), dim=-1)
        hit = (R @ E.T).max(0).values > 0.9995
        for i, h in zip(ids, hit.tolist()):
            if h:
                seen_r.add(all118[i][1]); n_replay += 1
        del rp
    held = sorted(i for i, (p, r) in all118.items() if p not in seen_p and r not in seen_r)
    print(f"[probe] excluded: 3k pool, {n_proj} projector-manifest captions, {n_replay} replay-set photos -> {len(held)} held-out", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(held)
    picks = []
    for i in held:
        negs = negatives_for(all118[i][0], args.m)
        if len(negs) >= args.min_negs:
            picks.append((i, all118[i][0], negs))
        if len(picks) >= args.n:
            break
    print(f"[probe] {len(picks)} held-out captions with >= {args.min_negs} negatives (of {len(held)} held-out)", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()          # as the trainer does: a CLIP projection otherwise stays fp16
    if args.ckpt != "base":
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
        assert not unexpected and not missing, f"checkpoint mismatch: {len(missing)} missing {len(unexpected)} unexpected"
        print(f"[probe] loaded {args.ckpt} (step {ck.get('step')})", flush=True)
    model = pipe.transformer.eval()
    sig, ts = rollout_schedule(pipe.scheduler, args.steps, args.K, device)
    proj = LatentProjector().to(device).eval()
    proj.load_state_dict(torch.load(args.proj, map_location=device, weights_only=False)["model"])
    head = RankHead(768).to(device).eval()
    proj_source = "pretrained"
    if args.head:
        hk = torch.load(args.head, map_location=device, weights_only=False)
        head.load_state_dict(hk["head"])
        if args.proj_from_head and hk.get("proj") is not None:
            proj.load_state_dict(hk["proj"]); proj_source = "refreshed (from the head file)"
        print(f"[probe] head from {args.head} (step {hk.get('step')}); projector: {proj_source}", flush=True)
    scorer = PatchScorer(args.score_model, device)

    lat_c = model.config.in_channels; H = args.height // pipe.vae_scale_factor
    # SHUFFLED-ANCHOR control (review finding): score every generation against ANOTHER caption's
    # photo. A head that still orders the positive first there has learned "looks like a typical
    # positive rollout", not agreement with the caption's photo.
    spaces = ["rgb", "latent", "latent_g", "rgb_g", "rgb_shuf", "latent_g_shuf"]
    shuf = {picks[i][0]: picks[(i + 1) % len(picks)][0] for i in range(len(picks))}
    recs = []
    t0 = time.time()
    with torch.no_grad():
        for n_done, (idx, prompt, negs) in enumerate(picks):
            prompts = [prompt] + [x["prompt"] for x in negs]
            emb, _, pooled, _ = pipe.encode_prompt(prompt=prompts, prompt_2=prompts, prompt_3=prompts,
                                                   do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            g = torch.Generator(device=device).manual_seed(int(idx))
            z0 = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=g).expand(len(prompts), -1, -1, -1).contiguous()
            zK = student_rollout(model, z0, emb, pooled, sig, ts, grad=False)
            u = ref_emb[int(idx)].to(device).float()[None]
            with torch.autocast("cuda", torch.bfloat16):
                e_lat = proj(zK).float()
            e_rgb = scorer.embed_latents(pipe.vae, zK)                   # decode, 8-bit, offline DINOv2 patch-mean
            u2 = ref_emb[int(shuf[idx])].to(device).float()[None]
            S = {"rgb": (e_rgb * u).sum(-1), "latent": (e_lat * u).sum(-1),
                 "latent_g": (head(e_lat) * head(u)).sum(-1), "rgb_g": (head(e_rgb) * head(u)).sum(-1),
                 "rgb_shuf": (e_rgb * u2).sum(-1), "latent_g_shuf": (head(e_lat) * head(u2)).sum(-1)}
            recs.append({"idx": int(idx), "prompt": prompt, "negatives": negs, "shuffled_anchor_idx": int(shuf[idx]),
                         **{k: [float(x) for x in v] for k, v in S.items()}})
            if n_done % 50 == 0:
                print(f"[probe] {n_done}/{len(picks)} {time.time() - t0:.0f}s", flush=True)

    # pairwise accuracy and margin per space, overall and per family
    summary = {}
    for sp in spaces:
        pairs = defaultdict(list); marg = []
        for r in recs:
            s = r[sp]
            for j, ng in enumerate(r["negatives"], start=1):
                pairs["all"].append(float(s[0] > s[j])); pairs[ng["family"]].append(float(s[0] > s[j]))
            marg.append(s[0] - float(np.mean(s[1:])))
        summary[sp] = {"pairwise_acc": {k: float(np.mean(v)) for k, v in pairs.items()},
                       "n_pairs": {k: len(v) for k, v in pairs.items()}, "margin": float(np.mean(marg)),
                       "positive_score": float(np.mean([r[sp][0] for r in recs]))}
    out = {"label": args.label, "ckpt": args.ckpt, "head": args.head, "projector": proj_source, "n_captions": len(recs), "seed": args.seed,
           "steps": args.steps, "excluded": {"projector_manifest": n_proj, "replay_photos": n_replay}, "negatives_version": args.negatives_version,
           "summary": summary, "records": recs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    fams = sorted({ng["family"] for r in recs for ng in r["negatives"]})
    print(f"\n[probe] {args.label}: pairwise accuracy (positive outranks the negative), {len(recs)} captions, "
          f"{summary['rgb']['n_pairs']['all']} pairs")
    print(f"{'space':10s} {'all':>7s} " + " ".join(f"{f:>10s}" for f in fams) + f" {'margin':>8s} {'pos':>7s}")
    for sp in spaces:
        pa = summary[sp]["pairwise_acc"]
        print(f"{sp:10s} {pa['all']:7.3f} " + " ".join(f"{pa.get(f, float('nan')):10.3f}" for f in fams)
              + f" {summary[sp]['margin']:+8.4f} {summary[sp]['positive_score']:7.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
