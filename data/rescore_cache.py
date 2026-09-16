#!/usr/bin/env python3
"""Prompt-aware SELECTION, stage 3: re-score the candidate cache with the shaped scorer.

Re-rolls the four cached candidates of every caption exactly as the trainer does (seed_base + j,
8-step guidance-7 teacher), decodes, embeds with the offline DINOv2 patch-mean scorer, and scores
each candidate against the caption's photo through the head of train/rank_head.py on both sides:
    shaped_j = < g(DINO(x_j)), g(DINO(photo)) >.
Writes a new cache directory in which `dino_patch_cos` IS the shaped score (so the unchanged
`CD_dinop_hard` selection picks its argmax) and `dino_patch_cos_raw` keeps the recomputed raw one,
and reports the decisive intermediate: how often the shaped argmax differs from the raw argmax, and
how both compare with the VQAScore oracle stored in the cache (`endpoint_vqa`, `oracle_idx`):
oracle agreement and regret (oracle VQAScore minus the selected candidate's).

    python3 data/rescore_cache.py --out cache/train_3k_ranked
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.rank_utils import RankHead, head_apply  # noqa: E402
from common.sampling import rollout, candidate_noise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train_3k")
    ap.add_argument("--head", default="cache/rank_pairs/head_dino.pt")
    ap.add_argument("--ref", default="cache/reward/ref_emb.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--score_model", default="facebook/dinov2-base")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")
    recs = []
    for f in sorted(glob.glob(f"{args.cache}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                recs.append(json.loads(ln))
    recs.sort(key=lambda r: r["idx"])
    if args.limit:
        recs = recs[:args.limit]
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    hk = torch.load(args.head, map_location="cpu", weights_only=False)
    head = RankHead(768, hk.get("width", 1024)).to(device).eval(); head.load_state_dict(hk["head"])

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for mdl in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        mdl.to(dtype=torch.bfloat16).eval()
    teacher = pipe.transformer.eval()
    from train.distill import PatchScorer
    scorer = PatchScorer(args.score_model, device)
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    lat_c = teacher.config.in_channels; H = args.height // pipe.vae_scale_factor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fo = open(out / "selection_rank0.jsonl", "w")
    st = {"n": 0, "same_argmax": 0, "raw_oracle": 0, "shaped_oracle": 0, "vqa_raw": 0.0, "vqa_shaped": 0.0, "vqa_oracle": 0.0,
          "vqa_random": 0.0, "cached_vs_recomputed_maxdiff": 0.0}
    t0 = time.time()
    with torch.no_grad():
        for r in recs:
            N = int(r["N"]); idx = int(r["idx"]); sb = int(r["seed_base"])
            emb, _, pooled, _ = pipe.encode_prompt(prompt=[r["prompt"]], prompt_2=[r["prompt"]], prompt_3=[r["prompt"]], do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            z0 = torch.cat([candidate_noise(sb, j, (1, lat_c, H, H), device) for j in range(N)], 0)
            zs, _ = rollout(teacher, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool, args.K, args.cfg, device, keep_states=True)
            e = scorer.embed_latents(pipe.vae, zs[args.K])       # [N,768]
            u = ref[idx].to(device).float()[None]
            raw = (e * u).sum(-1); shaped = (head_apply(head, e, False) * head_apply(head, u, False)).sum(-1)
            raw_l, sh_l = raw.tolist(), shaped.tolist()
            ja, js = int(np.argmax(raw_l)), int(np.argmax(sh_l))
            vqa = r.get("endpoint_vqa"); jo = int(r.get("oracle_idx", -1))
            st["n"] += 1; st["same_argmax"] += int(ja == js)
            if vqa is not None and jo >= 0:
                st["raw_oracle"] += int(ja == jo); st["shaped_oracle"] += int(js == jo)
                st["vqa_raw"] += vqa[ja]; st["vqa_shaped"] += vqa[js]; st["vqa_oracle"] += vqa[jo]; st["vqa_random"] += vqa[int(r["random_idx"])]
            if "dino_patch_cos" in r:
                st["cached_vs_recomputed_maxdiff"] = max(st["cached_vs_recomputed_maxdiff"], float(np.max(np.abs(np.array(r["dino_patch_cos"]) - np.array(raw_l)))))
                r["dino_patch_cos_cached"] = r["dino_patch_cos"]
            r["dino_patch_cos_raw"] = raw_l; r["dino_patch_cos"] = sh_l; r["dino_patch_argmax_idx"] = js; r["dino_patch_raw_argmax_idx"] = ja
            r["pas_head"] = args.head
            fo.write(json.dumps(r) + "\n")
            if st["n"] % 200 == 0:
                print(f"[rescore] {st['n']}/{len(recs)} {time.time() - t0:.0f}s | argmax changed {1 - st['same_argmax'] / st['n']:.3f} | "
                      f"oracle agreement raw {st['raw_oracle'] / st['n']:.3f} shaped {st['shaped_oracle'] / st['n']:.3f}", flush=True)
    fo.close()
    n = max(st["n"], 1)
    summary = {"captions": st["n"], "argmax_changed_frac": 1 - st["same_argmax"] / n,
               "oracle_agreement": {"raw_dino": st["raw_oracle"] / n, "shaped": st["shaped_oracle"] / n},
               "mean_vqa_of_selected": {"random": st["vqa_random"] / n, "raw_dino": st["vqa_raw"] / n, "shaped": st["vqa_shaped"] / n, "oracle": st["vqa_oracle"] / n},
               "regret": {"raw_dino": (st["vqa_oracle"] - st["vqa_raw"]) / n, "shaped": (st["vqa_oracle"] - st["vqa_shaped"]) / n},
               "cached_vs_recomputed_maxdiff": st["cached_vs_recomputed_maxdiff"], "head": args.head, "source_cache": args.cache}
    json.dump(summary, open(out / "rescore_summary.json", "w"), indent=1)
    print("[rescore] SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
