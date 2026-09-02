#!/usr/bin/env python3
"""Candidate cache: N teacher trajectories per caption, each scored against the caption's photo.

For every caption in the pool manifest, N candidates are sampled from the frozen teacher
(K Euler steps with classifier-free guidance) from deterministic seeds, decoded, and scored:

    dino_patch_cos   cosine between mean-pooled DINOv2 patch tokens of the candidate and of the
                     reference photograph (the selector used by the scored arm)
    dino_cos         the same with the DINOv2 CLS token
    clip_cos         CLIP image-embedding cosine to the reference photograph
    endpoint_vqa     VQAScore(image, caption); optional, needs the vendored t2v_metrics

plus `random_idx`, a fixed uniform draw, which is what the naive arm trains on. Only the scores and
indices are stored; the trainer re-rolls the selected candidate from its seed, so the images need
not be kept.

Seed convention (the trainer depends on it): candidate j of caption idx starts from
    manual_seed(seed + idx * 1000 + j)

Sharded over independent processes with RANK / WORLD_SIZE (no collectives), resumable: a rank
skips captions already present in its own output file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.sampling import candidate_noise, decode_and_save, encode_prompt, rollout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_manifest", required=True, help="data/build_pool.py manifest")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vqa", action="store_true", help="also compute VQAScore (t2v_metrics)")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--t2v_dir", default=str(ROOT / "third_party" / "t2v_metrics"))
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"selection_rank{rank}.jsonl"
    done: set[int] = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["idx"]))

    pool = json.loads(Path(args.pool_manifest).read_text())
    mine = [row for i, row in enumerate(pool) if i % world == rank]
    print(f"[r{rank}] {len(mine)} captions assigned, {len(done)} already cached", flush=True)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad_(False)

    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection
    dino = AutoModel.from_pretrained(args.dino_id).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(args.dino_id)
    clip = CLIPVisionModelWithProjection.from_pretrained(args.clip_id).to(device).eval()
    clip_proc = CLIPImageProcessor.from_pretrained(args.clip_id)

    vqa = None
    if args.vqa:
        sys.path.insert(0, args.t2v_dir)
        import t2v_metrics
        vqa = t2v_metrics.VQAScore(model=args.vqa_model, device=f"cuda:{local_rank}")

    @torch.no_grad()
    def reference_scores(ref_img, cand_imgs) -> dict:
        out = {}
        px = dino_proc(images=[ref_img] + cand_imgs, return_tensors="pt")["pixel_values"].to(device)
        h = dino(pixel_values=px).last_hidden_state
        cls = F.normalize(h[:, 0].float(), dim=-1)
        pat = F.normalize(h[:, 1:].float().mean(1), dim=-1)          # patch tokens only, CLS dropped
        out["dino_cos"] = (cls[0:1] * cls[1:]).sum(-1).cpu().tolist()
        out["dino_patch_cos"] = (pat[0:1] * pat[1:]).sum(-1).cpu().tolist()
        px = clip_proc(images=[ref_img] + cand_imgs, return_tensors="pt")["pixel_values"].to(device)
        e = F.normalize(clip(pixel_values=px).image_embeds.float(), dim=-1)
        out["clip_cos"] = (e[0:1] * e[1:]).sum(-1).cpu().tolist()
        return out

    lat_c = pipe.transformer.config.in_channels
    h_lat = args.height // pipe.vae_scale_factor
    tmp = out_root / f"_tmp_rank{rank}"
    from PIL import Image

    with torch.no_grad():
        neg_emb, neg_pool = encode_prompt(pipe, "", device)

    written = 0
    with open(out_file, "a") as handle:
        for row in mine:
            pidx = int(row["idx"])
            if pidx in done or not os.path.exists(row["reference"]):
                continue
            prompt, reference = row["prompt"], row["reference"]
            seed_base = args.seed + pidx * 1000
            with torch.no_grad():
                emb, pooled = encode_prompt(pipe, prompt, device)
                z0 = torch.cat([candidate_noise(seed_base, j, (1, lat_c, h_lat, h_lat), device)
                                for j in range(args.N)], 0)
                z_end = rollout(pipe.transformer, pipe.scheduler, z0, emb, pooled, neg_emb, neg_pool,
                                args.K, args.cfg, device)
                decode_and_save(pipe.vae, z_end, tmp)
                paths = [str(tmp / f"cand{j}.png") for j in range(args.N)]
                ref_img = Image.open(reference).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
                cands = [Image.open(p).convert("RGB") for p in paths]
                scores = reference_scores(ref_img, cands)
                if vqa is not None:
                    scores["endpoint_vqa"] = vqa(images=paths, texts=[prompt]).squeeze(1).float().cpu().tolist()

            record = {"idx": pidx, "prompt": prompt, "reference": reference,
                      "seed_base": seed_base, "N": args.N,
                      "random_idx": random.Random(args.seed + pidx).randrange(args.N)}
            for key, vals in scores.items():
                record[key] = vals
                record[f"{key}_argmax_idx"] = int(max(range(args.N), key=lambda j: vals[j]))
            if "endpoint_vqa" in record:
                record["oracle_idx"] = record["endpoint_vqa_argmax_idx"]
            handle.write(json.dumps(record) + "\n")
            written += 1
            if written % 20 == 0:
                handle.flush()
                print(f"[r{rank}] {written}/{len(mine)} cached", flush=True)
    print(f"[r{rank}] done: {written} newly cached", flush=True)


if __name__ == "__main__":
    main()
