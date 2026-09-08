#!/usr/bin/env python3
"""
Phase C1 stage 0 — offline candidate-selection cache.

For each TRAINING prompt (T2I-CompBench *_train, disjoint from the val/GenEval
final-eval sets): roll N frozen-teacher candidate trajectories with DETERMINISTIC
per-(prompt,candidate) seeds, decode the endpoint, and VQAScore it. Cache only:

    {idx, category, prompt, seed_base, endpoint_vqa:[N], oracle_idx, random_idx}

The trainer (train_pilot.py) re-rolls the *same* seeded candidates and selects
one by these indices, so B2 (random_idx) and B4 (oracle_idx) train on IDENTICAL
candidate pools differing only in the selection rule — the fairness the pilot
needs. No latent bank on disk (candidates are cheap to regenerate; the teacher
rollout is the same cost the M1 recipe already pays).

seed convention (MUST match train_pilot.py): candidate j of prompt idx uses
    torch.Generator().manual_seed(SEED + idx*1000 + j)     (== Exp-0 convention)

Output: phaseC/selection.jsonl  (sharded, resumable, merged by train_pilot loader)
Uses the SD3.5 bf16 fix + the _t2v_compat shim (see exp0/).
"""
from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, "exp0")
from generate_candidates import build_dev_pool, euler_cfg_sample, decode_and_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="phaseC")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--compbench_dir", default="T2I-CompBench/examples/dataset")
    ap.add_argument("--per_category", type=int, default=700)   # full train split ~5600 prompts
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--tmp_dir", default="phaseC/_tmp_endpoints")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    pool = build_dev_pool(Path(args.compbench_dir), args.per_category, args.seed)
    if rank == 0 and not (out_root / "prompts.json").exists():
        (out_root / "prompts.json").write_text(json.dumps(pool, indent=1))
    out_file = out_root / f"selection_rank{rank}.jsonl"
    done = set()
    if out_file.exists():
        for ln in out_file.read_text().splitlines():
            try: done.add(json.loads(ln)["idx"])
            except Exception: pass

    print(f"[r{rank}] loading SD3.5 (bf16, frozen)...", flush=True)
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dtype=torch.bfloat16); m.eval()
        for p in m.parameters(): p.requires_grad = False
    with torch.no_grad():
        neg_emb, _, neg_pool, _ = pipe.encode_prompt(prompt=[""], prompt_2=[""], prompt_3=[""],
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
    lat_c = pipe.transformer.config.in_channels
    H_lat = args.height // pipe.vae_scale_factor

    sys.path.insert(0, "t2v_metrics"); sys.path.insert(0, str(Path("exp0").resolve()))
    import _t2v_compat  # noqa: F401
    import t2v_metrics
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    print(f"[r{rank}] loading VQAScore ...", flush=True)
    vqa = t2v_metrics.VQAScore(model=args.vqa_model, device=str(device), cache_dir=hub)

    tmp = Path(args.tmp_dir) / f"r{rank}"; tmp.mkdir(parents=True, exist_ok=True)
    mine = [it for it in pool if it["idx"] % world == rank]
    print(f"[r{rank}] {len(mine)} prompts; {len(done)} cached", flush=True)

    with open(out_file, "a") as fh:
        for n, item in enumerate(mine):
            pidx, prompt, cat = item["idx"], item["prompt"], item["category"]
            if pidx in done: continue
            with torch.no_grad():
                emb, _, pooled, _ = pipe.encode_prompt(prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                    do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
                z0 = torch.empty(args.N, lat_c, H_lat, H_lat, dtype=torch.bfloat16, device=device)
                for j in range(args.N):
                    g = torch.Generator(device=device).manual_seed(args.seed + pidx * 1000 + j)
                    z0[j] = torch.randn(lat_c, H_lat, H_lat, device=device, dtype=torch.bfloat16, generator=g)
                zK = euler_cfg_sample(pipe.transformer, pipe.scheduler, z0, emb, pooled,
                                      neg_emb, neg_pool, args.K, args.cfg, device)   # [N,C,H,W]
                decode_and_save(pipe.vae, zK, tmp)                                   # writes cand0..N-1.png
                paths = [str(tmp / f"cand{j}.png") for j in range(args.N)]
                scores = vqa(images=paths, texts=[prompt]).squeeze(1).float().cpu().tolist()
            oracle = int(max(range(args.N), key=lambda j: scores[j]))
            rnd = random.Random(args.seed + pidx).randrange(args.N)
            fh.write(json.dumps({"idx": pidx, "category": cat, "prompt": prompt,
                                 "seed_base": args.seed + pidx * 1000, "N": args.N,
                                 "endpoint_vqa": scores, "oracle_idx": oracle,
                                 "random_idx": rnd}) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"[r{rank}] {n + 1}/{len(mine)} | oracle-vs-mean gain "
                      f"{scores[oracle] - sum(scores)/len(scores):+.3f}", flush=True)
    print(f"[r{rank}] done -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
