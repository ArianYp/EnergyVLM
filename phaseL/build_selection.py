#!/usr/bin/env python3
"""Build a larger, immutable preference cache from independent teacher seed pools.

Repeat zero is copied exactly from the audited Phase-C cache. Additional repeats
reconstruct four fresh frozen-SD3.5 teacher candidates per training prompt and
score their decoded endpoints with VQAScore. Images are discarded after scoring,
except for a small per-category audit set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp0"))

from exp0.generate_candidates import decode_and_save, euler_cfg_sample
from phaseI.train_preference import load_selection


RECORD_STRIDE = 1_000_000
SEED_STRIDE = 10_000_000


def expanded_index(source_idx: int, repeat: int) -> int:
    if source_idx < 0 or source_idx >= RECORD_STRIDE:
        raise ValueError(f"source index {source_idx} is outside the stable record stride")
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    return repeat * RECORD_STRIDE + source_idx


def expanded_seed(seed_base: int, repeat: int) -> int:
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    return int(seed_base) + repeat * SEED_STRIDE


def select_stratified(records: list[dict], limit_per_category: int) -> list[dict]:
    if limit_per_category <= 0:
        return records
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for record in records:
        category = str(record.get("category", "unknown"))
        if counts[category] >= limit_per_category:
            continue
        selected.append(record)
        counts[category] += 1
    return selected


def copied_record(source: dict) -> dict:
    source_idx = int(source["idx"])
    record = dict(source)
    record.update({
        "idx": expanded_index(source_idx, 0),
        "source_idx": source_idx,
        "seed_repeat": 0,
        "record_uid": f"p{source_idx:07d}-r000",
        "label_prompt": source["prompt"],
        "label_source": "correct_prompt",
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--source_selection_dir", default="phaseC")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--additional_repeats", type=int, default=3)
    parser.add_argument("--limit_per_category", type=int, default=0)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--vqa_model", default="clip-flant5-xxl")
    args = parser.parse_args()

    if args.additional_repeats < 1:
        raise ValueError("additional_repeats must be at least one")
    if args.N < 2 or args.K < 1 or args.height != 512:
        raise ValueError("require N >= 2, K >= 1, and height == 512")

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise FileNotFoundError(f"launcher must create immutable output directory: {out_root}")
    out_file = out_root / f"selection_rank{rank}.jsonl"
    if out_file.exists():
        raise FileExistsError(f"refusing to overwrite {out_file}")

    source_records = select_stratified(
        load_selection(args.source_selection_dir), args.limit_per_category
    )
    if not source_records:
        raise RuntimeError("stratified source selection is empty")
    source_candidate_counts = {len(record["endpoint_vqa"]) for record in source_records}
    if source_candidate_counts != {args.N}:
        raise ValueError(
            f"source candidate count {sorted(source_candidate_counts)} does not match N={args.N}"
        )

    first_by_category: dict[str, int] = {}
    for record in source_records:
        first_by_category.setdefault(str(record.get("category", "unknown")), int(record["idx"]))

    if rank == 0:
        prompts_path = out_root / "prompts.json"
        prompts_path.write_text(json.dumps([
            {
                "source_idx": int(record["idx"]),
                "category": record.get("category", "unknown"),
                "prompt": record["prompt"],
            }
            for record in source_records
        ], indent=2))

    print(
        f"[rank {rank}] loading frozen teacher for {len(source_records)} prompts, "
        f"{args.additional_repeats} new repeats",
        flush=True,
    )
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    for module in (
        pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    ):
        module.to(dtype=torch.bfloat16).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    with torch.no_grad():
        negative_embeds, _, negative_pool, _ = pipe.encode_prompt(
            prompt=[""], prompt_2=[""], prompt_3=[""],
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
        )
    latent_channels = pipe.transformer.config.in_channels
    latent_hw = args.height // pipe.vae_scale_factor

    sys.path.insert(0, str(ROOT / "t2v_metrics"))
    import _t2v_compat  # noqa: F401
    import t2v_metrics

    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    vqa = t2v_metrics.VQAScore(
        model=args.vqa_model, device=str(device), cache_dir=hub
    )
    tmp = out_root / "_tmp_endpoints" / f"rank{rank}"
    tmp.mkdir(parents=True, exist_ok=False)

    written = 0
    try:
        with out_file.open("x") as handle:
            for source in source_records:
                record = copied_record(source)
                if int(record["idx"]) % world == rank:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1

            for repeat in range(1, args.additional_repeats + 1):
                for source in source_records:
                    source_idx = int(source["idx"])
                    record_idx = expanded_index(source_idx, repeat)
                    if record_idx % world != rank:
                        continue
                    prompt = str(source["prompt"])
                    category = str(source.get("category", "unknown"))
                    seed_base = expanded_seed(int(source["seed_base"]), repeat)
                    with torch.no_grad():
                        embeds, _, pooled, _ = pipe.encode_prompt(
                            prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                            do_classifier_free_guidance=False, device=device,
                            num_images_per_prompt=1,
                        )
                        starts = torch.empty(
                            args.N, latent_channels, latent_hw, latent_hw,
                            dtype=torch.bfloat16, device=device,
                        )
                        for candidate_idx in range(args.N):
                            generator = torch.Generator(device=device).manual_seed(
                                seed_base + candidate_idx
                            )
                            starts[candidate_idx] = torch.randn(
                                latent_channels, latent_hw, latent_hw, device=device,
                                dtype=torch.bfloat16, generator=generator,
                            )
                        endpoints = euler_cfg_sample(
                            pipe.transformer, pipe.scheduler, starts, embeds, pooled,
                            negative_embeds, negative_pool, args.K, args.cfg, device,
                        )
                        decode_and_save(pipe.vae, endpoints, tmp)
                        image_paths = [str(tmp / f"cand{j}.png") for j in range(args.N)]
                        scores = (
                            vqa(images=image_paths, texts=[prompt])
                            .squeeze(1).float().cpu().tolist()
                        )

                    oracle_idx = int(max(range(args.N), key=scores.__getitem__))
                    random_idx = random.Random(seed_base + source_idx).randrange(args.N)
                    record = {
                        "idx": record_idx,
                        "source_idx": source_idx,
                        "seed_repeat": repeat,
                        "record_uid": f"p{source_idx:07d}-r{repeat:03d}",
                        "category": category,
                        "prompt": prompt,
                        "label_prompt": prompt,
                        "label_source": "correct_prompt",
                        "seed_base": seed_base,
                        "N": args.N,
                        "K": args.K,
                        "cfg": args.cfg,
                        "endpoint_vqa": scores,
                        "oracle_idx": oracle_idx,
                        "random_idx": random_idx,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    written += 1

                    if first_by_category[category] == source_idx:
                        sample_dir = out_root / "samples" / f"repeat_{repeat:03d}" / category
                        sample_dir.mkdir(parents=True, exist_ok=False)
                        for candidate_idx, image_path in enumerate(image_paths):
                            shutil.copy2(image_path, sample_dir / f"cand{candidate_idx}.png")
                        (sample_dir / "metadata.json").write_text(json.dumps({
                            "record_uid": record["record_uid"],
                            "category": category,
                            "prompt": prompt,
                            "scores": scores,
                            "oracle_idx": oracle_idx,
                            "seed_base": seed_base,
                        }, indent=2))

                    if written % 50 == 0:
                        print(
                            f"[rank {rank}] wrote {written} records; repeat={repeat}; "
                            f"margin={max(scores) - min(scores):+.4f}",
                            flush=True,
                        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[rank {rank}] complete: {written} records -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
