#!/usr/bin/env python3
"""Task H, stage 2 — render the actual fixed pair for each sampled edit-audit record.

The text-only sheet can establish that an edit changes one atom. It cannot establish the thing the
roadmap actually asks for: whether the frozen pair the model trains on *visibly differs in the
edited concept*. That needs the two images.

Reproduces the trainer's generation exactly — same seeding convention (`seed_base + candidate_idx`),
same teacher, same K and CFG, same canonical (pair_a, pair_b) rollout order — so the rendered images
are the ones the preference loss actually consumed, not a re-sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseC.train_pilot import rollout_states  # noqa: E402
from phaseI.train_preference import load_selection  # noqa: E402
from train_self_distill import sample_teacher_images, setup_distributed, vae_decode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default="phaseFP/fixedpair_101185")
    parser.add_argument("--answer_key", default="phaseFP/edit_audit/answer_key.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    rank, world, _local, device, is_main = setup_distributed(0)

    key = json.loads(Path(args.answer_key).read_text())
    wanted = {int(row["idx"]): row for row in key}
    records = {int(r["idx"]): r for r in load_selection(Path(args.cache_root) / "correct")}
    todo = sorted(i for i in wanted if i in records)
    mine = [i for n, i in enumerate(todo) if n % world == rank]
    if is_main:
        print(f"rendering {len(todo)} pairs across {world} rank(s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline
    from torchvision.utils import save_image

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3,
                   pipe.transformer):
        module.to(dtype=torch.bfloat16).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    teacher = pipe.transformer

    latent_hw = args.height // pipe.vae_scale_factor
    channels = teacher.config.in_channels
    with torch.no_grad():
        neg_embeds, _, neg_pool, _ = pipe.encode_prompt(
            prompt=[""], prompt_2=[""], prompt_3=[""],
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
        )

    written = 0
    for idx in mine:
        record = records[idx]
        target = out / f"p{idx:05d}"
        if (target / "pair_a.png").exists() and (target / "pair_b.png").exists():
            continue
        prompt = record["prompt"]
        seed_base = int(record["seed_base"])
        a, b = int(record["pair_a"]), int(record["pair_b"])
        with torch.no_grad():
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
            )
            starts = []
            for candidate in (a, b):  # canonical order, exactly as the trainer rolls out
                generator = torch.Generator(device=device).manual_seed(seed_base + candidate)
                starts.append(torch.randn(
                    1, channels, latent_hw, latent_hw, device=device,
                    dtype=torch.bfloat16, generator=generator,
                ))
            states, _ = rollout_states(
                teacher, pipe.scheduler, torch.cat(starts, 0), prompt_embeds, pooled,
                neg_embeds, neg_pool, args.K, args.cfg, device,
            )
            endpoints = states[args.K].float().detach()
            images = ((vae_decode(pipe.vae, endpoints) + 1) / 2).clamp(0, 1).cpu()
        target.mkdir(parents=True, exist_ok=True)
        save_image(images[0], target / "pair_a.png")
        save_image(images[1], target / "pair_b.png")
        (target / "meta.json").write_text(json.dumps({
            "idx": idx, "prompt": prompt, "label_prompt": record.get("label_prompt"),
            "pair_a": a, "pair_b": b, "seed_base": seed_base,
            "original_endpoint_vqa": record.get("original_endpoint_vqa"),
            "counterfactual_endpoint_vqa": record.get("counterfactual_endpoint_vqa"),
            "pair_reverses_under_counterfactual": record.get(
                "pair_reverses_under_counterfactual"),
        }, indent=1))
        written += 1
        if written % 10 == 0:
            print(f"[r{rank}] {written}/{len(mine)}", flush=True)
    print(f"[r{rank}] done, {written} pairs rendered", flush=True)


if __name__ == "__main__":
    main()
