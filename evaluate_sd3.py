#!/usr/bin/env python3
"""
Evaluate Sana teacher/base/distilled models on held-out prompts.

This script covers two questions:
1. Model quality: compare teacher vs base student vs distilled student using
   CLIP text-image similarity on held-out prompts.
2. Energy hypothesis (optional): if a DiT->DINOv2 projector checkpoint is
   available, test whether low-energy teacher samples tend to score higher
   than random teacher samples for the same prompt.

Baseline-only mode:
  If --distilled_ckpt / --checkpoint_dir are not provided (or --skip_distilled
  is passed), the script just evaluates the teacher and the base student.
  This is useful for getting reference CLIP scores before distillation runs.

Outputs:
  - summary.json
  - prompts.json
  - model_samples.jsonl
  - teacher_energy_samples.jsonl (if a projector checkpoint is available)
  - images/ (optional)

Example (GenEval2, all 800 prompts):
  python evaluate_sd3.py \
    --checkpoint_dir checkpoints/distill_sana_mse_5ep \
    --geneval2_path GenEval2/geneval2_data.jsonl \
    --output_dir logs/eval_sana

Baselines only (no distilled checkpoint needed):
  python evaluate_sd3.py \
    --geneval2_path GenEval2/geneval2_data.jsonl \
    --skip_distilled \
    --output_dir logs/eval_sana_baselines
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from diffusers import SanaPipeline
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


FALLBACK_PROMPTS = [
    "A portrait photo of a tabby cat wearing a red scarf",
    "A mountain lake at sunrise with pine trees reflecting in the water",
    "A wooden dining table with a bowl of oranges and a glass vase",
    "A rainy city street at night with neon signs and umbrellas",
    "A cozy reading corner with a lamp, a blanket, and a stack of books",
    "A macro photo of a butterfly resting on a purple flower",
    "A futuristic train station with polished metal floors",
    "A bowl of ramen with sliced egg, noodles, and steam rising",
    "A golden retriever running across a grassy park",
    "A modern office desk with a laptop, notebook, and coffee mug",
]

COMPOSITIONAL_PROMPTS = [
    "A red cube on top of a blue sphere, with a green cylinder under the sphere",
    "A yellow mug to the left of a laptop, while a spoon is inside the mug",
    "A cat sitting under a wooden chair, with a ball on top of the chair",
    "A bicycle leaning against the right side of a wall, with a backpack hanging on the left side",
    "A book placed between two candles, with a key behind the left candle",
    "A small dog in front of a large mirror, and a lamp behind the dog",
    "A plate on a table, with a fork to the left of the plate and a knife to the right",
    "A vase in the center, with three apples arranged in a triangle around it",
    "A clock above a window, and a plant below the window",
    "A chair facing a desk, with the chair's back toward the viewer",
]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return float(cov / denom)


def read_prompt_file(prompt_file: str) -> list[str]:
    path = Path(prompt_file)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict) and "prompts" in payload:
            return [str(item).strip() for item in payload["prompts"] if str(item).strip()]
        raise ValueError(f"Unsupported JSON prompt format in {prompt_file}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_geneval2_prompts(geneval2_path: str) -> tuple[list[str], str]:
    """Load prompts from GenEval2's `geneval2_data.jsonl` (or the repo dir).

    Each line is `{"prompt": str, "atom_count": int, "vqa_list": [...], "skills": [...]}`.
    We only use the "prompt" field here (Soft-TIFA VQA scoring is out of scope
    for this CLIP-score eval; see GenEval2/evaluation.py for that).
    """
    path = Path(geneval2_path)
    if path.is_dir():
        candidate = path / "geneval2_data.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"No geneval2_data.jsonl found under {geneval2_path}")
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"GenEval2 prompts file not found: {geneval2_path}")

    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = str(record.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
    return prompts, "geneval2"


def load_coco_eval_captions(coco_path: str) -> tuple[list[str], str]:
    path = Path(coco_path)
    if path.is_dir():
        candidates = [
            ("coco_val2017", path / "annotations" / "captions_val2017.json"),
            ("coco_val2014", path / "annotations" / "captions_val2014.json"),
            ("coco_val2017_flat", path / "captions_val2017.json"),
            ("coco_train2017", path / "annotations" / "captions_train2017.json"),
            ("coco_train2014", path / "annotations" / "captions_train2014.json"),
        ]
        for source_name, candidate in candidates:
            if candidate.exists():
                with open(candidate) as f:
                    ann = json.load(f)
                captions = [a["caption"] for a in ann["annotations"]]
                return captions, source_name
        raise FileNotFoundError(f"No COCO captions JSON found under {coco_path}")

    with open(path) as f:
        ann = json.load(f)
    return [a["caption"] for a in ann["annotations"]], path.name


def sample_prompts(pool: list[str], num_prompts: int, seed: int) -> list[str]:
    if not pool:
        return []
    rng = random.Random(seed)
    if num_prompts >= len(pool):
        prompts = pool[:]
        rng.shuffle(prompts)
        return prompts
    return rng.sample(pool, num_prompts)


def dedupe_keep_order(prompts: list[str]) -> list[str]:
    seen = set()
    result = []
    for prompt in prompts:
        if prompt not in seen:
            result.append(prompt)
            seen.add(prompt)
    return result


def resolve_checkpoint(checkpoint_dir: str | None, prefix: str, step: int | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    ckpt_dir = Path(checkpoint_dir)
    if step is not None:
        path = ckpt_dir / f"{prefix}_step{step}.pt"
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    final_path = ckpt_dir / f"{prefix}_final.pt"
    if final_path.exists():
        return final_path

    step_paths = sorted(ckpt_dir.glob(f"{prefix}_step*.pt"))
    if not step_paths:
        return None
    step_paths = sorted(
        step_paths,
        key=lambda p: int(p.stem.split("step")[-1]),
    )
    return step_paths[-1]


def resolve_training_checkpoint(checkpoint_dir: str | None, step: int | None) -> Path | None:
    """Locate a `checkpoint_step*.pt` / `checkpoint_final.pt` saved by train_distill_sd3.py.

    That training script saves the student under a 'model' key (and optionally 'ema')
    inside a single dict, rather than a bare `student_stepN.pt`.  We try those names
    first, then fall back to the legacy `student_*.pt` layout.
    """
    if checkpoint_dir is None:
        return None
    ckpt_dir = Path(checkpoint_dir)
    if step is not None:
        for name in (f"checkpoint_step{step}.pt", f"student_step{step}.pt"):
            path = ckpt_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No checkpoint for step {step} under {ckpt_dir}")

    for name in ("checkpoint_final.pt", "student_final.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path

    candidates = sorted(ckpt_dir.glob("checkpoint_step*.pt")) + sorted(ckpt_dir.glob("student_step*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: int(p.stem.split("step")[-1]))
    return candidates[-1]


def build_eval_prompts(args: argparse.Namespace) -> tuple[list[str], str]:
    """Resolve the prompt pool. Priority: --prompt_file > --geneval2_path > --coco_path > fallback.

    --num_prompts <= 0 means "use all prompts from the source" (useful for GenEval2's 800).
    """
    use_all = args.num_prompts is None or args.num_prompts <= 0

    prompt_source = "fallback_prompts"
    if args.prompt_file:
        prompt_pool = read_prompt_file(args.prompt_file)
        n = len(prompt_pool) if use_all else min(args.num_prompts, len(prompt_pool))
        prompts = sample_prompts(prompt_pool, n, args.seed)
        prompt_source = Path(args.prompt_file).name
    elif args.geneval2_path and Path(args.geneval2_path).exists():
        prompt_pool, prompt_source = load_geneval2_prompts(args.geneval2_path)
        n = len(prompt_pool) if use_all else min(args.num_prompts, len(prompt_pool))
        prompts = sample_prompts(prompt_pool, n, args.seed)
    elif args.coco_path and Path(args.coco_path).exists():
        prompt_pool, prompt_source = load_coco_eval_captions(args.coco_path)
        n = len(prompt_pool) if use_all else args.num_prompts
        prompts = sample_prompts(prompt_pool, n, args.seed)
    else:
        n = len(FALLBACK_PROMPTS) if use_all else min(args.num_prompts, len(FALLBACK_PROMPTS))
        prompts = sample_prompts(FALLBACK_PROMPTS, n, args.seed)

    if args.include_compositional_prompts:
        prompts = prompts + COMPOSITIONAL_PROMPTS

    prompts = dedupe_keep_order(prompts)
    return prompts, prompt_source


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def summarize_records(records: list[dict]) -> dict:
    clip_scores = [r["clipscore"] for r in records]
    clip_cosines = [r["clip_cosine"] for r in records]
    return {
        "num_samples": len(records),
        "mean_clipscore": safe_mean(clip_scores),
        "std_clipscore": safe_std(clip_scores),
        "mean_clip_cosine": safe_mean(clip_cosines),
        "std_clip_cosine": safe_std(clip_cosines),
    }


def prompt_mean_map(records: list[dict]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["prompt_idx"]].append(record["clipscore"])
    return {idx: float(mean(values)) for idx, values in grouped.items()}


def compare_prompt_means(left: dict[int, float], right: dict[int, float]) -> dict:
    common = sorted(set(left) & set(right))
    if not common:
        return {
            "num_prompts": 0,
            "mean_delta_clipscore": None,
            "left_better_prompt_fraction": None,
        }
    diffs = [left[idx] - right[idx] for idx in common]
    wins = sum(1 for diff in diffs if diff > 0)
    return {
        "num_prompts": len(common),
        "mean_delta_clipscore": float(mean(diffs)),
        "left_better_prompt_fraction": float(wins / len(common)),
    }


def _make_hook(capture_dict: dict, is_final_step: list[bool]):
    def hook(module, input, output):
        if is_final_step[0]:
            # Sana transformer blocks return a single Tensor [B, seq, D].
            hid = output[0] if isinstance(output, tuple) else output
            capture_dict["hidden"] = hid.detach()

    return hook


def generate_and_capture_hidden(
    pipe: SanaPipeline,
    prompt: str,
    num_images: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    seed: int,
    max_sequence_length: int,
) -> tuple[list[Image.Image], torch.Tensor]:
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae

    prompt_embeds, prompt_attention_mask, neg_prompt_embeds, neg_prompt_attention_mask = pipe.encode_prompt(
        prompt=[prompt] * num_images,
        negative_prompt=[""] * num_images,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=None,
    )
    prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([neg_prompt_attention_mask, prompt_attention_mask], dim=0)

    latent_channels = transformer.config.in_channels
    latents = pipe.prepare_latents(
        num_images,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator=torch.Generator(device=device.type).manual_seed(seed),
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    capture_dict = {}
    is_final = [False]

    last_block = transformer.transformer_blocks[-1]
    handle = last_block.register_forward_hook(_make_hook(capture_dict, is_final))

    try:
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            is_final[0] = i == len(timesteps) - 1
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        with torch.no_grad():
            latents_for_decode = (latents / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)
            images = vae.decode(latents_for_decode, return_dict=False)[0]
        pil_images = pipe.image_processor.postprocess(images.float(), output_type="pil")
    finally:
        handle.remove()

    if "hidden" not in capture_dict:
        raise RuntimeError("Failed to capture Sana hidden states at the final denoising step")

    # Conditional half of the CFG-doubled batch; mean-pool tokens.
    h_t = capture_dict["hidden"][num_images:].mean(dim=1)
    return pil_images, h_t


def compute_energies(
    h_t: torch.Tensor,
    pil_images: list[Image.Image],
    projector: torch.nn.Module,
    dinov2_model: torch.nn.Module,
    dinov2_processor: AutoImageProcessor,
    device: torch.device,
) -> list[float]:
    dinov2_inputs = dinov2_processor(images=pil_images, return_tensors="pt")
    pixel_values = dinov2_inputs["pixel_values"].to(device)

    with torch.no_grad():
        dinov2_out = dinov2_model(pixel_values=pixel_values)
        z_dino = F.normalize(dinov2_out.last_hidden_state[:, 0, :], p=2, dim=-1)

    z_dit = projector(h_t.float())
    return (1.0 - (z_dit * z_dino).sum(dim=-1)).cpu().tolist()


def compute_q_values(energies: list[float], tau: float) -> list[float]:
    energy_tensor = torch.tensor(energies, dtype=torch.float32)
    q = F.softmax(-energy_tensor / tau, dim=0)
    return q.tolist()


def compute_clip_scores(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    prompts: list[str],
    images: list[Image.Image],
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    clip_cosines = []
    clip_scores = []

    for start in range(0, len(images), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_images = images[start : start + batch_size]
        inputs = clip_processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            cosine = (image_features * text_features).sum(dim=-1).cpu().tolist()

        for value in cosine:
            clip_cosines.append(float(value))
            clip_scores.append(float(100.0 * max(value, 0.0)))

    return clip_cosines, clip_scores


def save_images(
    images: list[Image.Image],
    image_dir: Path,
    model_name: str,
    prompt_idx: int,
    max_image_prompts: int,
    suffixes: list[str] | None = None,
) -> list[str]:
    saved_paths = []
    if prompt_idx >= max_image_prompts:
        return saved_paths

    for image_idx, image in enumerate(images):
        suffix = ""
        if suffixes and image_idx < len(suffixes) and suffixes[image_idx]:
            suffix = f"_{suffixes[image_idx]}"
        path = image_dir / f"{model_name}_prompt{prompt_idx:04d}_img{image_idx:02d}{suffix}.png"
        image.save(path)
        saved_paths.append(str(path))
    return saved_paths


def load_teacher_pipeline(model_id: str, device: torch.device) -> SanaPipeline:
    # Sana's official recipe: transformer in fp16 (bf16 produces garbage for the
    # 1.6B 1024px variant, which has no BF16 release), but the Gemma-2 text
    # encoder and the DC-AE VAE must be in bf16 -- Gemma-2's attn_logit_softcapping
    # silently overflows in fp16 and can yield NaN prompt embeddings.
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.bfloat16)
    pipe.transformer.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _extract_student_state_dict(checkpoint_path: Path, use_ema: bool) -> dict:
    """Load a student state dict from either a bare state_dict file or a
    full training checkpoint dict with {"model", "ema", ...} keys."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        key = "ema" if use_ema and "ema" in obj else "model"
        print(f"  Loaded training checkpoint (step={obj.get('steps', '?')}), using key='{key}'")
        return obj[key]
    return obj


def load_student_pipeline(
    student_id: str,
    teacher_pipe: SanaPipeline,
    device: torch.device,
    checkpoint_path: Path | None = None,
    use_ema: bool = False,
) -> SanaPipeline:
    # Keep dtype consistent with the teacher (fp16) so the shared text encoder
    # and VAE run at one precision and no silent casts happen between stages.
    pipe = SanaPipeline.from_pretrained(student_id, torch_dtype=torch.float16)
    if checkpoint_path is not None:
        state_dict = _extract_student_state_dict(checkpoint_path, use_ema)
        pipe.transformer.load_state_dict(state_dict)
    pipe = pipe.to(device)

    # Match the training setup: only the student transformer differs.  Sana has
    # a single Gemma text encoder and tokenizer (unlike SD3's three-encoder stack).
    pipe.vae = teacher_pipe.vae
    pipe.text_encoder = teacher_pipe.text_encoder
    pipe.tokenizer = teacher_pipe.tokenizer
    pipe.scheduler = teacher_pipe.scheduler
    pipe.transformer.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def evaluate_model(
    model_name: str,
    pipe: SanaPipeline,
    prompts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_dir: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    records = []
    running_scores = []
    progress = tqdm(enumerate(prompts), total=len(prompts), desc=f"Eval {model_name}")

    for prompt_idx, prompt in progress:
        generator = torch.Generator(device=device.type).manual_seed(args.seed + prompt_idx * 1000)
        output = pipe(
            prompt=prompt,
            num_images_per_prompt=args.num_images_per_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        images = output.images
        prompt_batch = [prompt] * len(images)
        clip_cosines, clip_scores = compute_clip_scores(
            clip_model,
            clip_processor,
            prompt_batch,
            images,
            device,
            args.clip_batch_size,
        )

        saved_paths = []
        if image_dir is not None:
            saved_paths = save_images(images, image_dir, model_name, prompt_idx, args.max_image_prompts)

        for image_idx, (clip_cosine, clipscore) in enumerate(zip(clip_cosines, clip_scores)):
            running_scores.append(clipscore)
            records.append(
                {
                    "model": model_name,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "image_idx": image_idx,
                    "seed": args.seed + prompt_idx * 1000,
                    "clip_cosine": clip_cosine,
                    "clipscore": clipscore,
                    "image_path": saved_paths[image_idx] if image_idx < len(saved_paths) else None,
                }
            )

        progress.set_postfix(mean_clip=f"{safe_mean(running_scores):.2f}")

    return summarize_records(records), records


def evaluate_teacher_energy(
    teacher_pipe: SanaPipeline,
    projector_ckpt: Path,
    prompts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_dir: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    # Lazy import: only needed for the energy-hypothesis path.
    from projector import DiT2DINOProjector

    dinov2_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)
    dinov2_model = AutoModel.from_pretrained(args.dinov2_id).to(device).eval()

    dit_dim = (
        teacher_pipe.transformer.config.num_attention_heads
        * teacher_pipe.transformer.config.attention_head_dim
    )
    projector = DiT2DINOProjector(
        dit_dim=dit_dim,
        dinov2_dim=dinov2_model.config.hidden_size,
    ).to(device)
    projector.load_state_dict(torch.load(projector_ckpt, map_location="cpu"))
    projector.eval()

    records = []
    all_energies = []
    all_clip_cosines = []
    all_clip_scores = []
    top1_by_energy = []
    random_expected = []
    oracle_best = []
    better_than_prompt_mean = 0

    progress = tqdm(enumerate(prompts), total=len(prompts), desc="Eval teacher energy")
    for prompt_idx, prompt in progress:
        images, h_t = generate_and_capture_hidden(
            teacher_pipe,
            prompt,
            args.teacher_candidates,
            args.height,
            args.width,
            args.num_inference_steps,
            args.guidance_scale,
            device,
            args.seed + prompt_idx * 1000 + 1,
            args.max_sequence_length,
        )
        energies = compute_energies(
            h_t,
            images,
            projector,
            dinov2_model,
            dinov2_processor,
            device,
        )
        q_values = compute_q_values(energies, args.tau)
        prompt_batch = [prompt] * len(images)
        clip_cosines, clip_scores = compute_clip_scores(
            clip_model,
            clip_processor,
            prompt_batch,
            images,
            device,
            args.clip_batch_size,
        )

        suffixes = [f"E{energy:.4f}_q{q_value:.4f}" for energy, q_value in zip(energies, q_values)] if image_dir is not None else None
        saved_paths = []
        if image_dir is not None:
            saved_paths = save_images(
                images,
                image_dir,
                "teacher_energy",
                prompt_idx,
                args.max_image_prompts,
                suffixes=suffixes,
            )

        best_idx = min(range(len(energies)), key=lambda i: energies[i])
        top1_by_energy.append(clip_scores[best_idx])
        prompt_mean = float(mean(clip_scores))
        random_expected.append(prompt_mean)
        oracle_best.append(max(clip_scores))
        if clip_scores[best_idx] > prompt_mean:
            better_than_prompt_mean += 1

        for image_idx, (energy, q_value, clip_cosine, clipscore) in enumerate(zip(energies, q_values, clip_cosines, clip_scores)):
            all_energies.append(energy)
            all_clip_cosines.append(clip_cosine)
            all_clip_scores.append(clipscore)
            records.append(
                {
                    "model": "teacher_energy",
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "image_idx": image_idx,
                    "seed": args.seed + prompt_idx * 1000 + 1,
                    "energy": energy,
                    "q_value": q_value,
                    "clip_cosine": clip_cosine,
                    "clipscore": clipscore,
                    "selected_by_energy": image_idx == best_idx,
                    "image_path": saved_paths[image_idx] if image_idx < len(saved_paths) else None,
                }
            )

        progress.set_postfix(
            rho=f"{spearmanr(all_energies, all_clip_cosines) or 0.0:.3f}",
            top1=f"{safe_mean(top1_by_energy):.2f}",
        )

    summary = {
        "num_prompts": len(prompts),
        "teacher_candidates": args.teacher_candidates,
        "mean_energy": safe_mean(all_energies),
        "std_energy": safe_std(all_energies),
        "mean_clipscore": safe_mean(all_clip_scores),
        "mean_clip_cosine": safe_mean(all_clip_cosines),
        "energy_vs_clip_cosine_spearman": spearmanr(all_energies, all_clip_cosines),
        "energy_vs_clipscore_spearman": spearmanr(all_energies, all_clip_scores),
        "top1_by_energy_clipscore_mean": safe_mean(top1_by_energy),
        "random_teacher_clipscore_mean": safe_mean(random_expected),
        "oracle_teacher_clipscore_mean": safe_mean(oracle_best),
        "top1_by_energy_minus_random_clipscore": (
            safe_mean(top1_by_energy) - safe_mean(random_expected)
            if top1_by_energy and random_expected
            else None
        ),
        "top1_by_energy_better_than_prompt_mean_fraction": (
            float(better_than_prompt_mean / len(prompts)) if prompts else None
        ),
    }

    del projector
    del dinov2_model
    torch.cuda.empty_cache()
    return summary, records


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sana teacher/base/distilled models")
    parser.add_argument("--output_dir", default="logs/eval_sana")
    parser.add_argument("--teacher_id",
                        default="Efficient-Large-Model/Sana_1600M_1024px_diffusers")
    parser.add_argument("--student_id",
                        default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory containing distilled student and/or DiT->DINO projector checkpoints.")
    parser.add_argument("--distilled_ckpt", default=None,
                        help="Path to a specific distilled student checkpoint "
                             "(overrides auto-resolution from --checkpoint_dir).")
    parser.add_argument("--projector_ckpt", default=None,
                        help="Path to DiT->DINOv2 projector (for energy hypothesis; optional).")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to evaluate")
    parser.add_argument("--skip_distilled", action="store_true",
                        help="Always skip distilled-student evaluation (baselines only). "
                             "Useful for getting reference CLIP scores before distillation starts.")
    parser.add_argument("--geneval2_path",
                        default="GenEval2/geneval2_data.jsonl",
                        help="Path to GenEval2's geneval2_data.jsonl (or the repo dir "
                             "containing it). Default prompt source for this eval.")
    parser.add_argument("--coco_path", default=None,
                        help="COCO root or captions JSON (used only if --geneval2_path "
                             "does not resolve).")
    parser.add_argument("--prompt_file", default=None,
                        help="Explicit newline-delimited or JSON prompt file (highest priority).")
    parser.add_argument("--num_prompts", type=int, default=0,
                        help="Number of prompts to sample. 0 or negative means 'use all' "
                             "(e.g. all 800 GenEval2 prompts).")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--teacher_candidates", type=int, default=8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="CFG scale. Sana recommends ~4.5 (vs SD3's ~7.0).")
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma tokenizer max seq length (Sana default: 300).")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature used to convert energies into q values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_batch_size", type=int, default=8)
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--use_ema", action="store_true",
                        help="Load EMA weights from the training checkpoint (if present).")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--max_image_prompts", type=int, default=20)
    parser.add_argument(
        "--include_compositional_prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a fixed set of 10 hand-crafted compositional prompts. "
             "Off by default since GenEval2 already contains 800 compositional prompts.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images" if args.save_images else None
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    # Resolve distilled checkpoint (explicit path > auto-resolve from checkpoint_dir).
    # Skip entirely if --skip_distilled is set.
    if args.skip_distilled:
        distilled_ckpt = None
    elif args.distilled_ckpt:
        distilled_ckpt = Path(args.distilled_ckpt)
    else:
        distilled_ckpt = resolve_training_checkpoint(args.checkpoint_dir, args.step)

    # Projector checkpoint is entirely independent; only used for the energy path.
    projector_ckpt = Path(args.projector_ckpt) if args.projector_ckpt else resolve_checkpoint(args.checkpoint_dir, "projector", args.step)

    prompts, prompt_source = build_eval_prompts(args)
    if not prompts:
        raise ValueError("No evaluation prompts found")

    with open(output_dir / "prompts.json", "w") as f:
        json.dump({"prompt_source": prompt_source, "prompts": prompts}, f, indent=2)

    print(f"Using {len(prompts)} evaluation prompts from {prompt_source}")
    if distilled_ckpt is not None:
        print(f"Resolved distilled checkpoint: {distilled_ckpt}")
    elif args.skip_distilled:
        print("--skip_distilled set: evaluating baselines only (teacher + base student)")
    else:
        print("No distilled checkpoint provided; evaluating baselines only (teacher + base student)")
    if projector_ckpt is not None:
        print(f"Resolved projector checkpoint: {projector_ckpt}")
    else:
        print("No projector checkpoint provided; skipping teacher energy analysis")

    print("Loading CLIP scorer...")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_id)
    clip_model = CLIPModel.from_pretrained(args.clip_id).to(device).eval()

    summary = {
        "config": vars(args),
        "device": str(device),
        "prompt_source": prompt_source,
        "num_prompts_total": len(prompts),
        "resolved_checkpoints": {
            "distilled_ckpt": str(distilled_ckpt) if distilled_ckpt is not None else None,
            "projector_ckpt": str(projector_ckpt) if projector_ckpt is not None else None,
        },
        "models": {},
        "comparisons": {},
    }

    all_model_records = []
    teacher_energy_records = []

    print("Loading teacher pipeline...")
    teacher_pipe = load_teacher_pipeline(args.teacher_id, device)

    teacher_summary, teacher_records = evaluate_model(
        "teacher",
        teacher_pipe,
        prompts,
        clip_model,
        clip_processor,
        image_dir,
        args,
        device,
    )
    summary["models"]["teacher"] = teacher_summary
    all_model_records.extend(teacher_records)

    if projector_ckpt is not None and args.teacher_candidates > 0:
        teacher_energy_summary, teacher_energy_records = evaluate_teacher_energy(
            teacher_pipe,
            projector_ckpt,
            prompts,
            clip_model,
            clip_processor,
            image_dir,
            args,
            device,
        )
        summary["teacher_energy"] = teacher_energy_summary

    print("Loading base student pipeline...")
    base_student_pipe = load_student_pipeline(args.student_id, teacher_pipe, device, checkpoint_path=None)
    base_summary, base_records = evaluate_model(
        "base_student",
        base_student_pipe,
        prompts,
        clip_model,
        clip_processor,
        image_dir,
        args,
        device,
    )
    summary["models"]["base_student"] = base_summary
    all_model_records.extend(base_records)
    del base_student_pipe
    torch.cuda.empty_cache()

    if distilled_ckpt is not None:
        print("Loading distilled student pipeline...")
        distilled_pipe = load_student_pipeline(
            args.student_id, teacher_pipe, device,
            checkpoint_path=distilled_ckpt, use_ema=args.use_ema,
        )
        distilled_summary, distilled_records = evaluate_model(
            "distilled_student",
            distilled_pipe,
            prompts,
            clip_model,
            clip_processor,
            image_dir,
            args,
            device,
        )
        summary["models"]["distilled_student"] = distilled_summary
        all_model_records.extend(distilled_records)
        del distilled_pipe
        torch.cuda.empty_cache()

    teacher_prompt_means = prompt_mean_map(teacher_records)
    base_prompt_means = prompt_mean_map(base_records)
    summary["comparisons"]["teacher_vs_base_student"] = compare_prompt_means(
        teacher_prompt_means,
        base_prompt_means,
    )

    if distilled_ckpt is not None:
        distilled_records_only = [r for r in all_model_records if r["model"] == "distilled_student"]
        distilled_prompt_means = prompt_mean_map(distilled_records_only)
        summary["comparisons"]["distilled_vs_base_student"] = compare_prompt_means(
            distilled_prompt_means,
            base_prompt_means,
        )
        summary["comparisons"]["teacher_vs_distilled_student"] = compare_prompt_means(
            teacher_prompt_means,
            distilled_prompt_means,
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_jsonl(output_dir / "model_samples.jsonl", all_model_records)
    if teacher_energy_records:
        write_jsonl(output_dir / "teacher_energy_samples.jsonl", teacher_energy_records)

    print("\nSaved evaluation outputs:")
    print(f"  Summary: {output_dir / 'summary.json'}")
    print(f"  Prompts: {output_dir / 'prompts.json'}")
    print(f"  Samples: {output_dir / 'model_samples.jsonl'}")
    if teacher_energy_records:
        print(f"  Teacher energy samples: {output_dir / 'teacher_energy_samples.jsonl'}")
    if image_dir is not None:
        print(f"  Images: {image_dir}")

    del teacher_pipe
    del clip_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
