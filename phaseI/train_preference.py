#!/usr/bin/env python3
"""Controlled pairwise preference fine-tuning of the four-step M1 student.

This is deliberately not winner-only consistency distillation. The same cached top/bottom teacher endpoints
are presented as a pair and the loss explicitly asks the student to prefer one over the other relative to a
frozen M1 reference. A deterministic random-label arm is an identical-compute placebo.

An optional REPA branch runs on a separate stream of real COCO images. Its projector is external to the DiT,
so this trainer explicitly all-reduces its gradients; the older Phase-R trainer did not.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseC.train_pilot import rollout_states
from train_self_distill import (
    COCOPairDataset,
    SAMPLE_PROMPTS,
    dist_barrier,
    param_l2_distance,
    post_clip_norm,
    sample_teacher_images,
    setup_distributed,
    vae_decode,
)


# Bands over the pre-shift uniform variable, so each band keeps the scheduler's shifted density
# restricted to a contiguous third of its own quantile range.
SIGMA_BANDS = {"all": (0.0, 1.0), "low": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "high": (2 / 3, 1.0)}
SIGMA_BAND_WIDTH = {name: hi - lo for name, (lo, hi) in SIGMA_BANDS.items()}


class SelectionDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.items = records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def randomized_flip(record_idx: int, label_seed: int) -> bool:
    payload = f"phaseI:{label_seed}:{record_idx}".encode()
    return bool(hashlib.blake2b(payload, digest_size=1).digest()[0] & 1)


def preferred_pair(record: dict, labels: str, label_seed: int) -> tuple[int, int, float, bool]:
    scores = [float(x) for x in record["endpoint_vqa"]]
    top = max(range(len(scores)), key=scores.__getitem__)
    bottom = min(range(len(scores)), key=scores.__getitem__)
    flip = labels == "randomized" and randomized_flip(int(record["idx"]), label_seed)
    positive, negative = (bottom, top) if flip else (top, bottom)
    return positive, negative, scores[positive] - scores[negative], flip


def fixed_pair(record: dict) -> tuple[int, int, float, bool]:
    """Read the serialized orientation of a frozen unordered pair.

    The fixed-pair causal design forbids recomputing an extremum at training time: pair membership
    must be a property of the cache, identical across arms, so that only the sign varies. This
    validates the record rather than deriving anything from the scores.
    """
    required = {"pair_a", "pair_b", "pair_key", "positive_idx", "negative_idx", "orientation"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"record {record.get('idx')} missing fixed-pair keys {sorted(missing)}")
    a, b = int(record["pair_a"]), int(record["pair_b"])
    positive, negative = int(record["positive_idx"]), int(record["negative_idx"])
    orientation = int(record["orientation"])
    if a == b:
        raise ValueError(f"record {record['idx']}: degenerate fixed pair")
    if {positive, negative} != {a, b}:
        raise ValueError(
            f"record {record['idx']}: orientation {(positive, negative)} is not a permutation "
            f"of the fixed pair {(a, b)}"
        )
    if orientation not in (-1, 1):
        raise ValueError(f"record {record['idx']}: orientation must be +/-1, got {orientation}")
    expected_positive = a if orientation > 0 else b
    if positive != expected_positive:
        raise ValueError(
            f"record {record['idx']}: orientation {orientation:+d} disagrees with "
            f"positive_idx={positive}"
        )
    scores = [float(x) for x in record["original_endpoint_vqa"]]
    margin = scores[positive] - scores[negative]
    return positive, negative, margin, orientation < 0


def pair_manifest(records: list[dict]) -> dict[str, str]:
    return {str(int(record["idx"])): str(record["pair_key"]) for record in records}


def assert_pair_manifest(records: list[dict], manifest_path: str | Path) -> None:
    """Hard cross-arm identity gate: this arm must train on exactly the same unordered pairs."""
    expected = json.loads(Path(manifest_path).read_text())
    if isinstance(expected, dict) and "pairs" in expected:
        expected = expected["pairs"]
    observed = pair_manifest(records)
    if observed != expected:
        only_here = set(observed) - set(expected)
        only_there = set(expected) - set(observed)
        differing = [k for k in set(observed) & set(expected) if observed[k] != expected[k]]
        raise RuntimeError(
            "fixed-pair manifest mismatch: "
            f"{len(only_here)} extra, {len(only_there)} missing, {len(differing)} differing "
            f"(first differing: {differing[:5]})"
        )


def load_selection(selection_dir: str | Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(Path(selection_dir).glob("selection_rank*.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            required = {"idx", "prompt", "seed_base", "endpoint_vqa"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys {sorted(missing)}")
            records.append(record)
    if not records:
        raise FileNotFoundError(f"no selection_rank*.jsonl under {selection_dir}")
    records.sort(key=lambda item: int(item["idx"]))
    indices = [int(item["idx"]) for item in records]
    if len(indices) != len(set(indices)):
        raise ValueError("selection cache contains duplicate prompt indices")
    n_values = {len(item["endpoint_vqa"]) for item in records}
    if len(n_values) != 1 or next(iter(n_values)) < 2:
        raise ValueError(f"inconsistent candidate counts in selection cache: {sorted(n_values)}")
    return records


def atomic_torch_save(payload: dict, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, destination)


def allreduce_external_grads(parameters, world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("REPA projector parameter has no gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def enable_trainable_parameters(module: torch.nn.Module) -> int:
    """Make a copied frozen backbone trainable and return its parameter count."""
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    if count == 0:
        raise RuntimeError("Phase-I student has no trainable parameters")
    return count


@torch.no_grad()
def assert_external_params_synced(parameters, world_size: int, atol: float = 2e-6) -> float:
    if world_size <= 1:
        return 0.0
    import torch.distributed as dist
    parameters = list(parameters)
    max_spread = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        value = parameter.detach().float()
        v_min, v_max = value.clone(), value.clone()
        dist.all_reduce(v_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(v_max, op=dist.ReduceOp.MAX)
        max_spread = torch.maximum(max_spread, (v_max - v_min).max())
    spread = float(max_spread.item())
    if not math.isfinite(spread) or spread > atol:
        raise RuntimeError(f"REPA projector desynchronized across ranks: max spread={spread:.3e}")
    return spread


def save_and_log_samples(
    *, model, pipe, scheduler, prompt_embeds, pooled_embeds, output_dir: Path,
    step: int, device, wandb_module=None, prefix: str = "student",
) -> None:
    from torchvision.utils import save_image

    was_training = model.training
    model.eval()
    images = sample_teacher_images(
        model, pipe.vae, scheduler, prompt_embeds, pooled_embeds,
        height_lat=512 // pipe.vae_scale_factor,
        width_lat=512 // pipe.vae_scale_factor,
        latent_channels=model.config.in_channels,
        num_steps=4, device=device, seed=271828,
    )
    if was_training:
        model.train()
    sample_dir = output_dir / "samples" / f"step_{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    wb_images = []
    for idx, image in enumerate(images):
        path = sample_dir / f"{prefix}_{idx}.png"
        save_image(image, path)
        if wandb_module is not None:
            wb_images.append(wandb_module.Image(str(path), caption=SAMPLE_PROMPTS[idx]))
    if wandb_module is not None:
        wandb_module.log({f"samples/{prefix}_4step": wb_images}, step=step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", choices=["oracle", "randomized", "fixed"], required=True)
    parser.add_argument(
        "--label_source",
        choices=[
            "auto",
            "correct_prompt",
            "counterfactual_prompt",
            "randomized",
            "inverted_prompt",
        ],
        default="auto",
        help="Auditable origin of the pair orientation; does not change student conditioning.",
    )
    parser.add_argument(
        "--pair_source",
        choices=["scores", "fixed_indices"],
        default="scores",
        help=(
            "'scores' recomputes argmax/argmin from endpoint_vqa (legacy Phase-I). "
            "'fixed_indices' reads the serialized frozen pair and never recomputes an extremum."
        ),
    )
    parser.add_argument(
        "--pair_manifest",
        default=None,
        help="JSON map idx->pair_key that this arm's cache must reproduce exactly.",
    )
    parser.add_argument(
        "--noise_coupling",
        choices=["shared", "independent"],
        default="shared",
        help="Corruption noise for the positive and negative branches of the pair.",
    )
    parser.add_argument(
        "--sigma_band",
        choices=["all", "low", "mid", "high"],
        default="all",
        help="Restrict the shifted-uniform sigma draw to a band, for the coalescence ablation.",
    )
    parser.add_argument(
        "--telemetry_every",
        type=int,
        default=0,
        help="If >0, append per-example (sigma, errors, logit) rows to telemetry.jsonl.",
    )
    parser.add_argument(
        "--ratio_clip",
        type=float,
        default=0.0,
        help=(
            "PSO-style clipping of the reference-relative ratio to [1-eps, 1+eps]. 0 disables it "
            "and reproduces the audited Phase-I logit exactly. PSO's bundled config uses 0.1."
        ),
    )
    parser.add_argument("--selection_dir", default="phaseC")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument(
        "--m1_checkpoint",
        default="checkpoints/method/M1_pure_consistency_20260624_065127_55187/checkpoint_final.pt",
    )
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--preference_timesteps", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--sched_shift", type=float, default=3.0)
    parser.add_argument("--num_train_ts", type=float, default=1000.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--label_seed", type=int, default=20260806)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--lambda_repa", type=float, default=0.0)
    parser.add_argument("--repa_layer", type=int, default=8)
    parser.add_argument("--repa_projector_lr", type=float, default=1e-4)
    parser.add_argument("--dinov2_id", default="facebook/dinov2-base")
    parser.add_argument("--coco_path", default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO")
    parser.add_argument("--coco_split", default="train2017")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=750)
    parser.add_argument("--sample_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="sd-pref-repa-v1")
    parser.add_argument("--wandb_run_name", required=True)
    parser.add_argument("--wandb_group", required=True)
    parser.add_argument("--wandb_tags", default="phaseI,preference,pilot")
    args = parser.parse_args()

    if args.label_source == "auto":
        args.label_source = "randomized" if args.labels == "randomized" else "correct_prompt"
    if args.labels == "randomized" and args.label_source != "randomized":
        raise ValueError("randomized labels require label_source=randomized")
    if args.labels == "oracle" and args.label_source == "randomized":
        raise ValueError("oracle labels cannot use label_source=randomized")
    if args.labels == "fixed" and args.pair_source != "fixed_indices":
        raise ValueError("--labels fixed requires --pair_source fixed_indices")
    if args.pair_source == "fixed_indices" and args.labels != "fixed":
        raise ValueError("--pair_source fixed_indices requires --labels fixed")

    if args.num_steps < 1 or args.preference_timesteps < 1:
        raise ValueError("num_steps and preference_timesteps must be positive")
    if args.beta <= 0 or args.lambda_anchor < 0 or args.lambda_repa < 0:
        raise ValueError("beta must be positive; loss weights must be non-negative")
    if not 0.0 <= args.ratio_clip < 1.0:
        raise ValueError("ratio_clip must lie in [0, 1)")
    if args.height != 512:
        raise ValueError("Phase-I sampling/logging is fixed at 512px for a matched M1 comparison")

    rank, world_size, _local_rank, device, is_main = setup_distributed(0)
    torch.manual_seed(args.seed + rank * 1009)
    random.seed(args.seed + rank * 1009)

    output_dir = Path(args.output_dir).resolve()
    if is_main:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
    dist_barrier()

    records = load_selection(args.selection_dir)
    selection_files = sorted(Path(args.selection_dir).glob("selection_rank*.jsonl"))
    if args.pair_source == "fixed_indices":
        if args.pair_manifest:
            assert_pair_manifest(records, args.pair_manifest)
        arm_names = {record.get("arm") for record in records}
        if len(arm_names) != 1:
            raise RuntimeError(f"selection cache mixes fixed-pair arms: {sorted(arm_names)}")
    raw_top_bottom = []
    assigned_margins = []
    flips = 0
    reversal_records = 0
    for record in records:
        if args.pair_source == "fixed_indices":
            scores = [float(x) for x in record["original_endpoint_vqa"]]
            _p, _n, margin, flip = fixed_pair(record)
            reversal_records += int(bool(record.get("pair_reverses_under_counterfactual")))
        else:
            scores = [float(x) for x in record["endpoint_vqa"]]
            _p, _n, margin, flip = preferred_pair(record, args.labels, args.label_seed)
        raw_top_bottom.append(max(scores) - min(scores))
        assigned_margins.append(margin)
        flips += int(flip)

    code_files = [Path(__file__), ROOT / "phaseR" / "repa_loss.py", ROOT / "phaseC" / "train_pilot.py"]
    # Hash the 19.8 GB checkpoint once, not concurrently from every DDP rank.
    checkpoint_sha256 = file_sha256(args.m1_checkpoint) if is_main else None
    if world_size > 1:
        import torch.distributed as dist
        shared_hash = [checkpoint_sha256]
        dist.broadcast_object_list(shared_hash, src=0)
        checkpoint_sha256 = shared_hash[0]

    manifest = {
        **vars(args),
        "output_dir": str(output_dir),
        "git_revision": git_revision(),
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "m1_checkpoint_sha256": checkpoint_sha256,
        "selection_records": len(records),
        "selection_files_sha256": {path.name: file_sha256(path) for path in selection_files},
        "candidate_count": len(records[0]["endpoint_vqa"]),
        "raw_top_bottom_margin_mean": sum(raw_top_bottom) / len(raw_top_bottom),
        "assigned_margin_mean": sum(assigned_margins) / len(assigned_margins),
        "randomized_flip_fraction": flips / len(records),
        "fixed_pair_reversal_records": reversal_records,
        "selection_pair_key_sha256": hashlib.sha256(
            json.dumps(pair_manifest(records), sort_keys=True).encode()
        ).hexdigest()
        if args.pair_source == "fixed_indices"
        else None,
        "world_size": world_size,
        "created_unix": time.time(),
    }
    if is_main:
        (output_dir / "args.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    dist_barrier()

    from diffusers import StableDiffusion3Pipeline
    if is_main:
        print(f"Loading base pipeline and M1 checkpoint {args.m1_checkpoint}", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        module.to(dtype=torch.bfloat16).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    # Base SD3.5-M is used only to deterministically reconstruct the cached teacher candidates.
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    checkpoint = torch.load(args.m1_checkpoint, map_location="cpu", weights_only=False)
    student = copy.deepcopy(teacher).to(dtype=torch.float32)
    missing, unexpected = student.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"M1 load mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    del checkpoint
    # The teacher is frozen before deepcopy and deepcopy preserves requires_grad.
    # Explicitly re-enable the student before constructing DDP and the optimizer.
    trainable_parameter_count = enable_trainable_parameters(student)
    student.train()
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    reference = copy.deepcopy(student).to(dtype=torch.bfloat16).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    manifest["trainable_parameter_count"] = trainable_parameter_count
    if is_main:
        (output_dir / "args.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"Trainable student parameters: {trainable_parameter_count:,}", flush=True)
    dist_barrier()

    student_ddp = student
    student_module = student
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        student_ddp = DDP(student, device_ids=[device.index], find_unused_parameters=False)
        student_module = student_ddp.module

    repa_aux = None
    repa_dino = None
    repa_parameters: list[torch.nn.Parameter] = []
    coco_iter = None
    coco_loader = None
    coco_sampler = None
    coco_epoch = 0
    if args.lambda_repa > 0:
        from phaseR.repa_loss import RepaAux
        from transformers import Dinov2Model

        repa_dino = Dinov2Model.from_pretrained(args.dinov2_id).to(device).eval()
        for parameter in repa_dino.parameters():
            parameter.requires_grad = False
        repa_aux = RepaAux(
            student_module, layer=args.repa_layer, out_dim=768, device=device, dtype=torch.float32
        )
        # Discover the IMAGE-stream hidden width. `joint_attention_dim` is the text-conditioning
        # width, not the DiT hidden width; using it here would build a projector that only fails on
        # the first real REPA forward. Arm the hook for one shape-only forward instead.
        repa_aux._armed = True
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            dummy_latent = torch.zeros(
                1, student_module.config.in_channels,
                args.height // pipe.vae_scale_factor,
                args.height // pipe.vae_scale_factor,
                device=device, dtype=torch.bfloat16,
            )
            student_module(
                hidden_states=dummy_latent,
                timestep=torch.tensor([500.0], device=device, dtype=torch.bfloat16),
                encoder_hidden_states=torch.zeros(
                    1, 77, student_module.config.joint_attention_dim,
                    device=device, dtype=torch.bfloat16,
                ),
                pooled_projections=torch.zeros(
                    1, student_module.config.pooled_projection_dim,
                    device=device, dtype=torch.bfloat16,
                ),
                return_dict=False,
            )
        repa_aux._armed = False
        if "h" not in repa_aux._h:
            raise RuntimeError("REPA shape-discovery hook did not fire")
        repa_hidden_dim = repa_aux._h.pop("h").shape[-1]

        # Create the external projector with identical initialization on every rank.
        cpu_rng = torch.random.get_rng_state()
        torch.manual_seed(args.seed + 424242)
        repa_aux._ensure_proj(repa_hidden_dim)
        torch.random.set_rng_state(cpu_rng)
        repa_parameters = list(repa_aux.parameters())
        if world_size > 1:
            import torch.distributed as dist
            for parameter in repa_parameters:
                dist.broadcast(parameter.data, src=0)

        coco_dataset = COCOPairDataset(
            args.coco_path, split=args.coco_split, image_size=args.height
        )
        coco_sampler = DistributedSampler(
            coco_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=args.seed + 17, drop_last=True,
        ) if world_size > 1 else None
        coco_loader = DataLoader(
            coco_dataset, batch_size=1, sampler=coco_sampler,
            shuffle=(coco_sampler is None), drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0, collate_fn=lambda batch: batch[0],
        )
        coco_iter = iter(coco_loader)

    optimizer_groups = [{"params": list(student_module.parameters()), "lr": args.lr, "name": "student"}]
    if repa_parameters:
        optimizer_groups.append({
            "params": repa_parameters, "lr": args.repa_projector_lr, "name": "repa_projector"
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / max(1, args.num_warmup_steps))
    )

    dataset = SelectionDataset(records)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=(sampler is None),
        drop_last=True, collate_fn=lambda batch: batch[0]
    )
    data_iter = iter(loader)
    epoch = 0

    with torch.no_grad():
        neg_embeds, _, neg_pool, _ = pipe.encode_prompt(
            prompt=[""], prompt_2=[""], prompt_3=[""],
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
        )
        fixed_prompts = SAMPLE_PROMPTS[:4]
        fixed_embeds, _, fixed_pool, _ = pipe.encode_prompt(
            prompt=fixed_prompts, prompt_2=fixed_prompts, prompt_3=fixed_prompts,
            do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
        )

    wandb_module = None
    if is_main and not args.no_wandb:
        os.environ["WANDB_MODE"] = "online"
        import wandb
        wandb_module = wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_name,
            group=args.wandb_group,
            job_type=f"train-{args.labels}-repa{args.lambda_repa:g}",
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            + [args.label_source],
            config=manifest,
            dir=str(output_dir),
            resume="never",
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("system/*", step_metric="train/step")
        wandb.run.summary["output_dir"] = str(output_dir)
        wandb.run.summary["m1_checkpoint"] = str(Path(args.m1_checkpoint).resolve())

    # Step-zero samples make visual drift auditable instead of relying on memory of M1.
    dist_barrier()
    if is_main:
        save_and_log_samples(
            model=student_module, pipe=pipe, scheduler=pipe.scheduler,
            prompt_embeds=fixed_embeds, pooled_embeds=fixed_pool,
            output_dir=output_dir, step=0, device=device,
            wandb_module=wandb_module, prefix="student",
        )
    dist_barrier()

    lat_channels = student_module.config.in_channels
    latent_hw = args.height // pipe.vae_scale_factor
    progress = tqdm(total=args.num_steps, disable=not is_main, desc=f"pref-{args.labels}")
    global_step = 0
    last_log_time = time.monotonic()
    steps_since_log = 0
    repa_spread = 0.0

    while global_step < args.num_steps:
        try:
            record = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            record = next(data_iter)

        if args.pair_source == "fixed_indices":
            positive_idx, negative_idx, assigned_margin, label_flipped = fixed_pair(record)
            # Roll the teacher out in the pair's canonical (a, b) order, never in label order, so
            # the two endpoint tensors are bit-identical across arms and only the sign differs.
            rollout_order = (int(record["pair_a"]), int(record["pair_b"]))
        else:
            positive_idx, negative_idx, assigned_margin, label_flipped = preferred_pair(
                record, args.labels, args.label_seed
            )
            rollout_order = (positive_idx, negative_idx)
        positive_slot = rollout_order.index(positive_idx)
        negative_slot = rollout_order.index(negative_idx)
        prompt = record["prompt"]
        label_prompt = record.get("label_prompt", prompt)
        record_label_source = record.get("label_source", args.label_source)
        if args.label_source != "randomized" and record_label_source != args.label_source:
            raise RuntimeError(
                f"selection record {record['idx']} has label_source={record_label_source!r}, "
                f"expected {args.label_source!r}"
            )
        seed_base = int(record["seed_base"])

        with torch.no_grad():
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
                do_classifier_free_guidance=False, device=device, num_images_per_prompt=1,
            )
            starts = []
            for candidate_idx in rollout_order:
                generator = torch.Generator(device=device).manual_seed(seed_base + candidate_idx)
                starts.append(torch.randn(
                    1, lat_channels, latent_hw, latent_hw, device=device,
                    dtype=torch.bfloat16, generator=generator,
                ))
            z0 = torch.cat(starts, dim=0)
            states, sigmas = rollout_states(
                teacher, pipe.scheduler, z0, prompt_embeds, pooled,
                neg_embeds, neg_pool, args.K, args.cfg, device,
            )
            endpoints = states[args.K].float().detach()  # rollout_order slots

            count = args.preference_timesteps
            u = torch.rand(count, device=device)
            u = SIGMA_BANDS[args.sigma_band][0] + SIGMA_BAND_WIDTH[args.sigma_band] * u
            sigma = (args.sched_shift * u) / (1 + (args.sched_shift - 1) * u)
            sigma = sigma.clamp(1e-4, 1 - 1e-4)
            noise = torch.randn(count, lat_channels, latent_hw, latent_hw, device=device)
            # Drawn last so that the shared-noise path consumes RNG identically to the legacy run;
            # z_pos is then bit-identical between the two coupling modes and only z_neg differs.
            noise_negative = (
                torch.randn(count, lat_channels, latent_hw, latent_hw, device=device)
                if args.noise_coupling == "independent"
                else noise
            )
            sigma_b = sigma.view(-1, 1, 1, 1)
            pos = endpoints[positive_slot : positive_slot + 1].expand(count, -1, -1, -1)
            neg = endpoints[negative_slot : negative_slot + 1].expand(count, -1, -1, -1)
            z_pos = (1 - sigma_b) * pos + sigma_b * noise
            z_neg = (1 - sigma_b) * neg + sigma_b * noise_negative
            target_pos = noise - pos
            target_neg = noise_negative - neg
            z_input = torch.cat([z_pos, z_neg], dim=0).to(torch.bfloat16)
            targets = torch.cat([target_pos, target_neg], dim=0)
            timesteps = torch.cat([sigma, sigma], dim=0) * args.num_train_ts
            cond = prompt_embeds.repeat(2 * count, 1, 1)
            cond_pool = pooled.repeat(2 * count, 1)
            with torch.autocast("cuda", torch.bfloat16):
                v_reference = reference(
                    hidden_states=z_input, timestep=timesteps,
                    encoder_hidden_states=cond, pooled_projections=cond_pool,
                    return_dict=False,
                )[0]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            v_student = student_ddp(
                hidden_states=z_input, timestep=timesteps,
                encoder_hidden_states=cond, pooled_projections=cond_pool,
                return_dict=False,
            )[0]
        per_example_error = (v_student.float() - targets.float()).pow(2).mean(dim=(1, 2, 3))
        reference_error = (v_reference.float() - targets.float()).pow(2).mean(dim=(1, 2, 3))
        error_pos = per_example_error[:count].mean()
        error_neg = per_example_error[count:].mean()
        ref_pos = reference_error[:count].mean()
        ref_neg = reference_error[count:].mean()
        model_gap = error_pos - error_neg
        reference_gap = ref_pos - ref_neg
        # Per-branch reference-relative log ratio under the Gaussian transition bridge:
        #   log p_theta(u | z, c) - log p_0(u | z, c)  ∝  -(e_theta(u) - e_0(u)).
        # With no clipping, beta * (lr_pos - lr_neg) == -beta * (Delta_theta - Delta_0), i.e. the
        # audited Phase-I logit exactly. Under this bridge our objective IS the offline PSO /
        # Diffusion-DPO equation; the surviving differences from PSO are ratio clipping, pair-noise
        # coupling, the explicit M1 anchor, and beta.
        log_ratio_pos = -(error_pos - ref_pos)
        log_ratio_neg = -(error_neg - ref_neg)
        if args.ratio_clip > 0:
            lo = math.log(1.0 - args.ratio_clip)
            hi = math.log(1.0 + args.ratio_clip)
            log_ratio_pos = log_ratio_pos.clamp(lo, hi)
            log_ratio_neg = log_ratio_neg.clamp(lo, hi)
        preference_logit = args.beta * (log_ratio_pos - log_ratio_neg)
        loss_preference = -F.logsigmoid(preference_logit)
        loss_anchor = F.mse_loss(v_student.float(), v_reference.float())
        total_loss = loss_preference + args.lambda_anchor * loss_anchor

        loss_repa = torch.zeros((), device=device)
        coco_item = None
        if repa_aux is not None:
            assert coco_iter is not None and coco_loader is not None
            try:
                coco_item = next(coco_iter)
            except StopIteration:
                coco_epoch += 1
                if coco_sampler is not None:
                    coco_sampler.set_epoch(coco_epoch)
                coco_iter = iter(coco_loader)
                coco_item = next(coco_iter)
            with torch.no_grad():
                real_embeds, _, real_pool, _ = pipe.encode_prompt(
                    prompt=[coco_item["caption"]], prompt_2=[coco_item["caption"]],
                    prompt_3=[coco_item["caption"]], do_classifier_free_guidance=False,
                    device=device, num_images_per_prompt=1,
                )
                image = coco_item["image_tensor"].unsqueeze(0).to(device, dtype=pipe.vae.dtype)
                posterior = pipe.vae.encode(image).latent_dist
                z_real = (
                    (posterior.mean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                ).float().detach()
            loss_repa = repa_aux.loss(
                student_ddp, z_real,
                coco_item["image_tensor"].unsqueeze(0).to(device, dtype=torch.float32),
                repa_dino, real_embeds, real_pool,
                sigmas, pipe.scheduler.timesteps.to(device),
            )
            total_loss = total_loss + args.lambda_repa * loss_repa

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"non-finite total loss at step {global_step}: "
                f"pref={loss_preference.item()} anchor={loss_anchor.item()} repa={loss_repa.item()}"
            )
        total_loss.backward()
        allreduce_external_grads(repa_parameters, world_size)

        student_grad_pre_t = torch.nn.utils.clip_grad_norm_(student_module.parameters(), args.grad_clip)
        student_grad_pre = float(student_grad_pre_t.item())
        if not math.isfinite(student_grad_pre) or student_grad_pre <= 0:
            raise FloatingPointError(f"invalid student gradient norm {student_grad_pre} at step {global_step}")
        repa_grad_pre = 0.0
        if repa_parameters:
            repa_grad_pre_t = torch.nn.utils.clip_grad_norm_(repa_parameters, 5.0)
            repa_grad_pre = float(repa_grad_pre_t.item())
            if not math.isfinite(repa_grad_pre) or repa_grad_pre <= 0:
                raise FloatingPointError(f"invalid REPA projector gradient {repa_grad_pre}")

        optimizer.step()
        lr_scheduler.step()
        global_step += 1
        progress.update(1)
        steps_since_log += 1

        # Per-example rows are what task K needs to calibrate beta and what task N needs to test the
        # high-sigma coalescence hypothesis; step-level scalars average that structure away.
        if is_main and args.telemetry_every > 0 and global_step % args.telemetry_every == 0:
            # Hash the teacher endpoints in canonical pair order. Four arms that agree here are
            # provably training on the same two images, which is the claim the design rests on.
            endpoint_sha = hashlib.sha256(
                endpoints.detach().to(torch.float32).cpu().numpy().tobytes()
            ).hexdigest()
            sigma_row = sigma.detach().float().tolist()
            per_pos = per_example_error[:count].detach().float().tolist()
            per_neg = per_example_error[count:].detach().float().tolist()
            ref_row_pos = reference_error[:count].detach().float().tolist()
            ref_row_neg = reference_error[count:].detach().float().tolist()
            with open(output_dir / "telemetry.jsonl", "a") as handle:
                handle.write(json.dumps({
                    "step": global_step,
                    "idx": int(record["idx"]),
                    "category": record.get("category"),
                    "arm": record.get("arm", args.label_source),
                    "orientation": record.get("orientation"),
                    "pair_key": record.get("pair_key"),
                    "rollout_order": list(rollout_order),
                    "endpoints_sha256": endpoint_sha,
                    "positive_slot": positive_slot,
                    "reverses": record.get("pair_reverses_under_counterfactual"),
                    "assigned_margin": assigned_margin,
                    "sigma_band": args.sigma_band,
                    "noise_coupling": args.noise_coupling,
                    "sigma": sigma_row,
                    "e_theta_pos": per_pos,
                    "e_theta_neg": per_neg,
                    "e_ref_pos": ref_row_pos,
                    "e_ref_neg": ref_row_neg,
                    "delta_theta": float(model_gap.detach().item()),
                    "delta_ref": float(reference_gap.detach().item()),
                    "logit": float(preference_logit.detach().item()),
                    "beta": args.beta,
                    "grad_norm_pre_clip": student_grad_pre,
                }) + "\n")

        if repa_parameters and global_step == 1:
            repa_spread = assert_external_params_synced(repa_parameters, world_size)
            if is_main:
                print(f"REPA DDP synchronization gate passed: max spread={repa_spread:.3e}", flush=True)

        if is_main and global_step % args.log_every == 0:
            now = time.monotonic()
            elapsed = max(now - last_log_time, 1e-9)
            steps_per_second = steps_since_log / elapsed
            last_log_time, steps_since_log = now, 0
            drift = param_l2_distance(student_module, reference)
            metrics = {
                "train/step": global_step,
                "train/loss_total": float(total_loss.detach().item()),
                "train/loss_preference": float(loss_preference.detach().item()),
                "train/loss_anchor": float(loss_anchor.detach().item()),
                "train/loss_anchor_weighted": args.lambda_anchor * float(loss_anchor.detach().item()),
                "train/loss_repa": float(loss_repa.detach().item()),
                "train/loss_repa_weighted": args.lambda_repa * float(loss_repa.detach().item()),
                "train/preference_logit": float(preference_logit.detach().item()),
                "train/preference_accuracy": float(preference_logit.detach().item() > 0),
                "train/model_error_positive": float(error_pos.detach().item()),
                "train/model_error_negative": float(error_neg.detach().item()),
                "train/model_error_gap": float(model_gap.detach().item()),
                "train/reference_error_gap": float(reference_gap.detach().item()),
                "train/positive_error_improvement_vs_reference": float(
                    (ref_pos - error_pos).detach().item()
                ),
                "train/negative_error_increase_vs_reference": float(
                    (error_neg - ref_neg).detach().item()
                ),
                "train/gap_improvement_vs_reference": float((reference_gap - model_gap).detach().item()),
                "train/assigned_reward_margin": float(assigned_margin),
                "train/label_flipped": float(label_flipped),
                "train/sigma_mean": float(sigma.mean().item()),
                "train/sigma_min": float(sigma.min().item()),
                "train/sigma_max": float(sigma.max().item()),
                "train/reference_error_positive": float(ref_pos.detach().item()),
                "train/reference_error_negative": float(ref_neg.detach().item()),
                "train/pair_reverses_under_counterfactual": float(
                    bool(record.get("pair_reverses_under_counterfactual"))
                ),
                "train/orientation": float(record.get("orientation", 0) or 0),
                "train/label_prompt_is_counterfactual": float(
                    args.label_source == "counterfactual_prompt"
                ),
                "train/student_grad_norm_pre_clip": student_grad_pre,
                "train/student_grad_norm_post_clip": post_clip_norm(student_grad_pre, args.grad_clip),
                "train/student_clip_ratio": post_clip_norm(student_grad_pre, args.grad_clip) / student_grad_pre,
                "train/repa_projector_grad_norm_pre_clip": repa_grad_pre,
                "train/repa_projector_ddp_max_spread": repa_spread,
                "train/param_l2_from_m1": drift,
                "train/lr_student": lr_scheduler.get_last_lr()[0],
                "train/lr_repa_projector": (
                    lr_scheduler.get_last_lr()[1] if len(lr_scheduler.get_last_lr()) > 1 else 0.0
                ),
                "system/steps_per_second": steps_per_second,
                "system/max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "system/memory_reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
            }
            progress.set_postfix({
                "L": f"{metrics['train/loss_total']:.3g}",
                "pref": f"{metrics['train/loss_preference']:.3g}",
                "logit": f"{metrics['train/preference_logit']:+.2g}",
                "drift": f"{drift:.2g}",
            })
            if wandb_module is not None:
                wandb_module.log(metrics, step=global_step)

        if global_step % args.sample_every == 0 or global_step == args.num_steps:
            dist_barrier()
            if is_main:
                save_and_log_samples(
                    model=student_module, pipe=pipe, scheduler=pipe.scheduler,
                    prompt_embeds=fixed_embeds, pooled_embeds=fixed_pool,
                    output_dir=output_dir, step=global_step, device=device,
                    wandb_module=wandb_module, prefix="student",
                )
                # Log the actual positive/negative teacher pair that drove this update.
                pair_images = ((vae_decode(pipe.vae, endpoints) + 1) / 2).clamp(0, 1).cpu()
                from torchvision.utils import save_image
                pair_dir = output_dir / "preference_pairs" / f"step_{global_step:07d}"
                pair_dir.mkdir(parents=True, exist_ok=False)
                pair_paths = []
                for role, slot in (("positive", positive_slot), ("negative", negative_slot)):
                    path = pair_dir / f"{role}.png"
                    save_image(pair_images[slot], path)
                    pair_paths.append(path)
                (pair_dir / "metadata.json").write_text(json.dumps({
                    "prompt": prompt,
                    "conditioning_prompt": prompt,
                    "label_prompt": label_prompt,
                    "label_source": args.label_source,
                    "record_idx": int(record["idx"]),
                    "positive_candidate": int(positive_idx),
                    "negative_candidate": int(negative_idx),
                    "rollout_order": list(rollout_order),
                    "pair_key": record.get("pair_key"),
                    "orientation": record.get("orientation"),
                    "pair_reverses_under_counterfactual": record.get(
                        "pair_reverses_under_counterfactual"
                    ),
                    "assigned_reward_margin": assigned_margin,
                    "label_flipped": label_flipped,
                }, indent=2))
                if wandb_module is not None:
                    wandb_module.log({
                        "samples/training_pair": [
                            wandb_module.Image(
                                str(pair_paths[0]),
                                caption=(
                                    f"POS | score margin {assigned_margin:+.3f} | "
                                    f"condition: {prompt} | label text: {label_prompt}"
                                ),
                            ),
                            wandb_module.Image(
                                str(pair_paths[1]),
                                caption=f"NEG | condition: {prompt} | label text: {label_prompt}",
                            ),
                        ]
                    }, step=global_step)
            dist_barrier()

        if global_step % args.save_every == 0 or global_step == args.num_steps:
            dist_barrier()
            if is_main:
                checkpoint_path = output_dir / f"checkpoint_step{global_step}.pt"
                atomic_torch_save({
                    "model": student_module.state_dict(),
                    "repa_projector": repa_aux.projector.state_dict() if repa_aux is not None else None,
                    "step": global_step,
                    "labels": args.labels,
                    "label_source": args.label_source,
                    "lambda_repa": args.lambda_repa,
                    "wandb_run_name": args.wandb_run_name,
                    "manifest": manifest,
                }, checkpoint_path)
                (output_dir / "checkpoint_latest.json").write_text(json.dumps({
                    "step": global_step, "path": str(checkpoint_path),
                }, indent=2))
                if wandb_module is not None:
                    wandb_module.run.summary["latest_checkpoint"] = str(checkpoint_path)
                    wandb_module.run.summary["latest_checkpoint_step"] = global_step
                print(f"Saved {checkpoint_path}", flush=True)
            dist_barrier()

    progress.close()
    if is_main:
        final_step_path = output_dir / f"checkpoint_step{global_step}.pt"
        final_path = output_dir / "checkpoint_final.pt"
        if not final_path.exists():
            os.link(final_step_path, final_path)
        if wandb_module is not None:
            wandb_module.run.summary["final_checkpoint"] = str(final_path)
            wandb_module.run.summary["completed_steps"] = global_step
            wandb_module.finish()
    dist_barrier()
    if repa_aux is not None:
        repa_aux.remove()
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
