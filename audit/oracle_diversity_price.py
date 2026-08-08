#!/usr/bin/env python3
"""Task E item 7 — what does best-of-K selection itself cost in diversity?

The counterfactual campaign failed its DINO diversity gate. That is only damning if the student is
*less* diverse than the policy it is meant to amortize. Best-of-K selection is itself a
mode-narrowing operator, so the honest reference is not M1 alone but the teacher-best-of-4 policy.

The analytic displacement KL_4 = log 4 - 3/4 ~ 0.636 nats is NOT comparable to a DINO cosine
distance or an LPIPS score; it is in different units entirely. This measures the price empirically,
in exactly the DINO and LPIPS units `phaseC/diversity_eval.py` reports for the students.

Method: the frozen-teacher bank stores 8 candidates per prompt. For each prompt we draw `--n_sets`
quartets from those 8 candidates and keep the VQAScore argmax of each, giving `--n_sets` images
produced by the best-of-4 policy. The control draws `--n_sets` candidates uniformly. Both sets are
then scored with the same mean-pairwise-DINO and mean-pairwise-LPIPS estimators the student
evaluation uses, so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@torch.no_grad()
def dino_embed(paths, model, proc, device, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        h = model(pixel_values=px).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def lpips_pairwise_from_tensor(x, net):
    i, j = zip(*combinations(range(x.shape[0]), 2))
    d = net(x[list(i)], x[list(j)]).squeeze()
    return float(d.mean())


def load_lpips_tensor(paths, device):
    ts = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        ts.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(ts).to(device)


def mean_pairwise_dino(embeddings: torch.Tensor) -> float:
    cos = embeddings @ embeddings.T
    iu = torch.triu_indices(cos.shape[0], cos.shape[0], offset=1)
    return float(1.0 - cos[iu[0], iu[1]].mean())


def load_bank(scores_dir: Path, config: str) -> list[dict]:
    records = []
    for path in sorted(scores_dir.glob("scores_rank*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") == config:
                records.append(row)
    records.sort(key=lambda r: int(r["idx"]))
    return records


def boot_ci(values, rng, reps=10000):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", default="exp0/images")
    parser.add_argument("--scores_dir", default="exp0/scores")
    parser.add_argument("--config", default="cfg7_s8", help="teacher configuration subdirectory")
    parser.add_argument("--K", type=int, default=4, help="selection pool size (best-of-K)")
    parser.add_argument("--n_sets", type=int, default=4, help="images per prompt per policy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--dino_id", default="facebook/dinov2-base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    records = load_bank(Path(args.scores_dir), args.config)
    if not records:
        raise SystemExit(f"no scored records for config {args.config}")
    if args.limit:
        records = records[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    import lpips

    proc = AutoImageProcessor.from_pretrained(args.dino_id)
    dino = AutoModel.from_pretrained(args.dino_id, torch_dtype=torch.float16).to(device).eval()
    lp = lpips.LPIPS(net="alex").to(device).eval()

    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = 0
    for record in records:
        idx = int(record["idx"])
        scores = np.asarray(record["vqa"], dtype=np.float64)
        n_candidates = scores.shape[0]
        if n_candidates < args.K:
            skipped += 1
            continue
        prompt_dir = Path(args.images_root) / f"p{idx:05d}" / args.config
        paths = [prompt_dir / f"cand{j}.png" for j in range(n_candidates)]
        if not all(p.exists() for p in paths):
            skipped += 1
            continue

        # One embedding pass per prompt; both policies index into it.
        embeddings = dino_embed([str(p) for p in paths], dino, proc, device)
        tensors = load_lpips_tensor(paths, device)

        # best-of-K policy: n_sets independent quartets, keep each quartet's VQAScore argmax.
        selected = []
        for _ in range(args.n_sets):
            pool = rng.choice(n_candidates, size=args.K, replace=False)
            selected.append(int(pool[int(np.argmax(scores[pool]))]))
        # random policy: n_sets independent uniform draws, matched draw count.
        random_pick = [int(rng.integers(0, n_candidates)) for _ in range(args.n_sets)]

        entry = {"idx": idx, "category": record.get("category")}
        for name, picks in (("best", selected), ("random", random_pick)):
            # Duplicate picks are a real property of the policy (best-of-K concentrates), so they
            # are kept: a repeated image contributes zero distance, which is the effect being
            # measured, not a bug to deduplicate away.
            entry[f"{name}_dino"] = mean_pairwise_dino(embeddings[picks])
            entry[f"{name}_lpips"] = lpips_pairwise_from_tensor(tensors[picks], lp)
            entry[f"{name}_unique"] = len(set(picks))
            entry[f"{name}_mean_vqa"] = float(scores[picks].mean())
        rows.append(entry)

    if not rows:
        raise SystemExit("no prompts with a complete candidate bank")

    boot_rng = np.random.default_rng(args.seed + 1)
    summary = {}
    for metric in ("dino", "lpips"):
        best = [r[f"best_{metric}"] for r in rows]
        rand = [r[f"random_{metric}"] for r in rows]
        delta = [b - r for b, r in zip(best, rand)]
        m_b, lo_b, hi_b = boot_ci(best, np.random.default_rng(args.seed + 2))
        m_r, lo_r, hi_r = boot_ci(rand, np.random.default_rng(args.seed + 2))
        m_d, lo_d, hi_d = boot_ci(delta, boot_rng)
        summary[metric] = {
            "teacher_best_of_k": {"mean": m_b, "ci95": [lo_b, hi_b]},
            "teacher_random": {"mean": m_r, "ci95": [lo_r, hi_r]},
            "price": {"delta": m_d, "ci95": [lo_d, hi_d]},
        }
    for metric in ("mean_vqa", "unique"):
        best = [r[f"best_{metric}"] for r in rows]
        rand = [r[f"random_{metric}"] for r in rows]
        summary[metric] = {
            "teacher_best_of_k": float(np.mean(best)),
            "teacher_random": float(np.mean(rand)),
            "delta": float(np.mean(best) - np.mean(rand)),
        }

    report = {
        "config": args.config,
        "K": args.K,
        "n_sets": args.n_sets,
        "prompts": len(rows),
        "skipped": skipped,
        "seed": args.seed,
        "dino_id": args.dino_id,
        "estimator": (
            "mean pairwise DINOv2 CLS cosine distance and mean pairwise LPIPS(alex) over n_sets "
            "images per prompt, identical to phaseC/diversity_eval.py"
        ),
        "summary": summary,
        "per_prompt": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.md:
        d, l = summary["dino"], summary["lpips"]
        lines = [
            f"# Diversity price of teacher best-of-{args.K} selection",
            "",
            f"{len(rows)} prompts from the frozen-teacher `{args.config}` bank, "
            f"{args.n_sets} images per prompt per policy, same DINO/LPIPS estimators as the "
            "student diversity gate.",
            "",
            "| metric | teacher best-of-K | teacher random | price (best − random) |",
            "|---|---:|---:|---|",
            f"| DINO diversity | {d['teacher_best_of_k']['mean']:.5f} | "
            f"{d['teacher_random']['mean']:.5f} | **{d['price']['delta']:+.5f}** "
            f"[{d['price']['ci95'][0]:+.5f}, {d['price']['ci95'][1]:+.5f}] |",
            f"| LPIPS diversity | {l['teacher_best_of_k']['mean']:.5f} | "
            f"{l['teacher_random']['mean']:.5f} | **{l['price']['delta']:+.5f}** "
            f"[{l['price']['ci95'][0]:+.5f}, {l['price']['ci95'][1]:+.5f}] |",
            f"| mean VQAScore | {summary['mean_vqa']['teacher_best_of_k']:.5f} | "
            f"{summary['mean_vqa']['teacher_random']:.5f} | "
            f"{summary['mean_vqa']['delta']:+.5f} |",
            f"| distinct images per prompt | {summary['unique']['teacher_best_of_k']:.3f} | "
            f"{summary['unique']['teacher_random']:.3f} | {summary['unique']['delta']:+.3f} |",
            "",
            "## How to use this number",
            "",
            "The oracle policy the student is asked to amortize is itself less diverse than random "
            "teacher sampling by the amount above, measured in the same units as the student gate. "
            "A student whose diversity loss is no larger than this price has not become worse than "
            "what it imitates; a student that loses more has paid a cost the policy does not "
            "explain. Report both comparisons rather than the M1 comparison alone.",
            "",
            "The analytic best-of-4 KL displacement log 4 - 3/4 ~ 0.636 nats is in different units "
            "and cannot be compared with these values.",
        ]
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({"prompts": len(rows), "skipped": skipped, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
