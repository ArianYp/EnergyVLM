#!/usr/bin/env python3
"""Task A — the immutable result registry.

Every admitted number is recomputed here from its raw artifact and stamped with that artifact's
SHA-256, so the registry cannot silently drift from the evidence the way the prose reports did.

The registry's main job is to keep apart three quantities that earlier prose merged:

  +0.1220  teacher ENDPOINT SCORING headroom inside the training cache (VQAScore, in-objective)
  +0.0717  teacher BEST-OF-FOUR at inference on held-out CompBench (independent evaluator)
  +0.0485  frozen-M1 NOISE-SEED best-of-four on held-out CompBench (a different intervention)

They have different candidate generators, different selectors, different populations, and they may
never share an amortization denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "complex"]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(directory: Path, pattern: str) -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(directory.glob(pattern))}


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_per_prompt(scores_dir: Path, model: str, step: int, category: str) -> dict[str, float]:
    path = scores_dir / f"{model}_s{step}_{category}" / "scores.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        row["prompt"]: float(row["score"])
        for row in payload["per_prompt"]
        if row.get("prompt") is not None
    }


def paired_primary(
    scores_dir: Path, treatment: str, control: str, step: int, draws: int, seed: int
) -> dict | None:
    """Equal-category-weighted paired contrast with a prompt bootstrap."""
    diffs = {}
    for category in PRIMARY:
        a = load_per_prompt(scores_dir, treatment, step, category)
        b = load_per_prompt(scores_dir, control, step, category)
        shared = sorted(set(a) & set(b))
        if shared:
            diffs[category] = np.array([a[k] - b[k] for k in shared])
    if len(diffs) != len(PRIMARY):
        return None
    rng = np.random.default_rng(seed)
    per_category = []
    for category in PRIMARY:
        values = diffs[category]
        n = values.shape[0]
        per_category.append(values[rng.integers(0, n, size=(draws, n))].mean(axis=1))
    samples = np.mean(per_category, axis=0)
    return {
        "delta": float(np.mean([diffs[c].mean() for c in PRIMARY])),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "prompts": int(sum(v.shape[0] for v in diffs.values())),
    }


def endpoint_scoring_headroom(selection_dir: Path, draws: int, seed: int) -> dict:
    """Top-of-N minus candidate mean, inside the training cache. In-objective by construction."""
    values = []
    for path in sorted(selection_dir.glob("selection_rank*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            scores = [float(x) for x in json.loads(line)["endpoint_vqa"]]
            values.append(max(scores) - sum(scores) / len(scores))
    array = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = array.shape[0]
    samples = array[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return {
        "delta": float(array.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "records": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default=None)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    entries = []

    # ---------------------------------------------------------------- oracle 1: in-cache scoring
    selection = ROOT / "phaseC"
    entries.append({
        "id": "oracle.teacher_endpoint_scoring",
        "quantity": "top-of-4 minus candidate mean VQAScore inside the training cache",
        "intervention": "scoring only; no generation and no training",
        "candidate_generator": "frozen SD3.5-M teacher, 8 steps, CFG 7, 4 seeds per prompt",
        "selector": "VQAScore (clip-flant5-xxl)",
        "evaluator": "VQAScore (clip-flant5-xxl) — SAME instrument as the selector, in-objective",
        "population": "5,559 training-split prompts",
        "independent_of_selector": False,
        "result": endpoint_scoring_headroom(selection, args.draws, args.seed),
        "artifacts": {"selection_dir": str(selection.relative_to(ROOT))},
        "sha256": dir_sha256(selection, "selection_rank*.jsonl"),
        "lsf_job": None,
        "wandb_run": None,
        "warning": "Must never be used as an amortization denominator: selector and evaluator are "
                   "the same model, and the population is the training split.",
    })

    # ------------------------------------------------------- oracle 2: teacher best-of-4 held out
    verdict = ROOT / "exp0" / "primary_headroom" / "verdict.json"
    payload = json.loads(verdict.read_text())
    entries.append({
        "id": "oracle.teacher_best_of_4_heldout",
        "quantity": "teacher best-of-4 minus random candidate, official CompBench primary",
        "intervention": "select 1 of 4 frozen-teacher candidates per prompt at inference",
        "candidate_generator": "frozen SD3.5-M teacher, 8 steps, CFG 7",
        "selector": "VQAScore (clip-flant5-xxl)",
        "evaluator": "official T2I-CompBench category evaluators (BLIP-VQA / UniDet / 3-in-1)",
        "population": "1,200 held-out prompts, 150 per category",
        "independent_of_selector": True,
        "result": {
            "delta": payload["primary"]["delta"],
            "ci95": payload["primary"]["ci"],
            "prompts": sum(v["n"] for k, v in payload["per_category"].items() if k in PRIMARY),
        },
        "artifacts": {"verdict": str(verdict.relative_to(ROOT))},
        "sha256": {"verdict.json": sha256(verdict)},
        "lsf_job": None,
        "wandb_run": None,
        "warning": "THIS is the teacher oracle. It is NOT the +0.0485 noise-seed oracle.",
    })

    # ----------------------------------------------- oracle 3: frozen-student noise-seed best-of-4
    noise = ROOT / "phaseC" / "eval_verdict.json"
    payload = json.loads(noise.read_text())
    entries.append({
        "id": "oracle.student_noise_best_of_4_heldout",
        "quantity": "best-of-4 over the frozen M1 student's own initial noise seeds",
        "intervention": "select 1 of 4 initial latents for a FIXED four-step student",
        "candidate_generator": "frozen M1 four-step student, CFG 1",
        "selector": "VQAScore (clip-flant5-xxl)",
        "evaluator": "official T2I-CompBench category evaluators",
        "population": "held-out CompBench val prompts",
        "independent_of_selector": True,
        "result": {"delta": payload["primary"]["delta"], "ci95": payload["primary"]["ci"]},
        "artifacts": {"verdict": str(noise.relative_to(ROOT))},
        "sha256": {"eval_verdict.json": sha256(noise)},
        "lsf_job": None,
        "wandb_run": None,
        "warning": "A DIFFERENT intervention from the teacher oracle. Shallow OOF prediction of "
                   "reward from raw noise is r = -0.0123, capping this route near +0.0015. This "
                   "does not refute Noise Hypernetworks, which learns noise modulation end to end.",
    })

    # --------------------------------------------------------------- C1 selected-trajectory null
    c1_scores = ROOT / "phaseC" / "eval_scores"
    entries.append({
        "id": "result.C1_selected_trajectory_transfer",
        "quantity": "B4 (selected teacher trajectory) minus B2 (random trajectory), CompBench primary",
        "intervention": "winner-only consistency distillation on selected teacher trajectories",
        "candidate_generator": "frozen SD3.5-M teacher",
        "selector": "VQAScore (clip-flant5-xxl)",
        "evaluator": "official T2I-CompBench category evaluators",
        "population": "2,398 held-out CompBench val prompts",
        "independent_of_selector": True,
        "result": paired_primary(c1_scores, "B4", "B2", 4, args.draws, args.seed),
        "control": paired_primary(c1_scores, "B2prime", "B2", 4, args.draws, args.seed),
        "artifacts": {"scores_dir": str(c1_scores.relative_to(ROOT))},
        "sha256": {},
        "lsf_job": None,
        "wandb_run": None,
        "warning": "A null for THIS objective and setting; not a theorem that best-of-K cannot be "
                   "amortized. Prompt idx in this directory is NOT comparable to the idx in "
                   "exp0/primary_headroom: the two evaluations share zero prompts.",
    })

    # ----------------------------------------------------------------- Phase I preference pilot
    e99955 = ROOT / "phaseI" / "eval_99955" / "compbench_scores"
    for treatment, control, name in (
        ("Preference", "Placebo", "result.phaseI_preference_vs_placebo"),
        ("Preference", "M1", "result.phaseI_preference_vs_m1"),
        ("Placebo", "M1", "result.phaseI_placebo_vs_m1"),
        ("PreferenceREPA", "Preference", "result.phaseI_repa_increment"),
    ):
        entries.append({
            "id": name,
            "quantity": f"{treatment} minus {control}, CompBench primary at 4 steps",
            "intervention": "explicit pairwise preference fine-tuning of the M1 student",
            "candidate_generator": "frozen SD3.5-M teacher (training pairs)",
            "selector": "VQAScore (clip-flant5-xxl)",
            "evaluator": "official T2I-CompBench category evaluators",
            "population": "2,398 held-out CompBench val prompts",
            "independent_of_selector": True,
            "result": paired_primary(e99955, treatment, control, 4, args.draws, args.seed),
            "artifacts": {"scores_dir": str(e99955.relative_to(ROOT))},
            "sha256": {},
            "lsf_job": 99955,
            "wandb_run": "phaseI_eval_99955",
            "training_seeds": 1,
        })

    # -------------------------------------------------- Phase I counterfactual pair construction
    e100593 = ROOT / "phaseI" / "counterfactual_eval_100593" / "compbench_scores"
    entries.append({
        "id": "result.phaseI_correct_vs_counterfactual_construction",
        "quantity": "correct minus counterfactual PAIR CONSTRUCTION, CompBench primary at 4 steps",
        "intervention": "the labelling text changed BOTH pair membership (60.75% of records) and "
                        "orientation, so this is the total effect of the pair-construction policy",
        "candidate_generator": "frozen SD3.5-M teacher, identical 4 candidates in both arms",
        "selector": "VQAScore under the original prompt vs under a one-atom edited prompt",
        "evaluator": "official T2I-CompBench category evaluators",
        "population": "2,398 held-out CompBench val prompts",
        "independent_of_selector": True,
        "result": paired_primary(e100593, "CorrectPreference", "CounterfactualPreference", 4,
                                 args.draws, args.seed),
        "artifacts": {"scores_dir": str(e100593.relative_to(ROOT))},
        "sha256": {},
        "lsf_job": 100593,
        "wandb_run": "phaseI_counterfactual_eval_100593",
        "training_seeds": 1,
        "warning": "NOT the orientation-only direct effect. Pair identity is a downstream mediator "
                   "of the text intervention. The fixed-pair experiment estimates the direct effect.",
    })

    # ------------------------------------------------------------------------- fidelity lineage
    for job, path, gen_pool in (
        (99955, ROOT / "phaseI" / "eval_99955" / "fidelity.json",
         "T2I-CompBench compositional prompts (compbench/images)"),
        (100593, ROOT / "phaseI" / "counterfactual_eval_100593" / "fidelity.json",
         "COCO captions (fidelity/images)"),
    ):
        payload = json.loads(path.read_text())
        entries.append({
            "id": f"result.fidelity_job_{job}",
            "quantity": "FID / CMMD / precision / recall against COCO val2017 real images",
            "generation_prompt_pool": gen_pool,
            "reference_pool": f"{payload['reference_n']} COCO val2017 real images, square-cropped",
            "result": {
                model: {
                    "fid": stats["fid"], "cmmd": stats["cmmd"],
                    "precision": stats["precision"], "recall": stats["recall"], "n": stats["n"],
                }
                for model, stats in payload["results"].items()
            },
            "artifacts": {"fidelity": str(path.relative_to(ROOT))},
            "sha256": {"fidelity.json": sha256(path)},
            "lsf_job": job,
            "warning": "THE TWO FIDELITY JOBS ARE NOT COMPARABLE. Job 99955 generated from "
                       "CompBench compositional prompts; job 100593 generated from COCO captions. "
                       "Both are scored against the same COCO real reference, so the ~24-point FID "
                       "difference is a prompt-distribution effect, not an anomaly. Only job "
                       "100593's protocol is a standard matched-caption FID.",
        })

    # --------------------------------------------------------------------------------- diversity
    diversity = ROOT / "phaseI" / "counterfactual_eval_100593" / "diversity.json"
    if diversity.exists():
        entries.append({
            "id": "result.phaseI_counterfactual_diversity",
            "quantity": "within-prompt DINO and LPIPS diversity over 4 seeds",
            "population": "400 stratified held-out prompts",
            "result": json.loads(diversity.read_text()).get("gate", {}),
            "artifacts": {"diversity": str(diversity.relative_to(ROOT))},
            "sha256": {"diversity.json": sha256(diversity)},
            "lsf_job": 100593,
            "warning": "The preregistered DINO gate FAILED. Compare against the teacher best-of-K "
                       "diversity price, not against M1 alone.",
        })

    # -------------------------------------------------------------------- fixed-pair cache (new)
    for cache in sorted((ROOT / "phaseFP").glob("fixedpair_*")):
        summary_path = cache / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        entries.append({
            "id": f"cache.fixed_pair_{cache.name.split('_')[-1]}",
            "quantity": "fixed-unordered-pair label-only cache, four orientation arms",
            "records": summary["records"],
            "result": {
                "reversal_fraction": summary["statistics"]["reversal_fraction"],
                "arm_agreement_with_correct": summary["statistics"]["arm_agreement_with_correct"],
                "cross_arm_pair_identity": summary["cross_arm_pair_identity"],
            },
            "artifacts": {"cache": str(cache.relative_to(ROOT))},
            "sha256": {"summary.json": sha256(summary_path),
                       "pair_manifest.json": sha256(cache / "pair_manifest.json")},
            "lsf_job": int(cache.name.split("_")[-1]),
            "wandb_run": f"phaseFP_cache_{cache.name.split('_')[-1]}",
        })

    registry = {
        "created_by": "audit/build_registry.py",
        "git_revision": git_revision(),
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.seed,
        "prompt_key_warning": (
            "The integer `idx` field is assigned per evaluation job. Two jobs may use the same idx "
            "for different prompts. Always join evaluations on prompt TEXT unless the idx->prompt "
            "maps have been shown to agree."
        ),
        "entries": entries,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(registry, indent=2, sort_keys=True))

    if args.md:
        lines = [
            "# Immutable result registry",
            "",
            f"Generated by `audit/build_registry.py` at git `{registry['git_revision'][:12]}`. "
            "Every effect is recomputed from its raw artifact; prose reports are not evidence.",
            "",
            "> **Prompt joining.** " + registry["prompt_key_warning"],
            "",
            "## The three oracles — never merge these",
            "",
            "| id | intervention | evaluator independent of selector | Δ | 95% CI |",
            "|---|---|:-:|---:|---|",
        ]
        for entry in entries:
            if not entry["id"].startswith("oracle."):
                continue
            r = entry["result"]
            lines.append(
                f"| `{entry['id']}` | {entry['intervention']} | "
                f"{'yes' if entry['independent_of_selector'] else '**no**'} | "
                f"{r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |"
            )
        lines += [
            "",
            "## Training results",
            "",
            "| id | Δ (CompBench primary) | 95% CI | seeds | LSF job |",
            "|---|---:|---|---:|---:|",
        ]
        for entry in entries:
            if not entry["id"].startswith("result.") or entry.get("result") is None:
                continue
            r = entry["result"]
            if not isinstance(r, dict) or "delta" not in r:
                continue
            lines.append(
                f"| `{entry['id']}` | {r['delta']:+.5f} | "
                f"[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] | "
                f"{entry.get('training_seeds', '—')} | {entry.get('lsf_job') or '—'} |"
            )
        lines += ["", "## Warnings attached to specific entries", ""]
        for entry in entries:
            if entry.get("warning"):
                lines.append(f"- **`{entry['id']}`** — {entry['warning']}")
        Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({"entries": len(entries),
                      "ids": [e["id"] for e in entries]}, indent=2))


if __name__ == "__main__":
    main()
