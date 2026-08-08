#!/usr/bin/env python3
"""Build the fixed-unordered-pair label-only causal cache.

The completed Phase-I counterfactual campaign let each labelling text pick its own top and bottom
candidate, so the unordered pair changed for 60.75% of records. That comparison therefore estimates
the total effect of the text-conditioned pair-construction policy, not the orientation-only direct
effect.

This builder freezes the pair once, using the ORIGINAL-prompt scores:

    a_i = argmax_j R(x_ij, c_i)      b_i = argmin_j R(x_ij, c_i)

and emits four arms that differ only in the orientation of that fixed pair. Every arm shares the
same idx, prompt, seed_base, candidate indices and pair_key, so training can assert bit-identical
pair identity and never recompute an extremum.

CPU only. The source cache phaseI/counterfactual_data_100590 already stores both score vectors over
the same four candidates, so no teacher regeneration is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARMS = ("correct", "counterfactual", "random", "inverted")

# Provenance fields copied verbatim from the source cache. Score vectors are renamed explicitly so
# that no downstream reader can mistake a counterfactual score for the selection score.
CARRIED_FIELDS = (
    "category",
    "prompt",
    "seed_base",
    "N",
    "edit_family",
    "edit_source",
    "edit_target",
    "edit_start",
    "edit_end",
)


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


def load_jsonl_dir(directory: Path) -> list[dict]:
    records: list[dict] = []
    paths = sorted(directory.glob("selection_rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no selection_rank*.jsonl under {directory}")
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    records.sort(key=lambda item: int(item["idx"]))
    return records


def pair_key(idx: int, seed_base: int, a: int, b: int) -> str:
    """Identity of the unordered pair. Deliberately excludes orientation."""
    lo, hi = sorted((int(a), int(b)))
    payload = f"fixedpair:v1:{int(idx)}:{int(seed_base)}:{lo}:{hi}".encode()
    return hashlib.sha256(payload).hexdigest()


def balanced_random_signs(indices: list[int], label_seed: int) -> dict[int, int]:
    """Deterministic and EXACTLY balanced +1/-1 assignment.

    A per-record hash coin is deterministic but only approximately balanced; with 5,042 records the
    residual imbalance is a real confound for a placebo arm. Ranking by hash and splitting at the
    median makes the arm exactly balanced while staying a pure function of (idx, label_seed).
    """
    ranked = sorted(
        indices,
        key=lambda idx: hashlib.blake2b(
            f"fixedpair:random:{label_seed}:{idx}".encode(), digest_size=16
        ).digest(),
    )
    half = len(ranked) // 2
    signs = {idx: +1 for idx in ranked[:half]}
    signs.update({idx: -1 for idx in ranked[half:]})
    return signs


def build_records(
    source: list[dict], label_seed: int, tie_policy: str
) -> tuple[dict[str, list[dict]], dict]:
    indices = [int(r["idx"]) for r in source]
    if len(indices) != len(set(indices)):
        raise ValueError("source cache contains duplicate prompt indices")
    random_signs = balanced_random_signs(indices, label_seed)

    arms: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    diagnostics = {
        "reversals": 0,
        "counterfactual_ties": 0,
        "original_margins": [],
        "counterfactual_margins_on_fixed_pair": [],
        "reversal_by_category": Counter(),
        "reversal_by_edit_family": Counter(),
        "records_by_category": Counter(),
        "records_by_edit_family": Counter(),
        "arm_positive_is_a": Counter(),
        "arm_agrees_with_correct": Counter(),
    }

    for record in source:
        idx = int(record["idx"])
        seed_base = int(record["seed_base"])
        original = [float(x) for x in record["original_endpoint_vqa"]]
        counterfactual = [float(x) for x in record["endpoint_vqa"]]
        if len(original) != len(counterfactual):
            raise ValueError(f"record {idx}: score vectors have different lengths")
        n = len(original)
        if n < 2:
            raise ValueError(f"record {idx}: fewer than two candidates")

        a = max(range(n), key=original.__getitem__)
        b = min(range(n), key=original.__getitem__)
        if a == b:
            raise ValueError(f"record {idx}: degenerate pair, all original scores equal")

        original_margin = original[a] - original[b]
        counterfactual_margin = counterfactual[a] - counterfactual[b]
        if counterfactual_margin == 0.0:
            diagnostics["counterfactual_ties"] += 1
            counterfactual_sign = +1 if tie_policy == "keep" else -1
        else:
            counterfactual_sign = +1 if counterfactual_margin > 0 else -1
        reverses = counterfactual_sign < 0

        category = str(record.get("category", "unknown"))
        edit_family = str(record.get("edit_family", "unknown"))
        diagnostics["records_by_category"][category] += 1
        diagnostics["records_by_edit_family"][edit_family] += 1
        if reverses:
            diagnostics["reversals"] += 1
            diagnostics["reversal_by_category"][category] += 1
            diagnostics["reversal_by_edit_family"][edit_family] += 1
        diagnostics["original_margins"].append(original_margin)
        diagnostics["counterfactual_margins_on_fixed_pair"].append(counterfactual_margin)

        key = pair_key(idx, seed_base, a, b)
        signs = {
            "correct": +1,
            "counterfactual": counterfactual_sign,
            "random": random_signs[idx],
            "inverted": -1,
        }
        label_prompts = {
            "correct": record["prompt"],
            "counterfactual": record.get("label_prompt", record["prompt"]),
            "random": record["prompt"],
            "inverted": record["prompt"],
        }

        for arm in ARMS:
            sign = signs[arm]
            positive, negative = (a, b) if sign > 0 else (b, a)
            diagnostics["arm_positive_is_a"][arm] += int(positive == a)
            diagnostics["arm_agrees_with_correct"][arm] += int(sign == signs["correct"])
            out = {
                "idx": idx,
                **{field: record[field] for field in CARRIED_FIELDS if field in record},
                # Selection scores under the original prompt. This is the only vector that
                # determined pair membership.
                "original_endpoint_vqa": original,
                "counterfactual_endpoint_vqa": counterfactual,
                # `endpoint_vqa` is retained under its legacy name for schema compatibility with
                # load_selection(), but the trainer must use the explicit indices below.
                "endpoint_vqa": original,
                "pair_a": a,
                "pair_b": b,
                "pair_key": key,
                "positive_idx": positive,
                "negative_idx": negative,
                "orientation": sign,
                "arm": arm,
                "original_pair_margin": original_margin,
                "counterfactual_pair_margin": counterfactual_margin,
                "pair_reverses_under_counterfactual": reverses,
                "label_source": {
                    "correct": "correct_prompt",
                    "counterfactual": "counterfactual_prompt",
                    "random": "randomized",
                    "inverted": "inverted_prompt",
                }[arm],
                "label_prompt": label_prompts[arm],
            }
            arms[arm].append(out)

    return arms, diagnostics


def write_arm(records: list[dict], directory: Path, shards: int) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    buckets: list[list[dict]] = [[] for _ in range(shards)]
    for position, record in enumerate(records):
        buckets[position % shards].append(record)
    hashes = {}
    for shard, bucket in enumerate(buckets):
        path = directory / f"selection_rank{shard}.jsonl"
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in bucket))
        hashes[path.name] = file_sha256(path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument(
        "--source_root",
        default="phaseI/counterfactual_data_100590",
        help="Completed counterfactual cache holding both score vectors on identical candidates.",
    )
    parser.add_argument("--label_seed", type=int, default=20260806)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--expected_records", type=int, default=5042)
    parser.add_argument("--min_reversal_fraction", type=float, default=0.20)
    parser.add_argument("--min_original_margin", type=float, default=0.20)
    parser.add_argument("--tie_policy", choices=["keep", "flip"], default="keep")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="sd-pref-repa-fixedpair-v1")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite existing cache root {out_root}")
    source_root = Path(args.source_root).resolve()

    correct_source = load_jsonl_dir(source_root / "correct")
    counterfactual_source = load_jsonl_dir(source_root / "counterfactual")
    if len(correct_source) != len(counterfactual_source):
        raise SystemExit("correct and counterfactual shards disagree on record count")

    # The counterfactual shard carries original_endpoint_vqa; assert it is bit-identical to the
    # correct shard's selection scores before trusting it as the pair-defining vector.
    correct_by_idx = {int(r["idx"]): r for r in correct_source}
    for record in counterfactual_source:
        idx = int(record["idx"])
        peer = correct_by_idx[idx]
        if [float(x) for x in record["original_endpoint_vqa"]] != [
            float(x) for x in peer["endpoint_vqa"]
        ]:
            raise SystemExit(f"record {idx}: original_endpoint_vqa disagrees with the correct shard")
        if int(record["seed_base"]) != int(peer["seed_base"]) or record["prompt"] != peer["prompt"]:
            raise SystemExit(f"record {idx}: prompt/seed provenance disagrees across shards")

    arms, diagnostics = build_records(counterfactual_source, args.label_seed, args.tie_policy)

    total = len(counterfactual_source)
    reversal_fraction = diagnostics["reversals"] / total
    mean_original_margin = statistics.mean(diagnostics["original_margins"])
    random_positive_fraction = diagnostics["arm_positive_is_a"]["random"] / total
    inverted_agreement = diagnostics["arm_agrees_with_correct"]["inverted"] / total
    counterfactual_agreement = diagnostics["arm_agrees_with_correct"]["counterfactual"] / total

    gates = {
        "record_count": total == args.expected_records,
        "reversal_fraction": reversal_fraction >= args.min_reversal_fraction,
        "mean_original_margin": mean_original_margin >= args.min_original_margin,
        "random_arm_balanced": abs(random_positive_fraction - 0.5) <= 0.5 / total + 1e-9,
        "inverted_arm_fully_opposed": inverted_agreement == 0.0,
        "counterfactual_arm_partially_disagrees": 0.0 < (1.0 - counterfactual_agreement) < 1.0,
    }
    passed = all(gates.values())

    file_hashes = {}
    if passed:
        out_root.mkdir(parents=True, exist_ok=False)
        for arm in ARMS:
            file_hashes[arm] = write_arm(arms[arm], out_root / arm, args.shards)

    # Cross-arm identity: the unordered pair must be byte-identical everywhere.
    keys_by_arm = {arm: {r["idx"]: r["pair_key"] for r in arms[arm]} for arm in ARMS}
    reference_keys = keys_by_arm["correct"]
    identity_ok = all(keys_by_arm[arm] == reference_keys for arm in ARMS)
    if not identity_ok:
        raise SystemExit("cross-arm pair_key identity assertion failed")

    summary = {
        "created_by": "phaseFP/build_fixed_pair_selection.py",
        "git_revision": git_revision(),
        "code_sha256": {
            "phaseFP/build_fixed_pair_selection.py": file_sha256(Path(__file__)),
        },
        "source_root": str(source_root),
        "source_sha256": {
            f"{arm}/{path.name}": file_sha256(path)
            for arm in ("correct", "counterfactual")
            for path in sorted((source_root / arm).glob("selection_rank*.jsonl"))
        },
        "configuration": vars(args),
        "records": total,
        "arms": list(ARMS),
        "candidate_count": int(counterfactual_source[0]["N"]),
        "cross_arm_pair_identity": identity_ok,
        "pass": passed,
        "gates": gates,
        "statistics": {
            "reversals": diagnostics["reversals"],
            "reversal_fraction": reversal_fraction,
            "counterfactual_ties": diagnostics["counterfactual_ties"],
            "tie_policy": args.tie_policy,
            "original_pair_margin": {
                "mean": mean_original_margin,
                "median": statistics.median(diagnostics["original_margins"]),
                "min": min(diagnostics["original_margins"]),
                "max": max(diagnostics["original_margins"]),
            },
            "counterfactual_pair_margin": {
                "mean": statistics.mean(diagnostics["counterfactual_margins_on_fixed_pair"]),
                "median": statistics.median(
                    diagnostics["counterfactual_margins_on_fixed_pair"]
                ),
            },
            "random_arm_positive_is_a_fraction": random_positive_fraction,
            "arm_agreement_with_correct": {
                arm: diagnostics["arm_agrees_with_correct"][arm] / total for arm in ARMS
            },
            "records_by_category": dict(diagnostics["records_by_category"]),
            "records_by_edit_family": dict(diagnostics["records_by_edit_family"]),
            "reversal_rate_by_category": {
                category: diagnostics["reversal_by_category"][category] / count
                for category, count in diagnostics["records_by_category"].items()
            },
            "reversal_rate_by_edit_family": {
                family: diagnostics["reversal_by_edit_family"][family] / count
                for family, count in diagnostics["records_by_edit_family"].items()
            },
        },
        "output_sha256": file_hashes,
    }

    if passed:
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        reversal_idx = sorted(
            int(r["idx"]) for r in arms["correct"] if r["pair_reverses_under_counterfactual"]
        )
        (out_root / "reversal_subset.json").write_text(
            json.dumps(
                {
                    "count": len(reversal_idx),
                    "fraction": reversal_fraction,
                    "idx": reversal_idx,
                },
                indent=2,
            )
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))

    if not args.no_wandb:
        import os

        os.environ.setdefault("WANDB_MODE", "online")
        import wandb

        run_name = args.wandb_run_name or out_root.name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=run_name,
            group=args.wandb_group or run_name,
            job_type="build-fixed-pair-cache",
            config=summary["configuration"],
            resume="never",
        )
        wandb.log({f"cache/{k}": v for k, v in summary["statistics"].items() if isinstance(v, (int, float))})
        for key, value in gates.items():
            wandb.run.summary[f"gate/{key}"] = value
        wandb.run.summary["pass"] = passed
        wandb.run.summary["output_root"] = str(out_root)
        wandb.finish()

    if not passed:
        raise SystemExit(f"fixed-pair cache gates failed: {gates}")


if __name__ == "__main__":
    main()
