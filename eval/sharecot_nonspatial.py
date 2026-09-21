#!/usr/bin/env python3
"""Share-CoT (ShareGPT4V-7B + chain-of-thought) scoring for the CompBench++ **non-spatial** category,
wrapped so the numbers are the OFFICIAL ones. See docs/sharecot.md.

Why this exists
---------------
T2I-CompBench++ TABLE XIII (and every paper that copies its baseline rows, e.g. CTCal Table 1)
reports the Non-Spatial column with **Share-CoT**, not CLIPScore. eval/compbench.py maps
`non_spatial -> clipscore`, whose values (~0.31) are on a different scale and barely separate models
at all. This script adds the Share-CoT column without touching that driver.

Design: the same three phases as eval/compbench.py, the scoring itself delegated to the vendored,
UNMODIFIED official script:

  stage    our image tree -> `<out>/samples/<prompt>_<qid:06d>.png`  (symlinks)
  run      `third_party/T2I-CompBench/MLLM_eval/ShareGPT4V-CoT_eval/Share_eval.py
            --category action --cot --file-path <out>` from its own cwd
  collect  `<out>/sharegpt4v/vqa_result.json` -> `<out>/scores.json`, in the same schema as every
           other `compbench_scores/*/scores.json` here, with `"evaluator": "sharegpt4v_cot"`.

Category name: the official script has NO `non_spatial` branch. CompBench's non-spatial prompts are
action-relation prompts (`examples/dataset/non_spatial_val.txt`) and both MLLM evaluators
(`Share_eval.py`, `gpt4v_eval.py`) call that category **`action`**. Passing `--category non_spatial`
would fall through every branch and die with a NameError on `query`. So `action` is the official
setting here.

Scale: the official aggregation maps the model's 1-5 verdict through {1:20, 2:40, 3:60, 4:80, 5:100}
and averages, i.e. its `score.txt` avg is ~75. Published tables divide by 100 (SD1.4 0.7487, SD2
0.7567, SD3 0.7782), so `scores.json` stores score/100 and keeps the raw 20..100 value as `score_raw`.

Official quirk we deliberately reproduce: when the model's answer is not parseable JSON the upstream
aggregator `continue`s with `score_i` still at its initial 100, i.e. **an unparseable answer scores
1.0**. That is upstream's behaviour and every published number contains it. `collect` re-derives the
scores from the raw answers with byte-identical logic, asserts they equal upstream's
`vqa_result.json`, and reports `n_parse_failed` / `n_no_answer` as diagnostics so the contamination
is visible rather than silent.

The output directory `<label>_s<steps>_non_spatial_sharecot` sits next to the CLIPScore one and its
rows carry category "non_spatial": every consumer that globs `compbench_scores/*/scores.json` must
skip `evaluator == "sharegpt4v_cot"` (eval/compare_arms.py, eval/log_to_wandb.py and
paper/verify_numbers.py do; scripts/eval_alignment.lsf only averages the eight official columns).

Usage (inside the Share-CoT environment of docs/sharecot.md, on a GPU node; scripts/sharecot_score.lsf)
  python eval/sharecot_nonspatial.py --evaldir out/eval/eval_<label> --label <label> --images cand0.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPBENCH = Path(os.environ.get("COMPBENCH_DIR", ROOT / "third_party" / "T2I-CompBench"))
SHARE_DIR = COMPBENCH / "MLLM_eval" / "ShareGPT4V-CoT_eval"
SHARE_EVAL = SHARE_DIR / "Share_eval.py"
SEEDED_RUNNER = ROOT / "eval" / "compat" / "seeded_share_eval.py"
# cwd for the official script. NOT SHARE_DIR: `build_vision_tower` only accepts a vision tower that
# `os.path.exists()` or starts with "openai"/"laion", and the ShareGPT4V-7B config names the HF id
# `Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12`, whose weights are NOT in the 7B checkpoint
# (verified: 0 `vision_tower` keys in its index). Upstream's Readme therefore git-clones the tower
# into `Lin-Chen/` next to the script. We build that layout here instead so the vendored repo's tree
# stays clean.
RUNROOT = Path(os.environ.get("SHARECOT_RUNROOT", ROOT / "cache" / "sharecot_runroot"))

# The official 1-5 -> percentage map for the non-attribute categories
# (Share_eval.py, the `else` branch of the `map = {...}` assignment).
OFFICIAL_MAP = {"1": 20, "2": 40, "3": 60, "4": 80, "5": 100}


def _safe_for_filename(prompt: str) -> bool:
    # Share_eval.py reads the prompt as name.split('_')[0] and the id as name.split('_')[-1], so
    # exactly one underscore may appear in the name.
    return "_" not in prompt and "/" not in prompt


def _infer_steps(images_root: Path) -> int:
    steps = set()
    for p in sorted(images_root.glob("p*/s*"))[:200]:
        m = re.fullmatch(r"s(\d+)", p.name)
        if m:
            steps.add(int(m.group(1)))
    if len(steps) != 1:
        sys.exit(f"could not infer --steps from {images_root} (found {sorted(steps)}); pass --steps")
    return steps.pop()


# ── stage ────────────────────────────────────────────────────────────────────

def stage(out: Path, images_root: Path, prompts: list, category: str, steps: int,
          names: list, limit: int) -> dict:
    prompts = [p for p in prompts if p["category"] == category]
    if limit:
        prompts = prompts[:limit]
    if not prompts:
        sys.exit(f"no prompts of category {category!r}")

    samples = out / "samples"
    if samples.exists():
        shutil.rmtree(samples)
    samples.mkdir(parents=True)

    manifest, missing, unsafe = [], [], []
    for item in prompts:
        prompt, idx = item["prompt"], item["idx"]
        if not _safe_for_filename(prompt):
            unsafe.append(prompt)
            continue
        srcs = [images_root / f"p{idx:05d}" / f"s{steps}" / n for n in names]
        absent = [str(s) for s in srcs if not s.exists()]
        if absent:
            missing += absent
            continue
        for name, src in zip(names, srcs):
            qid = len(manifest)
            dst = samples / f"{prompt}_{qid:06d}.png"
            dst.symlink_to(src.resolve())
            manifest.append({
                "question_id": qid, "idx": idx, "category": item["category"],
                "prompt": prompt, "image": name, "src": str(src.resolve()),
                "staged": dst.name,
            })
    if unsafe:
        print(f"  SKIPPED {len(unsafe)} prompts unusable as filenames: {unsafe[:3]}")
    if missing:
        sys.exit(f"stage: {len(missing)} images missing, e.g. {missing[0]}; "
                 f"a partial benchmark is not the benchmark")
    meta = {"images_root": str(images_root.resolve()), "steps": steps,
            "categories": [category], "n": len(manifest), "manifest": manifest}
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"staged {len(manifest)} images -> {samples}")
    return meta


# ── run ──────────────────────────────────────────────────────────────────────

def _snapshot(repo_id: str) -> Path:
    """Local snapshot dir of an already-downloaded HF repo (no network)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, local_files_only=True))


def ensure_runroot(model_path: str) -> Path:
    """Materialise `<RUNROOT>/<mm_vision_tower>` -> the cached ViT snapshot."""
    RUNROOT.mkdir(parents=True, exist_ok=True)
    cfg_path = (Path(model_path) / "config.json") if Path(model_path).is_dir() \
        else (_snapshot(model_path) / "config.json")
    tower = json.loads(cfg_path.read_text()).get("mm_vision_tower", "")
    if not tower or tower.startswith(("openai", "laion")) or Path(tower).is_absolute():
        return RUNROOT                       # upstream's builder accepts these as-is
    link = RUNROOT / tower
    link.parent.mkdir(parents=True, exist_ok=True)
    target = _snapshot(tower).resolve()
    if link.is_symlink() and link.resolve() == target:
        return RUNROOT
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
    print(f"[share-cot] vision tower {tower} -> {target}")
    return RUNROOT


def run(out: Path, args) -> None:
    if not SHARE_EVAL.exists():
        sys.exit(f"official script not found at {SHARE_EVAL} (clone T2I-CompBench into third_party/ or set COMPBENCH_DIR)")
    runroot = ensure_runroot(args.model_path)
    argv = [sys.executable, str(SEEDED_RUNNER),
            "--category", args.official_category,
            "--file-path", str(out),
            "--model-path", args.model_path,
            "--model-name", args.model_name,
            "--cot"]
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
    env.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / "cache" / "xdg"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "xdg" / "matplotlib"))
    env["PYTHONNOUSERSITE"] = "1"          # a user-site python would shadow the env otherwise
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["SHARECOT_SEED"] = str(args.seed)
    env["SHARECOT_SCRIPT"] = str(SHARE_EVAL)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(f"[share-cot] cwd={runroot}\n[share-cot] $ {' '.join(argv)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(argv, cwd=runroot, env=env)
    dt = time.time() - t0
    if rc != 0:
        sys.exit(f"Share_eval.py exited {rc}")
    (out / "runtime.json").write_text(json.dumps({"run_seconds": dt}, indent=1))
    print(f"[share-cot] done in {dt/60:.1f} min")


# ── collect ──────────────────────────────────────────────────────────────────

def _official_score(answers: dict) -> tuple[float, bool]:
    """Upstream's aggregation for one image, byte-identical to Share_eval.py.

    Returns (score in 20..100, parsed_ok). `parsed_ok` is False exactly when the initial 100 survives
    because nothing parsed -- upstream cannot tell that apart from an honest verdict of 5, which is
    why we track it separately.
    """
    score_i = 100
    parsed = False
    for _id, ans in answers.items():
        if ans[-1:] != "}":                      # upstream: ans[-1] (IndexError on "")
            ans = ans + "\"\n}"
        try:
            js = json.loads(ans.replace("\n", ""))
            level = js["score"]
            score_i *= OFFICIAL_MAP[str(level)] / 100
            parsed = True
        except Exception:
            continue
    return score_i, parsed


def collect(out: Path, meta: dict, args) -> dict:
    folder = out / args.folder_name
    raw_path = folder / "vqa_result.json"
    score_path = folder / "score.json"
    total_path = folder / "total.json"
    if not raw_path.exists():
        sys.exit(f"no result at {raw_path} -- did `run` succeed?")

    official = {int(r["question_id"]): float(r["answer"]) for r in json.loads(raw_path.read_text())}
    answers = json.loads(score_path.read_text()) if score_path.exists() else {}
    details = json.loads(total_path.read_text()) if total_path.exists() else {}

    # Re-derive with identical logic and refuse to report if we disagree with upstream.
    n_parse_failed = n_no_answer = 0
    replicated = {}
    for k, v in answers.items():
        s, parsed = _official_score(v)
        replicated[int(k)] = s
        if not v:
            n_no_answer += 1
        elif not parsed:
            n_parse_failed += 1
    mismatch = [q for q in official if q in replicated and abs(official[q] - replicated[q]) > 1e-9]
    if mismatch:
        sys.exit(f"replication of the official parse disagrees on {len(mismatch)} images "
                 f"(e.g. qid {mismatch[0]}): refusing to report")

    rows = []
    for m in meta["manifest"]:
        q = m["question_id"]
        if q not in official:
            continue
        det = details.get(str(q), {})
        rows.append({**m,
                     "score": official[q] / 100.0,
                     "score_raw": official[q],
                     "answer": det.get("ans1", ""),
                     "description": det.get("description", "")})
    if not rows:
        sys.exit("no scores matched the manifest")
    if len(rows) != len(meta["manifest"]):
        print(f"  WARNING only {len(rows)}/{len(meta['manifest'])} staged images scored")

    per_image = rows
    by_idx: dict = {}
    for r in rows:
        by_idx.setdefault(r["idx"], []).append(r)
    per_prompt = []
    for _i, rs in sorted(by_idx.items()):
        row = {k: rs[0][k] for k in ("question_id", "idx", "category", "prompt", "src")}
        row.update({"score": sum(r["score"] for r in rs) / len(rs), "n_images": len(rs),
                    "image_scores": [r["score"] for r in rs]})
        per_prompt.append(row)

    scores = [r["score"] for r in per_prompt]
    per_cat: dict = {}
    for r in per_prompt:
        per_cat.setdefault(r["category"], []).append(r["score"])
    runtime = json.loads((out / "runtime.json").read_text())["run_seconds"] \
        if (out / "runtime.json").exists() else None

    summary = {
        "dir": str(out), "evaluator": "sharegpt4v_cot", "steps": meta["steps"],
        "images_root": meta["images_root"], "n": len(per_prompt),
        "mean": sum(scores) / len(scores),
        "per_category": {c: {"n": len(v), "mean": sum(v) / len(v)} for c, v in sorted(per_cat.items())},
        "per_prompt": per_prompt,
        "images_per_prompt": sorted({len(v) for v in by_idx.values()}),
        "per_image": per_image,
        # ── Share-CoT specific provenance ──
        "official_script": str(SHARE_EVAL),
        "official_category": args.official_category,
        "cot": True,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "seed": args.seed,
        "temperature": 0.2, "top_p": 0.7,
        "score_scale": "official 20..100 divided by 100 (published tables' scale)",
        "mean_raw": sum(r["score_raw"] for r in per_image) / len(per_image),
        "n_parse_failed": n_parse_failed,
        "n_no_answer": n_no_answer,
        "run_seconds": runtime,
        "seconds_per_image": (runtime / len(per_image)) if runtime else None,
    }
    (out / "scores.json").write_text(json.dumps(summary, indent=1))

    print(f"sharegpt4v_cot @ {meta['steps']} steps | {Path(meta['images_root']).name}")
    print(f"  n={len(per_prompt)}  mean={summary['mean']:.4f}  (raw {summary['mean_raw']:.2f}/100)")
    print(f"  unparseable answers scored 1.0 by upstream: {n_parse_failed}"
          f"   images with no answer at all: {n_no_answer}")
    if runtime:
        print(f"  {runtime/60:.1f} min for {len(per_image)} images "
              f"= {runtime/len(per_image):.1f} s/image")
    print(f"  -> {out / 'scores.json'}")
    return summary


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaldir", required=True, help="eval root, e.g. out/eval/eval_<label> (scripts/eval_alignment.lsf)")
    ap.add_argument("--label", required=True, help="model label = the dir under compbench/images/")
    ap.add_argument("--prompts", default="pools/eval/compbench_prompts.json",
                    help="the prompt pool the images were generated from (its idx numbering MUST match the image tree)")
    ap.add_argument("--images", default="cand0.png", help="comma-separated image names per prompt")
    ap.add_argument("--steps", type=int, default=0, help="0 = infer from the image tree")
    ap.add_argument("--category", default="non_spatial", help="our category name (prompts json)")
    ap.add_argument("--official_category", default="action",
                    help="Share_eval.py's name for it; non_spatial == action upstream")
    ap.add_argument("--folder_name", default="sharegpt4v",
                    help="Share_eval.py's output folder (its --folder-name)")
    ap.add_argument("--model_path", default="Lin-Chen/ShareGPT4V-7B")
    ap.add_argument("--model_name", default="llava-v1.5-7b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="override the output dir")
    ap.add_argument("--skip_run", action="store_true",
                    help="reuse an existing vqa_result.json (re-collect only)")
    args = ap.parse_args()

    evaldir = Path(args.evaldir).resolve()
    images_root = evaldir / "compbench" / "images" / args.label
    if not images_root.is_dir():
        sys.exit(f"no images at {images_root}")
    steps = args.steps or _infer_steps(images_root)
    out = Path(args.out).resolve() if args.out else (
        evaldir / "compbench_scores" / f"{args.label}_s{steps}_{args.category}_sharecot")
    out.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(Path(args.prompts).read_text())
    names = [n for n in args.images.split(",") if n]

    if args.skip_run and (out / "manifest.json").exists():
        meta = json.loads((out / "manifest.json").read_text())
        print(f"reusing {out/'manifest.json'} ({meta['n']} staged)")
    else:
        meta = stage(out, images_root, prompts, args.category, steps, names, args.limit)
        run(out, args)
    collect(out, meta, args)


if __name__ == "__main__":
    main()
