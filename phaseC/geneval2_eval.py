#!/usr/bin/env python3
"""
GenEval2 driver — the pre-registration's SECOND primary (selector-independent) metric.

`GenEval2/evaluation.py` is run UNMODIFIED, so the score is the official one. Its
judge is Qwen3-VL-8B-Instruct: no clip-flant5 lineage, so like the T2I-CompBench
evaluators it is independent of the B4 selector. Official headline metric is
**Soft-TIFA** with geometric-mean pooling (`--method soft_tifa_gm`, per GenEval2's
README); `tifa` and `vqascore` are available for comparison.

  stage    our image tree -> the {prompt: image_filepath} json evaluation.py wants
  run      invoke GenEval2/evaluation.py in the venv that has transformers>=4.57
  collect  join its score lists back onto the pool -> per-prompt + per-skill scores

Two things this has to get right.

**Joining scores back to prompts.** evaluation.py writes a bare list of per-atom score
lists, one entry per line of the benchmark jsonl, with no prompt key. So the join is
positional and only valid if the pool is the benchmark file in file order — which is
what `build_eval_pool.py geneval2` guarantees and `collect` re-verifies against the
benchmark file before trusting the alignment.

**Environment.** transformers 4.49 in the shared env has no Qwen3-VL, and the SD3.5
training/eval stack is pinned against it. `--python` therefore defaults to a separate
venv (`cache/venv_geneval2`) that shadows transformers only for this evaluator; the
shared env is untouched.

Usage
  python phaseC/geneval2_eval.py stage   --images phaseC/geneval2/images/B4 --steps 4 \
      --pool phaseC/geneval2/prompts.json --out phaseC/geneval2_scores/B4_s4
  python phaseC/geneval2_eval.py run     --dir phaseC/geneval2_scores/B4_s4
  python phaseC/geneval2_eval.py collect --dir phaseC/geneval2_scores/B4_s4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENEVAL2 = REPO / "GenEval2"
DEFAULT_PYTHON = REPO / "cache" / "venv_geneval2" / "bin" / "python"

# how evaluation.py pools a prompt's per-atom scores into its per-prompt score
POOLING = {"soft_tifa_gm": "gmean", "soft_tifa_am": "mean", "tifa": "mean",
           "vqascore": "mean"}


def _gmean(xs):
    from scipy.stats import gmean
    return float(gmean(xs))


# ── stage ────────────────────────────────────────────────────────────────────

def cmd_stage(args):
    pool = json.loads(Path(args.pool).read_text())
    if args.limit:
        pool = pool[: args.limit]
    images_root = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image_data, missing, kept = {}, [], []
    for item in pool:
        src = images_root / f"p{item['idx']:05d}" / f"s{args.steps}" / args.image_name
        if not src.exists():
            missing.append(str(src))
            continue
        image_data[item["prompt"]] = str(src.resolve())
        kept.append(item)

    (out / "image_filepaths.json").write_text(json.dumps(image_data, indent=1))
    (out / "manifest.json").write_text(json.dumps({
        "images_root": str(images_root.resolve()), "steps": args.steps,
        "pool": str(Path(args.pool).resolve()), "benchmark": str(Path(args.benchmark).resolve()),
        "n": len(kept), "n_missing": len(missing), "manifest": kept}, indent=1))

    print(f"staged {len(kept)} prompt->image entries -> {out / 'image_filepaths.json'}")
    if missing:
        sys.exit(f"stage: {len(missing)} of {len(pool)} images missing (e.g. {missing[0]}); evaluation.py "
                 f"scores the whole benchmark, so a partial set cannot be joined back. Finish generation first.")


# ── run ──────────────────────────────────────────────────────────────────────

def cmd_run(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    out_file = d / f"scores_raw_{args.method}.json"
    if out_file.exists() and args.skip_done:
        print(f"skip, {out_file} already present")
        return

    python = Path(args.python)
    if not python.exists():
        sys.exit(f"{python} not found — GenEval2 needs transformers>=4.57 for Qwen3-VL; "
                 f"create it with:\n"
                 f"  python -m venv --system-site-packages cache/venv_geneval2\n"
                 f"  cache/venv_geneval2/bin/pip install 'transformers>=4.57' accelerate")

    argv = [str(python), "evaluation.py",
            "--benchmark_data", meta["benchmark"],
            "--image_filepath_data", str(d / "image_filepaths.json"),
            "--method", args.method,
            "--output_file", str(out_file)]
    print(f"[geneval2] cwd={GENEVAL2}\n[geneval2] $ {' '.join(argv)}", flush=True)

    env = dict(os.environ)
    env.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    rc = subprocess.call(argv, cwd=GENEVAL2, env=env)
    if rc != 0:
        sys.exit(f"GenEval2 evaluation.py exited {rc}")
    (d / "method.txt").write_text(args.method)
    print(f"[geneval2] done -> {out_file}")


# ── collect ──────────────────────────────────────────────────────────────────

def cmd_collect(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    method = args.method or (d / "method.txt").read_text().strip()
    raw_path = d / f"scores_raw_{method}.json"
    if not raw_path.exists():
        sys.exit(f"no result at {raw_path} — did `run` succeed?")
    raw = json.loads(raw_path.read_text())

    # evaluation.py iterates the BENCHMARK file, not our staged subset, and writes one
    # entry per line with no key. Verify that alignment before joining anything.
    bench = [json.loads(ln) for ln in
             Path(meta["benchmark"]).read_text().splitlines() if ln.strip()]
    if len(raw) != len(bench):
        sys.exit(f"{len(raw)} score lists vs {len(bench)} benchmark lines — cannot align")
    pool = {it["idx"]: it for it in meta["manifest"]}
    if len(pool) != len(bench):
        sys.exit(f"staged {len(pool)} prompts but the benchmark has {len(bench)} lines; "
                 f"evaluation.py scored the full benchmark, so a partial stage cannot be "
                 f"joined positionally. Generate all prompts, then re-stage.")
    for i, b in enumerate(bench):
        if pool[i]["prompt"] != b["prompt"]:
            sys.exit(f"pool idx {i} is {pool[i]['prompt']!r} but benchmark line {i} is "
                     f"{b['prompt']!r} — the pool is not the benchmark in file order")

    pooling = POOLING[method]
    rows, per_skill = [], {}
    for i, (b, sl) in enumerate(zip(bench, raw)):
        sl = [float(x) for x in sl]
        score = _gmean(sl) if pooling == "gmean" else sum(sl) / len(sl)
        rows.append({"idx": i, "category": "geneval2", "prompt": b["prompt"],
                     "score": score, "atom_scores": sl,
                     "skills": b.get("skills", []), "atom_count": b.get("atom_count")})
        # per-skill = per-ATOM scores grouped by that atom's skill (skills is aligned
        # with vqa_list, hence with the atom score list)
        for sk, v in zip(b.get("skills", []), sl):
            per_skill.setdefault(sk, []).append(v)

    scores = [r["score"] for r in rows]
    summary = {
        "dir": str(d), "evaluator": f"geneval2_{method}", "method": method,
        "pooling": pooling, "steps": meta["steps"], "images_root": meta["images_root"],
        "n": len(rows),
        # GenEval2 reports on a 0-100 scale; per-prompt scores stay in [0,1] so they
        # are directly comparable with the CompBench evaluators in the paired analysis
        "mean": sum(scores) / len(scores),
        "official_score_x100": 100 * sum(scores) / len(scores),
        "per_skill": {k: {"n_atoms": len(v), "mean": sum(v) / len(v)}
                      for k, v in sorted(per_skill.items())},
        "per_prompt": rows,
    }
    (d / "scores.json").write_text(json.dumps(summary, indent=1))

    print(f"geneval2 [{method}] @ {meta['steps']} steps | {Path(meta['images_root']).name}")
    print(f"  n={len(rows)}  mean={summary['mean']:.4f}  "
          f"official={summary['official_score_x100']:.2f}")
    for k, v in summary["per_skill"].items():
        print(f"    {k:12s} atoms={v['n_atoms']:5d}  mean={v['mean']:.4f}")
    print(f"  -> {d / 'scores.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    bench_default = str(GENEVAL2 / "geneval2_data.jsonl")

    s = sub.add_parser("stage")
    s.add_argument("--images", required=True, help="e.g. phaseC/geneval2/images/B4")
    s.add_argument("--pool", default="phaseC/geneval2/prompts.json")
    s.add_argument("--benchmark", default=bench_default)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--image_name", default="cand0.png")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_stage)

    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--method", default="soft_tifa_gm", choices=sorted(POOLING))
    r.add_argument("--python", default=str(DEFAULT_PYTHON))
    r.add_argument("--skip_done", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("collect")
    c.add_argument("--dir", required=True)
    c.add_argument("--method", default=None, choices=sorted(POOLING))
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
