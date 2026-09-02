#!/usr/bin/env python3
"""GenEval2 scoring with the official evaluator (third_party/GenEval2/evaluation.py, unmodified).

  stage    our image tree -> the {prompt: image_path} json evaluation.py wants
  run      invoke evaluation.py (Qwen3-VL judge; needs transformers >= 4.57, so a separate
           interpreter can be given with --python)
  collect  join its score lists back onto the pool -> per-prompt and per-skill scores

evaluation.py writes one score list per line of the benchmark file with no key, so the join is
positional: the pool must be the benchmark file in file order (data/build_eval_pool.py guarantees
it and `collect` re-verifies before joining). Headline metric is Soft-TIFA with geometric-mean
pooling (`--method soft_tifa_gm`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENEVAL2 = Path(os.environ.get("GENEVAL2_DIR", ROOT / "third_party" / "GenEval2"))
POOLING = {"soft_tifa_gm": "gmean", "soft_tifa_am": "mean", "tifa": "mean", "vqascore": "mean"}


def cmd_stage(args):
    pool = json.loads(Path(args.pool).read_text())
    images_root, out = Path(args.images), Path(args.out)
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
        "benchmark": str(Path(args.benchmark).resolve()), "n": len(kept),
        "n_missing": len(missing), "manifest": kept}, indent=1))
    print(f"staged {len(kept)} prompts -> {out / 'image_filepaths.json'}")
    if missing:
        print(f"  MISSING {len(missing)} images; evaluation.py needs every benchmark prompt")


def cmd_run(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    out_file = d / f"scores_raw_{args.method}.json"
    if out_file.exists() and args.skip_done:
        print(f"skip, {out_file} already present")
        return
    argv = [args.python, "evaluation.py", "--benchmark_data", meta["benchmark"],
            "--image_filepath_data", str(d / "image_filepaths.json"),
            "--method", args.method, "--output_file", str(out_file)]
    print(f"[geneval2] cwd={GENEVAL2}\n[geneval2] $ {' '.join(argv)}", flush=True)
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    rc = subprocess.call(argv, cwd=GENEVAL2, env=env)
    if rc != 0:
        sys.exit(f"GenEval2 evaluation.py exited {rc}")
    (d / "method.txt").write_text(args.method)


def cmd_collect(args):
    from scipy.stats import gmean
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    method = args.method or (d / "method.txt").read_text().strip()
    raw = json.loads((d / f"scores_raw_{method}.json").read_text())
    bench = [json.loads(ln) for ln in Path(meta["benchmark"]).read_text().splitlines() if ln.strip()]
    pool = {it["idx"]: it for it in meta["manifest"]}
    if len(raw) != len(bench) or len(pool) != len(bench):
        sys.exit(f"{len(raw)} score lists, {len(pool)} staged, {len(bench)} benchmark lines: cannot align")
    for i, b in enumerate(bench):
        if pool[i]["prompt"] != b["prompt"]:
            sys.exit(f"pool idx {i} is not benchmark line {i}; the pool must be the benchmark in file order")
    rows, per_skill = [], {}
    for i, (b, sl) in enumerate(zip(bench, raw)):
        sl = [float(x) for x in sl]
        score = float(gmean(sl)) if POOLING[method] == "gmean" else sum(sl) / len(sl)
        rows.append({"idx": i, "category": "geneval2", "prompt": b["prompt"], "score": score,
                     "atom_scores": sl, "skills": b.get("skills", [])})
        for sk, v in zip(b.get("skills", []), sl):
            per_skill.setdefault(sk, []).append(v)
    scores = [r["score"] for r in rows]
    summary = {"dir": str(d), "evaluator": f"geneval2_{method}", "steps": meta["steps"],
               "images_root": meta["images_root"], "n": len(rows), "mean": sum(scores) / len(scores),
               "official_score_x100": 100 * sum(scores) / len(scores),
               "per_skill": {k: {"n_atoms": len(v), "mean": sum(v) / len(v)} for k, v in sorted(per_skill.items())},
               "per_prompt": rows}
    (d / "scores.json").write_text(json.dumps(summary, indent=1))
    print(f"geneval2 [{method}] @ {meta['steps']} steps | n={len(rows)} mean={summary['mean']:.4f} "
          f"official={summary['official_score_x100']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--images", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--benchmark", default=str(GENEVAL2 / "geneval2_data.jsonl"))
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--image_name", default="cand0.png")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_stage)
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--method", default="soft_tifa_gm", choices=sorted(POOLING))
    r.add_argument("--python", default=os.environ.get("GENEVAL2_PYTHON", sys.executable))
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
