#!/usr/bin/env python3
"""T2I-CompBench scoring with the official evaluators, run unmodified from third_party/T2I-CompBench.

  stage    our image tree -> the layout the evaluators expect ({dir}/samples/<prompt>_<id>.png)
  run      invoke the official evaluator for the staged category, from its own cwd
  collect  join its vqa_result.json back onto the manifest -> per-prompt scores

Category -> evaluator:
    color / shape / texture      BLIPvqa_eval/BLIP_vqa.py            (BLIP-VQA)
    spatial                      UniDet_eval/2D_spatial_eval.py      (UniDet)
    3d_spatial                   UniDet_eval/3D_spatial_eval.py      (UniDet + depth)
    numeracy                     UniDet_eval/numeracy_eval.py        (UniDet)
    non_spatial                  CLIPScore_eval/CLIP_similarity.py   (CLIPScore)
    complex                      3-in-1 = BLIP-VQA + UniDet-2D + CLIPScore

Filename contract imposed by the evaluators: the staged name must be exactly "<prompt>_<id>.png"
with one underscore (BLIP reads the prompt as name.split('_')[0], UniDet reads the id as
name.split('_')[1]), so prompts containing '_' or '/' are skipped rather than mis-parsed. BLIP and
CLIPScore also emit question_id as the POSITION in the id-sorted sample listing, which equals our
id only for contiguous ids 0..N-1; `stage` produces that and `collect` asserts it.

Usage
  python eval/compbench.py stage   --images out/eval/compbench/images/LABEL --prompts pools/eval/compbench_prompts.json \
                                   --steps 4 --categories color --out out/eval/compbench_scores/LABEL_s4_color
  python eval/compbench.py run     --dir out/eval/compbench_scores/LABEL_s4_color --skip_done
  python eval/compbench.py collect --dir out/eval/compbench_scores/LABEL_s4_color
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPBENCH = Path(os.environ.get("COMPBENCH_DIR", ROOT / "third_party" / "T2I-CompBench"))
COMPAT = ROOT / "eval" / "compat"

CATEGORY_EVALUATOR = {
    "color": "blip", "shape": "blip", "texture": "blip",
    "spatial": "unidet_2d", "3d_spatial": "unidet_3d", "numeracy": "unidet_numeracy",
    "non_spatial": "clipscore", "complex": "three_in_one",
}

# evaluator -> (cwd, argv, per-image result json relative to the staging dir). `--complex True` is
# passed only where wanted: the scripts declare the flag as argparse type=bool, so "False" is truthy.
EVALUATORS = {
    "blip": (COMPBENCH / "BLIPvqa_eval",
             lambda d, np_num: ["python", "BLIP_vqa.py", "--out_dir", str(d), "--np_num", str(np_num)],
             "annotation_blip/vqa_result.json"),
    "unidet_2d": (COMPBENCH / "UniDet_eval",
                  lambda d, _: ["python", "2D_spatial_eval.py", "--outpath", str(d)],
                  "labels/annotation_obj_detection_2d/vqa_result.json"),
    "unidet_2d_complex": (COMPBENCH / "UniDet_eval",
                          lambda d, _: ["python", "2D_spatial_eval.py", "--outpath", str(d), "--complex", "True"],
                          "labels/annotation_obj_detection_2d/vqa_result.json"),
    "unidet_3d": (COMPBENCH / "UniDet_eval",
                  lambda d, _: ["python", "3D_spatial_eval.py", "--outpath", str(d)],
                  "labels/annotation_obj_detection_3d/vqa_result.json"),
    "unidet_numeracy": (COMPBENCH / "UniDet_eval",
                        lambda d, _: ["python", "numeracy_eval.py", "--outpath", str(d)],
                        "annotation_num/vqa_result.json"),
    "clipscore": (COMPBENCH,
                  lambda d, _: ["python", "CLIPScore_eval/CLIP_similarity.py", "--outpath", str(d)],
                  "annotation_clip/vqa_result.json"),
    "clipscore_complex": (COMPBENCH,
                          lambda d, _: ["python", "CLIPScore_eval/CLIP_similarity.py", "--outpath", str(d), "--complex", "True"],
                          "annotation_clip/vqa_result.json"),
}
COMPOSITE = {"three_in_one": ["blip", "unidet_2d_complex", "clipscore_complex"]}
POSITIONAL_QID = {"blip", "clipscore", "clipscore_complex"}


def _parts(evaluator: str) -> list[str]:
    return COMPOSITE.get(evaluator, [evaluator])


def cmd_stage(args):
    prompts = json.loads(Path(args.prompts).read_text())
    if args.categories:
        wanted = set(args.categories.split(","))
        prompts = [p for p in prompts if p["category"] in wanted]
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        sys.exit("no prompts selected")
    images_root, out = Path(args.images), Path(args.out)
    samples = out / "samples"
    if samples.exists():
        shutil.rmtree(samples)
    samples.mkdir(parents=True)

    manifest, missing, unsafe = [], [], []
    for item in prompts:
        prompt, idx = item["prompt"], item["idx"]
        if "_" in prompt or "/" in prompt:
            unsafe.append(prompt)
            continue
        # one or several images per prompt (official protocol: 10); each staged under its own
        # question_id, `collect` averages back to one score per prompt
        names = [n for n in args.image_name.split(",") if n]
        srcs = [images_root / f"p{idx:05d}" / f"s{args.steps}" / n for n in names]
        absent = [str(s) for s in srcs if not s.exists()]
        if absent:
            missing += absent
            continue
        for name, src in zip(names, srcs):
            qid = len(manifest)
            (samples / f"{prompt}_{qid:06d}.png").symlink_to(src.resolve())
            manifest.append({"question_id": qid, "idx": idx, "category": item["category"],
                             "prompt": prompt, "image": name, "src": str(src.resolve())})
    (out / "manifest.json").write_text(json.dumps({
        "images_root": str(images_root.resolve()), "steps": args.steps,
        "categories": sorted({m["category"] for m in manifest}), "n": len(manifest),
        "manifest": manifest}, indent=1))
    print(f"staged {len(manifest)} images -> {samples}")
    if unsafe:
        print(f"  skipped {len(unsafe)} prompts unusable as filenames")
    if missing:
        msg = f"{len(missing)} of {len(prompts)} images missing, e.g. {missing[0]}"
        if not args.allow_missing:
            sys.exit(f"stage: {msg}; a partial benchmark is not the benchmark (pass --allow_missing to override)")
        print(f"  WARNING {msg}")


def cmd_run(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    evaluator = args.evaluator
    if evaluator is None:
        needed = {CATEGORY_EVALUATOR.get(c) for c in meta["categories"]}
        if len(needed) != 1 or None in needed:
            sys.exit(f"categories {meta['categories']} need evaluators {needed}; stage one category at a time")
        evaluator = needed.pop()
    for part in _parts(evaluator):
        cwd, argv_fn, rel = EVALUATORS[part]
        if args.skip_done and (d / rel).exists():
            print(f"[{part}] skip, {rel} already present", flush=True)
            continue
        argv = argv_fn(d, args.np_num)
        print(f"[{part}] cwd={cwd}\n[{part}] $ {' '.join(argv)}", flush=True)
        env = dict(os.environ)
        # eval/compat supplies `ruamel_yaml`; it must precede cwd, which supplies the evaluators'
        # own top-level modules.
        env["PYTHONPATH"] = os.pathsep.join([str(COMPAT), str(cwd), env.get("PYTHONPATH", "")])
        rc = subprocess.call(argv, cwd=cwd, env=env)
        if rc != 0:
            sys.exit(f"{part} exited {rc}")
    (d / "evaluator.txt").write_text(evaluator)
    print(f"[{evaluator}] done")


def _complex_routing(dataset_dir: Path, split: str):
    """Official per-prompt routing for `complex`: membership in complex_{split}_{spatial,action}.txt,
    keyed on the text before the first period, lowercased (as in 3_in_1_eval/3_in_1.py)."""
    def key(s: str) -> str:
        return s.strip("\n").split(".")[0].lower()

    def load(name: str) -> set:
        f = dataset_dir / name
        if not f.exists():
            sys.exit(f"missing 3-in-1 routing file {f}")
        return {key(ln) for ln in f.read_text().splitlines() if ln.strip()}
    return load(f"complex_{split}_spatial.txt"), load(f"complex_{split}_action.txt")


def _compose_3_in_1(attr, spat, act, prompt, spatial_set, action_set):
    """The official 3-in-1 weighting, reproduced from upstream's if/elif/else."""
    k = prompt.split(".")[0].lower()
    if k in spatial_set:
        return (spat + attr) * 0.5, "spatial+attr"
    if k in action_set:
        return (act + attr) * 0.5, "action+attr"
    return (attr + spat + act) / 3.0, "attr+spatial+action"


def _load_result(d: Path, part: str, n_staged: int) -> dict:
    result_path = d / EVALUATORS[part][2]
    if not result_path.exists():
        sys.exit(f"no result at {result_path}; did `run` succeed?")
    by_qid = {int(r["question_id"]): float(r["answer"]) for r in json.loads(result_path.read_text())}
    if part in POSITIONAL_QID and sorted(by_qid) != list(range(n_staged)):
        sys.exit(f"{part} emits positional question_ids; got {len(by_qid)} over {n_staged} staged images")
    return by_qid


def cmd_collect(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    evaluator = args.evaluator or (d / "evaluator.txt").read_text().strip()
    n_staged = len(meta["manifest"])
    if evaluator == "three_in_one":
        attr_s, spat_s, act_s = (_load_result(d, p, n_staged) for p in COMPOSITE["three_in_one"])
        spatial_set, action_set = _complex_routing(Path(args.dataset_dir), args.routing_split)
        rows = []
        for m in meta["manifest"]:
            q = m["question_id"]
            if q in attr_s and q in spat_s and q in act_s:
                score, branch = _compose_3_in_1(attr_s[q], spat_s[q], act_s[q], m["prompt"], spatial_set, action_set)
                rows.append({**m, "score": score, "branch": branch})
    else:
        by_qid = _load_result(d, evaluator, n_staged)
        rows = [{**m, "score": by_qid[m["question_id"]]} for m in meta["manifest"] if m["question_id"] in by_qid]
    if not rows:
        sys.exit("no scores matched the manifest")
    # several staged images per prompt -> one score per prompt (mean over its images)
    per_image = rows
    by_idx: dict = {}
    for r in rows:
        by_idx.setdefault(r["idx"], []).append(r)
    rows = []
    for i, rs in sorted(by_idx.items()):
        row = {k: rs[0][k] for k in ("question_id", "idx", "category", "prompt", "src")}
        row.update({"score": sum(r["score"] for r in rs) / len(rs), "n_images": len(rs),
                    "image_scores": [r["score"] for r in rs]})
        if "branch" in rs[0]:
            row["branch"] = rs[0]["branch"]
        rows.append(row)
    scores = [r["score"] for r in rows]
    per_cat: dict[str, list[float]] = {}
    for r in rows:
        per_cat.setdefault(r["category"], []).append(r["score"])
    summary = {"dir": str(d), "evaluator": evaluator, "steps": meta["steps"],
               "images_root": meta["images_root"], "n": len(rows), "mean": sum(scores) / len(scores),
               "per_category": {c: {"n": len(v), "mean": sum(v) / len(v)} for c, v in sorted(per_cat.items())},
               "per_prompt": rows, "images_per_prompt": sorted({len(v) for v in by_idx.values()}),
               "per_image": per_image}
    (d / "scores.json").write_text(json.dumps(summary, indent=1))
    print(f"{evaluator} @ {meta['steps']} steps | n={len(rows)} mean={summary['mean']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--images", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--image_name", default="cand0.png")
    s.add_argument("--categories", default=None, help="comma-separated")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--allow_missing", action="store_true", help="stage a partial set instead of failing")
    s.set_defaults(func=cmd_stage)
    choices = sorted(set(EVALUATORS) | set(COMPOSITE))
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--evaluator", choices=choices, default=None)
    r.add_argument("--np_num", type=int, default=8, help="BLIP noun-phrase passes")
    r.add_argument("--skip_done", action="store_true")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("collect")
    c.add_argument("--dir", required=True)
    c.add_argument("--evaluator", choices=choices, default=None)
    c.add_argument("--dataset_dir", default=str(COMPBENCH / "examples" / "dataset"))
    c.add_argument("--routing_split", default="val", choices=["train", "val"])
    c.set_defaults(func=cmd_collect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
