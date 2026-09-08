#!/usr/bin/env python3
"""
T2I-CompBench official-evaluator driver — the PRIMARY (selector-independent)
metric of the Phase C1 pre-registration.

The vendored evaluators (T2I-CompBench/{BLIPvqa_eval,UniDet_eval,CLIPScore_eval})
are run UNMODIFIED so the numbers are the official ones. This script only:

  stage    our image tree -> the layout they expect ({dir}/samples/<prompt>_<id>.png)
  run      invoke the right official evaluator for a category, from its own cwd
  collect  join their vqa_result.json back onto the manifest -> per-prompt scores

Category -> evaluator (the official mapping, all 8 categories):
    color / shape / texture      BLIPvqa_eval/BLIP_vqa.py            (BLIP-VQA)
    spatial                      UniDet_eval/2D_spatial_eval.py      (UniDet)
    3d_spatial                   UniDet_eval/3D_spatial_eval.py      (UniDet + depth)
    numeracy                     UniDet_eval/numeracy_eval.py        (UniDet)
    non_spatial                  CLIPScore_eval/CLIP_similarity.py   (CLIPScore)
    complex                      3-in-1 = BLIP-VQA + UniDet-2D + CLIPScore

No evaluator here touches clip-flant5, so all are independent of the B4
selector. See reports/phaseC_eval_preregistration.md §3.

Filename contract (imposed by the evaluators, and brittle):
  BLIP   reads the prompt as  name.split('_')[0]   and the id as name.split('_')[-1]
  UniDet reads the id as      name.split('_')[1]
  CLIP   reads the prompt as  name.split('_')[0]
=> the staged name must be exactly "<prompt>_<id>.png" with ONE underscore, so a
   prompt containing '_' or '/' is rejected rather than silently mis-parsed.

Second, weaker contract (checked in `collect`): BLIP and CLIPScore do NOT parse
the id out of the filename for their `question_id` — they enumerate the sample
directory sorted by that id and emit `question_id = position`. That equals our id
only while the staged ids are contiguous 0..N-1, which `stage` produces and
`collect` asserts. UniDet parses the real id and is unaffected.

The `complex` category composes three sub-evaluators; the composition is done here
rather than by the upstream composer -- see `_compose_3_in_1`.

Usage
  stage   --images phaseA/images/M1 --steps 4 --categories color --limit 40 \
          --prompts phaseA/prompts.json --out phaseC/eval/M1_s4_color
  run     --dir phaseC/eval/M1_s4_color --evaluator blip
  collect --dir phaseC/eval/M1_s4_color
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPBENCH = REPO / "T2I-CompBench"

CATEGORY_EVALUATOR = {
    "color": "blip",
    "shape": "blip",
    "texture": "blip",
    "spatial": "unidet_2d",
    "3d_spatial": "unidet_3d",
    "numeracy": "unidet_numeracy",
    "non_spatial": "clipscore",
    "complex": "three_in_one",
}

# evaluator -> (cwd, argv, path of the per-image result json relative to the staging dir)
#
# `--complex True` is passed only where wanted: the vendored scripts declare the flag
# as argparse `type=bool`, so `--complex False` would also evaluate truthy.
EVALUATORS = {
    "blip": (
        COMPBENCH / "BLIPvqa_eval",
        lambda d, np_num: ["python", "BLIP_vqa.py", "--out_dir", str(d), "--np_num", str(np_num)],
        "annotation_blip/vqa_result.json",
    ),
    "unidet_2d": (
        COMPBENCH / "UniDet_eval",
        lambda d, _: ["python", "2D_spatial_eval.py", "--outpath", str(d)],
        "labels/annotation_obj_detection_2d/vqa_result.json",
    ),
    "unidet_2d_complex": (
        COMPBENCH / "UniDet_eval",
        lambda d, _: ["python", "2D_spatial_eval.py", "--outpath", str(d), "--complex", "True"],
        "labels/annotation_obj_detection_2d/vqa_result.json",
    ),
    "unidet_3d": (
        COMPBENCH / "UniDet_eval",
        lambda d, _: ["python", "3D_spatial_eval.py", "--outpath", str(d)],
        "labels/annotation_obj_detection_3d/vqa_result.json",
    ),
    # numeracy_eval.py writes to {outpath}/annotation_num/, NOT {outpath}/labels/... —
    # it is the one UniDet script that skips the `labels/` level (verified job 85655).
    "unidet_numeracy": (
        COMPBENCH / "UniDet_eval",
        lambda d, _: ["python", "numeracy_eval.py", "--outpath", str(d)],
        "annotation_num/vqa_result.json",
    ),
    # Same detector/dedup/scoring as `unidet_numeracy`; only the prompt-parsing lexicon is
    # overridable, via COMPBENCH_OBJECTS. Needed to score prompt distributions outside
    # CompBench's own 101-object vocabulary (GenEval counting) without swapping evaluators.
    # Verified to leave all 300 CompBench numeracy parses unchanged — phaseF/geneval_gap/.
    "unidet_numeracy_ext": (
        COMPBENCH / "UniDet_eval",
        lambda d, _: ["python", "numeracy_eval_ext.py", "--outpath", str(d)],
        "annotation_num/vqa_result.json",
    ),
    "clipscore": (
        COMPBENCH,
        lambda d, _: ["python", "CLIPScore_eval/CLIP_similarity.py", "--outpath", str(d)],
        "annotation_clip/vqa_result.json",
    ),
    "clipscore_complex": (
        COMPBENCH,
        lambda d, _: ["python", "CLIPScore_eval/CLIP_similarity.py", "--outpath", str(d),
                      "--complex", "True"],
        "annotation_clip/vqa_result.json",
    ),
}

# Composite evaluators: `run` executes each part in order into the SAME staging dir
# (each writes to a different result path), and `collect` combines them.
COMPOSITE = {
    "three_in_one": ["blip", "unidet_2d_complex", "clipscore_complex"],
}

# Evaluators whose `question_id` is a position in the id-sorted sample listing rather
# than the id parsed out of the filename (see module docstring).
POSITIONAL_QID = {"blip", "clipscore", "clipscore_complex"}


def _safe_for_filename(prompt: str) -> bool:
    return "_" not in prompt and "/" not in prompt


def _parts(evaluator: str) -> list[str]:
    return COMPOSITE.get(evaluator, [evaluator])


# ── stage ────────────────────────────────────────────────────────────────────

def cmd_stage(args):
    prompts = json.loads(Path(args.prompts).read_text())
    if args.categories:
        wanted = set(args.categories.split(","))
        prompts = [p for p in prompts if p["category"] in wanted]
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        sys.exit("no prompts selected")

    images_root = Path(args.images)
    out = Path(args.out)
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
        # One or several images per prompt (the official protocol uses 10). Each image is staged
        # under its own question_id; `collect` averages back to one score per prompt.
        names = [n for n in args.image_name.split(",") if n]
        srcs = [images_root / f"p{idx:05d}" / f"s{args.steps}" / n for n in names]
        absent = [str(s) for s in srcs if not s.exists()]
        if absent:
            missing += absent
            continue
        for name, src in zip(names, srcs):
            qid = len(manifest)
            dst = samples / f"{prompt}_{qid:06d}.png"
            dst.symlink_to(src.resolve())          # symlink: the images are already on lustre
            manifest.append({
                "question_id": qid, "idx": idx, "category": item["category"],
                "prompt": prompt, "image": name, "src": str(src.resolve()), "staged": dst.name,
            })

    meta = {
        "images_root": str(images_root.resolve()), "steps": args.steps,
        "categories": sorted({m["category"] for m in manifest}),
        "n": len(manifest), "manifest": manifest,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))

    print(f"staged {len(manifest)} images -> {samples}")
    if unsafe:
        print(f"  SKIPPED {len(unsafe)} prompts unusable as filenames (contain '_' or '/'):")
        for p in unsafe[:5]:
            print(f"    {p!r}")
    if missing:
        msg = f"{len(missing)} of {len(prompts)} images missing, e.g. {missing[0]}"
        if not args.allow_missing:
            sys.exit(f"stage: {msg}; a partial benchmark is not the benchmark (pass --allow_missing to override)")
        print(f"  WARNING {msg}")
    cats = sorted({m["category"] for m in manifest})
    print(f"  categories: {cats}")
    print(f"  evaluators: {sorted({CATEGORY_EVALUATOR.get(c, '?') for c in cats})}")


# ── run ──────────────────────────────────────────────────────────────────────

def cmd_run(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    cats = meta["categories"]

    evaluator = args.evaluator
    if evaluator is None:
        needed = {CATEGORY_EVALUATOR.get(c) for c in cats}
        if len(needed) != 1 or None in needed:
            sys.exit(f"categories {cats} need evaluators {needed}; pass --evaluator "
                     f"explicitly and stage one category at a time")
        evaluator = needed.pop()

    for part in _parts(evaluator):
        cwd, argv_fn, rel = EVALUATORS[part]
        if args.skip_done and (d / rel).exists():
            print(f"[{part}] skip, {rel} already present", flush=True)
            continue
        argv = argv_fn(d, args.np_num)
        print(f"[{part}] cwd={cwd}\n[{part}] $ {' '.join(argv)}", flush=True)

        env = dict(os.environ)
        env.setdefault("TORCH_HOME", str(REPO / "cache" / "torch"))
        env.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
        # _compbench_compat supplies `ruamel_yaml`, which the evaluators import in
        # preference to ruamel.yaml — see that package's docstring. It must precede
        # cwd, which supplies the evaluators' own top-level `models`/`utils`/`experts`.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "phaseC" / "_compbench_compat"), str(cwd), env.get("PYTHONPATH", "")]
        )

        rc = subprocess.call(argv, cwd=cwd, env=env)
        if rc != 0:
            sys.exit(f"{part} exited {rc}")
    (d / "evaluator.txt").write_text(evaluator)
    print(f"[{evaluator}] done")


# ── 3-in-1 composition ───────────────────────────────────────────────────────

def _complex_routing(dataset_dir: Path, split: str):
    """The official per-prompt routing for the `complex` category.

    `3_in_1_eval/3_in_1.py` decides each prompt's weighting by membership in
    `complex_{split}_spatial.txt` / `complex_{split}_action.txt`, keyed on the text
    before the first period, lowercased. Reproduced exactly, including the fact that
    `complex_{split}_spatialaction.txt` is NOT consulted — those prompts fall through
    to the 3-way branch, which is the equal-weight blend they should get.
    """
    def key(s: str) -> str:
        return s.strip("\n").split(".")[0].lower()

    def load(name: str) -> set:
        f = dataset_dir / name
        if not f.exists():
            sys.exit(f"missing 3-in-1 routing file {f}")
        return {key(ln) for ln in f.read_text().splitlines() if ln.strip()}

    return load(f"complex_{split}_spatial.txt"), load(f"complex_{split}_action.txt")


def _compose_3_in_1(attr, spat, act, prompt, spatial_set, action_set):
    """The official 3-in-1 weighting for one prompt.

    Upstream `3_in_1.py` cannot be invoked on this layout: it hardcodes 10 images per
    prompt (`num=10`) and indexes the sub-scores by line position in complex_val.txt,
    whereas the pre-registered protocol samples 1 image per prompt over a subset. The
    weighting itself is three lines of arithmetic, reproduced here verbatim from
    upstream's if/elif/else and proven equal to it over all 300 complex_val prompts by
    `phaseC/test_3in1_identity.py`.
    """
    k = prompt.split(".")[0].lower()
    if k in spatial_set:                       # spatial relation + attribute
        return (spat + attr) * 0.5, "spatial+attr"
    if k in action_set:                        # action relation + attribute
        return (act + attr) * 0.5, "action+attr"
    return (attr + spat + act) / 3.0, "attr+spatial+action"


# ── collect ──────────────────────────────────────────────────────────────────

def _load_result(d: Path, part: str, n_staged: int) -> dict:
    """-> {question_id: score} for one sub-evaluator, with the qid contract checked."""
    result_path = d / EVALUATORS[part][2]
    if not result_path.exists():
        sys.exit(f"no result at {result_path} — did `run` succeed?")
    raw = json.loads(result_path.read_text())
    by_qid = {int(r["question_id"]): float(r["answer"]) for r in raw}
    if part in POSITIONAL_QID and sorted(by_qid) != list(range(n_staged)):
        sys.exit(f"{part} emits question_id = position in the id-sorted sample listing, "
                 f"which only equals our id for contiguous ids 0..N-1; got "
                 f"{len(by_qid)} ids over {n_staged} staged images. Refusing to "
                 f"join possibly-misaligned scores.")
    return by_qid


def cmd_collect(args):
    d = Path(args.dir).resolve()
    meta = json.loads((d / "manifest.json").read_text())
    evaluator = args.evaluator or (d / "evaluator.txt").read_text().strip()
    n_staged = len(meta["manifest"])

    if evaluator == "three_in_one":
        attr_s, spat_s, act_s = (_load_result(d, p, n_staged)
                                 for p in COMPOSITE["three_in_one"])
        spatial_set, action_set = _complex_routing(
            Path(args.dataset_dir), args.routing_split)
        rows, unrouted = [], 0
        for m in meta["manifest"]:
            q = m["question_id"]
            if not (q in attr_s and q in spat_s and q in act_s):
                continue
            score, branch = _compose_3_in_1(attr_s[q], spat_s[q], act_s[q],
                                            m["prompt"], spatial_set, action_set)
            if branch == "attr+spatial+action":
                unrouted += 1
            rows.append({**m, "score": score, "branch": branch,
                         "sub": {"attribute": attr_s[q], "spatial": spat_s[q],
                                 "action": act_s[q]}})
        if unrouted:
            print(f"  {unrouted}/{len(rows)} prompts took the 3-way branch "
                  f"(spatial+action, or absent from both routing lists)")
    else:
        by_qid = _load_result(d, evaluator, n_staged)
        rows = [{**m, "score": by_qid[m["question_id"]]}
                for m in meta["manifest"] if m["question_id"] in by_qid]
    if not rows:
        sys.exit("no scores matched the manifest")

    # Several staged images per prompt (official protocol: 10) -> one score per prompt, the mean
    # over its images. With one image per prompt this is the identity.
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
        if "sub" in rs[0]:
            row["sub"] = {k: sum(r["sub"][k] for r in rs) / len(rs) for k in rs[0]["sub"]}
        rows.append(row)
    n_per_prompt = sorted({len(v) for v in by_idx.values()})

    scores = [r["score"] for r in rows]
    per_cat = {}
    for r in rows:
        per_cat.setdefault(r["category"], []).append(r["score"])

    summary = {
        "dir": str(d), "evaluator": evaluator, "steps": meta["steps"],
        "images_root": meta["images_root"], "n": len(rows),
        "mean": sum(scores) / len(scores),
        "per_category": {c: {"n": len(v), "mean": sum(v) / len(v)}
                         for c, v in sorted(per_cat.items())},
        "per_prompt": rows,
        "images_per_prompt": n_per_prompt,
        "per_image": per_image,
    }
    if evaluator == "three_in_one":
        per_branch, per_sub = {}, {}
        for r in rows:
            per_branch.setdefault(r["branch"], []).append(r["score"])
            for k, v in r["sub"].items():
                per_sub.setdefault(k, []).append(v)
        summary["routing_split"] = args.routing_split
        summary["per_branch"] = {b: {"n": len(v), "mean": sum(v) / len(v)}
                                 for b, v in sorted(per_branch.items())}
        summary["sub_evaluator_mean"] = {k: sum(v) / len(v) for k, v in sorted(per_sub.items())}
    (d / "scores.json").write_text(json.dumps(summary, indent=1))

    print(f"{evaluator} @ {meta['steps']} steps | {Path(meta['images_root']).name}")
    print(f"  n={len(rows)}  mean={summary['mean']:.4f}")
    for c, v in summary["per_category"].items():
        print(f"    {c:12s} n={v['n']:4d}  mean={v['mean']:.4f}")
    for b, v in summary.get("per_branch", {}).items():
        print(f"    [branch] {b:22s} n={v['n']:4d}  mean={v['mean']:.4f}")
    if "sub_evaluator_mean" in summary:
        print("    [sub]  " + "  ".join(f"{k}={v:.4f}"
                                        for k, v in summary["sub_evaluator_mean"].items()))
    print(f"  -> {d / 'scores.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage")
    s.add_argument("--images", required=True, help="e.g. phaseA/images/M1")
    s.add_argument("--prompts", default="phaseA/prompts.json")
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
    r.add_argument("--skip_done", action="store_true",
                   help="skip sub-evaluators whose result json already exists "
                        "(makes an interrupted 3-in-1 resumable)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("collect")
    c.add_argument("--dir", required=True)
    c.add_argument("--evaluator", choices=choices, default=None)
    c.add_argument("--dataset_dir", default=str(COMPBENCH / "examples" / "dataset"),
                   help="3-in-1 routing lists live here (complex_{split}_{spatial,action}.txt)")
    c.add_argument("--routing_split", default="val", choices=["train", "val"],
                   help="which complex_* routing lists to use; must match the eval split")
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
