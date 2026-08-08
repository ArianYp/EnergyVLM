#!/usr/bin/env python3
"""Package every results artifact into one self-contained zip.

Deliberately excludes checkpoints (~10-20 GB each), generated image pools, W&B run directories and
the VQA annotation trees — `reports/phaseFP_results/merged_compbench_scores` alone dereferences to
13 GB because its symlinks point at full evaluator working directories. What is kept is everything
needed to re-derive every number: the reports, the analysis code, and the per-prompt score files.

The visual audit is made self-contained: its 175 referenced images are copied in and the HTML paths
rewritten, so the contact sheet opens correctly from the extracted zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOC_SUFFIX = {".md", ".json", ".html", ".csv", ".txt"}
CODE_SUFFIX = {".py", ".lsf", ".sh"}
SKIP_DIR = {"_results_bundle", "_bundle", "wandb", "__pycache__", "images", "tmp", "samples", "preference_pairs",
            "annotation1_blip", "annotation2_blip", "annotation3_blip", "annotation4_blip",
            "annotation5_blip", "annotation6_blip", "annotation7_blip", "annotation8_blip",
            "VQA", "labels", "annotation_blip", "annotation_obj_detection_2d",
            "annotation_obj_detection_3d", "annotation_num", "annotation_clip"}


def copy_tree(src: Path, dst: Path, suffixes: set[str], max_mb: float = 25.0) -> int:
    """Copy matching files, PRUNING skipped directories during the walk.

    rglob() enumerates every path before any filter applies, and the evaluation roots hold
    hundreds of thousands of generated PNGs — the walk alone dominates. os.walk with in-place
    pruning of `dirs` never descends into them.
    """
    import os

    n = 0
    if not src.exists():
        return 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR and not d.startswith("p0")]
        rootp = Path(root)
        for fname in files:
            p_ = rootp / fname
            if p_.is_symlink() or p_.suffix.lower() not in suffixes:
                continue
            try:
                if p_.stat().st_size > max_mb * 1e6:
                    continue
            except OSError:
                continue
            out = dst / p_.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p_, out)
            n += 1
    return n


def collect_scores(stage: Path) -> dict:
    """Per-prompt score files — the raw data every reported number is computed from."""
    counts = {}
    # CompBench + GenEval2, from each evaluation root (symlinks resolved, trees not walked).
    for root in sorted(ROOT.glob("phaseFP/eval_*")) + sorted(ROOT.glob("phaseI/*eval*")):
        for kind in ("compbench_scores", "geneval2_scores"):
            base = root / kind
            if not base.exists():
                continue
            for d in sorted(base.iterdir()):
                real = d.resolve() if d.is_symlink() else d
                f = real / "scores.json"
                if f.exists():
                    out = stage / "scores" / root.name / kind / d.name / "scores.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out)
                    counts[kind] = counts.get(kind, 0) + 1
    # Teacher selection-headroom scores and the exact per-candidate expectations.
    for base, tag in ((ROOT / "phaseFP/teacher_ceiling_101493/compbench_scores", "teacher_ceiling"),
                      (ROOT / "phaseFP/exact_headroom_101942/cand_scores", "exact_headroom")):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            f = d / "scores.json"
            if f.exists():
                out = stage / "scores" / tag / d.name / "scores.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                counts[tag] = counts.get(tag, 0) + 1
    return counts


def selfcontain_visual_audit(stage: Path) -> int:
    src = ROOT / "reports/phaseFP_results/visual_audit"
    dst = stage / "reports/phaseFP_results/visual_audit"
    html = src / "contact_sheet.html"
    if not html.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    text = html.read_text()
    srcs = re.findall(r"<img src='([^']+)'", text)
    (dst / "img").mkdir(exist_ok=True)
    mapping, n = {}, 0
    for s in srcs:
        p = (src / s).resolve()
        if not p.exists():
            continue
        # Flatten to a unique name: <model>_<prompt-dir>.png
        parts = p.parts
        name = f"{parts[-4]}_{parts[-3]}.png" if len(parts) >= 4 else p.name
        if s not in mapping:
            shutil.copy2(p, dst / "img" / name)
            mapping[s] = f"img/{name}"
            n += 1
    for old, new in mapping.items():
        text = text.replace(f"<img src='{old}'", f"<img src='{new}'")
    (dst / "contact_sheet.html").write_text(text)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"energyvlm_results_{stamp}"
    stage = ROOT / "_results_bundle" / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    summary = {"created_utc": stamp, "contents": {}}

    # ---- reports: every markdown / json / html result document ---------------------------------
    summary["contents"]["reports"] = copy_tree(
        ROOT / "reports", stage / "reports", DOC_SUFFIX)
    # ---- analysis + training code -------------------------------------------------------------
    for sub in ("audit", "phaseFP", "phaseM"):
        summary["contents"][f"code/{sub}"] = copy_tree(
            ROOT / sub, stage / sub, CODE_SUFFIX | {".md", ".json", ".csv"})
    # ---- the LSF job scripts that produced everything -----------------------------------------
    n = 0
    (stage / "ablations").mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "ablations").glob("*.lsf")):
        if p.name.startswith(("phaseFP", "phaseM", "probe_gpu", "phaseI", "phaseL")):
            shutil.copy2(p, stage / "ablations" / p.name)
            n += 1
    summary["contents"]["ablations"] = n
    # ---- the modified trainer -----------------------------------------------------------------
    (stage / "phaseI").mkdir(parents=True, exist_ok=True)
    for f in ("train_preference.py", "test_preference.py"):
        if (ROOT / "phaseI" / f).exists():
            shutil.copy2(ROOT / "phaseI" / f, stage / "phaseI" / f)
    # ---- raw per-prompt scores ----------------------------------------------------------------
    summary["contents"]["scores"] = collect_scores(stage)
    # ---- self-contained visual audit ----------------------------------------------------------
    summary["contents"]["visual_audit_images"] = selfcontain_visual_audit(stage)

    readme = [
        "# EnergyVLM — results bundle",
        "",
        f"Created {stamp}. Start with `reports/START_HERE.md`.",
        "",
        "## What is here",
        "",
        "| path | contents |",
        "|---|---|",
        "| `reports/START_HERE.md` | **the report** — all results, explained |",
        "| `reports/phaseFP_results/` | full results, per-seed tables, mechanism verdict, Pareto |",
        "| `reports/audit_2026-08-07/` | zero-training audit, PSO reduction, selection diversity price |",
        "| `reports/registry/` | immutable result registry |",
        "| `reports/phaseFP_results/visual_audit/` | non-cherry-picked contact sheet (self-contained) |",
        "| `scores/` | **raw per-prompt score files** — every number is recomputable from these |",
        "| `phaseFP/`, `audit/`, `phaseM/` | analysis and training code |",
        "| `ablations/` | the LSF job scripts that produced the runs |",
        "",
        "## What is deliberately excluded",
        "",
        "Model checkpoints (~10-20 GB each), generated image pools, W&B run directories, and the",
        "evaluator annotation trees. `merged_compbench_scores` dereferences to 13 GB; only the",
        "`scores.json` files it points at are included, which is all the analyses read.",
        "",
        "## Reproducing a headline number",
        "",
        "```bash",
        "# dose slope across three seeds",
        "python3 phaseFP/analyze_seeds.py --seed_set 's1=correct:CorrectFixed:<scores>' ...",
        "# any CompBench contrast split by evaluator architecture",
        "python3 audit/evaluator_family_bootstrap.py --scores_dir <scores> \\",
        "    --treatment CorrectFixed --control CounterfactualFixed --out x.json",
        "```",
        "",
        "Paths in the scripts assume the original repository layout; `scores/` here is grouped by",
        "evaluation job rather than merged, so point `--scores_dir` at a merged view of it.",
    ]
    (stage / "README.md").write_text("\n".join(readme) + "\n")
    (stage / "MANIFEST.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    out = Path(args.out) if args.out else ROOT / "_results_bundle" / f"{name}.zip"
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=stage.parent, base_dir=name)
    size = out.stat().st_size
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    files = sum(1 for _ in stage.rglob("*") if _.is_file())
    print(json.dumps({"zip": str(out), "size_mb": round(size / 1e6, 1), "files": files,
                      "sha256_16": sha, "contents": summary["contents"]}, indent=2))


if __name__ == "__main__":
    main()
