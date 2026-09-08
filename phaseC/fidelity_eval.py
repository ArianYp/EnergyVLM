#!/usr/bin/env python3
"""
Publication-scale fidelity for the Phase C1 gate: FID **and** CMMD vs COCO-val2017 real.

The pre-registration (§3, §5) makes fidelity a *gate*, not a headline: a B4−B2
alignment gain "bought by mode-collapse or quality loss is not a pass", with the
explicit condition **CMMD(B4) <= CMMD(B2) + tolerance**. Phase A's fidelity check ran
at 640 CompBench images with CMMD only, which it flagged as an internal gate; this is
the pre-registered version — COCO captions, >=5k samples, matched preprocessing, FID
alongside CMMD, plus improved precision/recall to separate "worse fidelity" from
"narrower support" (the two failure modes the gate cares about).

Metrics, all against the same reference set of real COCO val2017 images:

  FID        Inception-pool3 Frechet distance via clean-fid (`mode="clean"`), the
             standard whose resizing pipeline is applied identically to both sides.
  CMMD       CLIP-ViT-L/14 MMD, sigma=10, x1000 — imported from `exp0/fidelity_cmmd.py`
             rather than reimplemented, so these numbers are directly comparable with
             the Phase-A fidelity report.
  precision  improved precision/recall (Kynkaanniemi et al. 2019, k=3) on the CLIP
  / recall   features: precision falls when samples leave the real manifold (quality
             loss), recall falls when they cover less of it (diversity collapse).
             **Recall is the load-bearing half here.** k-NN radii in 768-d are large
             relative to the spread of the features, so precision saturates near 1 for
             anything remotely image-like -- `phaseC/test_metrics.py` shows it stays
             >0.9 even for a deliberately off-manifold set, while recall correctly
             collapses to 0.02 for a mode-collapsed one. Treat precision as a coarse
             sanity check and read quality loss off FID/CMMD instead.

Preprocessing is matched by construction — both sides pass through the same resizer —
and `--square_ref` (default) center-crops the real images to square first, so the
comparison is not confounded by generated images being square while COCO is not.

CIs: CMMD and precision/recall are bootstrapped over generated images. FID is a
plug-in estimator of a covariance-based quantity and is strongly sample-size biased,
so instead of a bootstrap it gets a **split-half** check: FID on two disjoint halves
of the generated set, which shows how much of a model-to-model difference is noise.

Usage
  python phaseC/fidelity_eval.py --gen_root phaseC/fidelity/images \
      --models B4,B2,M1,base --steps 4,8,28 --out phaseC/fidelity_report.md
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "exp0"))
sys.path.insert(0, str(REPO / "phaseW"))
from fidelity_cmmd import clip_embed as legacy_clip_embed  # noqa: E402  ViT-L/14@224, NOT unit-norm
import cmmd_ref  # noqa: E402  the reference CMMD (ViT-L/14@336, crop+bicubic, unit-norm, sigma 10)
from cmmd_ref import cmmd  # noqa: E402

BOOT = 1_000

# clean-fid hardcodes its Inception cache to /tmp on Linux (cleanfid.features
# .feature_extractor: `path = "./" if Windows else "/tmp"`), which is node-local and
# ephemeral — so every job on a fresh node re-downloads 95 MB from an NVIDIA CDN, and a
# node without egress fails outright. Seed /tmp from a copy kept on shared storage
# instead; check_download_inception() skips the download when the file is already there.
INCEPTION_WEIGHT = REPO / "cache" / "cleanfid" / "inception-2015-12-05.pt"


def ensure_inception_weight():
    dst = Path("/tmp/inception-2015-12-05.pt")
    if dst.exists() or not INCEPTION_WEIGHT.exists():
        return  # already staged, or no cached copy -> let clean-fid download it
    try:
        shutil.copy(INCEPTION_WEIGHT, dst)
        print(f"[fid] staged Inception weight {INCEPTION_WEIGHT} -> {dst}")
    except OSError as e:
        print(f"[fid] could not stage Inception weight ({e}); clean-fid will download it")


def square_crop(img_np):
    """Center crop to square, before clean-fid's resizer."""
    h, w = img_np.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img_np[top:top + s, left:left + s]


def inception_feats(files, model, square: bool, batch_size=128, num_workers=8):
    from cleanfid import fid as cfid
    return cfid.get_files_features(
        [str(f) for f in files], model, mode="clean", verbose=False,
        batch_size=batch_size, num_workers=num_workers,
        custom_image_tranform=square_crop if square else None)


def fid_from_feats(a, b):
    from cleanfid.fid import frechet_distance
    return float(frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                  b.mean(0), np.cov(b, rowvar=False)))


def precision_recall(real: torch.Tensor, fake: torch.Tensor, k: int = 3):
    """Improved precision/recall (Kynkaanniemi et al. 2019) on L2-normalised features.

    A sample is "in" the other set's manifold if it lies within that set's k-th
    nearest-neighbour radius for some member. precision = fraction of fake in the real
    manifold (quality); recall = fraction of real in the fake manifold (coverage).
    """
    def radii(x):
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).values[:, -1]

    real, fake = torch.nn.functional.normalize(real, dim=1), \
        torch.nn.functional.normalize(fake, dim=1)
    r_real, r_fake = radii(real), radii(fake)
    d = torch.cdist(fake, real)                                  # [n_fake, n_real]
    precision = float((d <= r_real[None, :]).any(1).float().mean())
    recall = float((d.T <= r_fake[None, :]).any(1).float().mean())
    return precision, recall


def cmmd_boot_ci(gen, ref, rng, reps, sigma=10.0, scale=1000.0):
    """Bootstrap CMMD over generated images, reusing one cached kernel.

    Recomputing `cmmd` per replicate would redo a 5k x 5k `cdist` thousands of times.
    MMD^2 is a mean over kernel entries, so the kernel is computed once and the
    replicates only re-index it: k(x,x) over resampled rows AND columns, k(x,y) over
    resampled rows, k(y,y) constant.
    """
    def k(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma * sigma))

    K_gg, K_gr, k_rr = k(gen, gen), k(gen, ref), k(ref, ref).mean()
    n = gen.shape[0]
    vals = np.empty(reps)
    for b in range(reps):
        idx = torch.as_tensor(rng.integers(0, n, n), device=gen.device)
        vals[b] = float((K_gg[idx][:, idx].mean() + k_rr - 2 * K_gr[idx].mean()) * scale)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="phaseC/fidelity/images")
    ap.add_argument("--models", default="B4,B2,M1,base")
    ap.add_argument("--gate_models", default="B4,B2",
                    help="treatment,control model labels for the non-inferiority gate")
    ap.add_argument("--title", default="Phase C1")
    ap.add_argument("--steps", default="4,8,28")
    ap.add_argument("--image_name", default="cand0.png")
    ap.add_argument("--coco_dir",
                    default="/lustre/scratch126/cellgen/lotfollahi/ha11/COCO/val2017")
    ap.add_argument("--cmmd_impl", default="reference", choices=["reference", "legacy"],
                    help="reference = the published CMMD recipe (cmmd_ref.py). legacy = the "
                         "project's earlier unnormalised CLIP-L/14@224 RBF-MMD, kept only so old "
                         "reports can be reproduced; its output is NOT CMMD and is labelled so")
    ap.add_argument("--clip_id", default="openai/clip-vit-large-patch14",
                    help="legacy CLIP model only; the reference implementation fixes its own")
    ap.add_argument("--n_ref", type=int, default=5000, help="0 = all val2017")
    ap.add_argument("--min_gen", type=int, default=1000,
                    help="refuse to report a set smaller than this (publication scale)")
    ap.add_argument("--square_ref", type=int, default=1)
    ap.add_argument("--cmmd_tol", type=float, default=2.0,
                    help="pre-registered gate tolerance: pass if CMMD(B4) <= CMMD(B2) + tol")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="phaseC/fidelity_report.md")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    rng = np.random.default_rng(args.seed)
    models = args.models.split(",")
    gate_models = args.gate_models.split(",")
    if len(gate_models) != 2:
        raise ValueError("--gate_models must be treatment,control")
    gate_treatment, gate_control = gate_models
    if gate_treatment not in models or gate_control not in models:
        raise ValueError("--gate_models labels must both appear in --models")
    steps = [int(x) for x in args.steps.split(",")]

    coco = sorted(Path(args.coco_dir).glob("*.jpg"))
    if args.n_ref and args.n_ref < len(coco):
        coco = random.Random(args.seed).sample(coco, args.n_ref)
        coco.sort()
    print(f"reference: {len(coco)} COCO val2017 real images "
          f"({'square-cropped' if args.square_ref else 'as-is'})", flush=True)

    from cleanfid.fid import build_feature_extractor
    ensure_inception_weight()
    incep = build_feature_extractor("clean", device=torch.device(device), use_dataparallel=False)
    if args.cmmd_impl == "reference":
        clip = cmmd_ref.load_clip(device)
        embed = lambda paths: cmmd_ref.clip_embed(paths, clip, device)                  # noqa: E731
        cmmd_label = f"CMMD (reference: {cmmd_ref.CLIP_ID}, crop+bicubic 336, unit-norm, sigma 10)"
    else:
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        proc = CLIPImageProcessor.from_pretrained(args.clip_id)
        clip = CLIPVisionModelWithProjection.from_pretrained(
            args.clip_id, torch_dtype=torch.float16).to(device).eval()
        embed = lambda paths: legacy_clip_embed(paths, clip, proc, device)             # noqa: E731
        cmmd_label = f"unnormalised CLIP RBF-MMD ({args.clip_id} @224, HF processor, NOT CMMD)"
    print(f"[cmmd] {cmmd_label}", flush=True)

    print("embedding reference ...", flush=True)
    ref_i = inception_feats(coco, incep, bool(args.square_ref))
    ref_c = embed([str(p) for p in coco])

    # generated sets
    root = Path(args.gen_root)
    gen_i, gen_c, counts = {}, {}, {}
    for m in models:
        for s in steps:
            files = sorted((root / m).glob(f"p*/s{s}/{args.image_name}"))
            if len(files) < args.min_gen:
                if files:
                    print(f"  SKIP {m}@{s}: only {len(files)} images "
                          f"(< --min_gen {args.min_gen})", flush=True)
                continue
            gen_i[(m, s)] = inception_feats(files, incep, False)
            gen_c[(m, s)] = embed([str(f) for f in files])
            counts[(m, s)] = len(files)
            print(f"  embedded {m}@{s}: {len(files)} images", flush=True)
    if not gen_i:
        sys.exit(f"no generated set under {root} reached --min_gen {args.min_gen}")

    ref_c_dev = ref_c.to(device)
    res = {}
    for key in gen_i:
        m, s = key
        gi, gc = gen_i[key], gen_c[key]
        n = counts[key]
        gc_dev = gc.to(device)

        fid = fid_from_feats(gi, ref_i)
        half = rng.permutation(n)
        fid_h = [fid_from_feats(gi[half[: n // 2]], ref_i),
                 fid_from_feats(gi[half[n // 2:]], ref_i)]
        cm = cmmd(gc_dev, ref_c_dev)
        cm_lo, cm_hi = cmmd_boot_ci(gc_dev, ref_c_dev, rng, args.boot)
        prec, rec = precision_recall(ref_c_dev.float(), gc_dev.float())

        res[f"{m}@{s}"] = {
            "model": m, "steps": s, "n": n,
            "fid": fid, "fid_split_half": fid_h,
            "cmmd": cm, "cmmd_ci": [cm_lo, cm_hi],
            "precision": prec, "recall": rec,
        }
        print(f"  {m}@{s}: FID={fid:.2f} (halves {fid_h[0]:.2f}/{fid_h[1]:.2f})  "
              f"CMMD={cm:.2f} [{cm_lo:.2f},{cm_hi:.2f}]  P={prec:.3f} R={rec:.3f}", flush=True)

    # CMMD to the teacher's own distribution — "did distillation keep the teacher's
    # image distribution", needing no real-image set
    for key in gen_i:
        m, s = key
        if ("base", 28) in gen_c and key != ("base", 28):
            res[f"{m}@{s}"]["cmmd_vs_base28"] = cmmd(gen_c[key].to(device),
                                                     gen_c[("base", 28)].to(device))

    md = [f"# {args.title} — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)\n\n",
          f"reference **{len(coco)}** COCO val2017 real images"
          f"{' (center-cropped square)' if args.square_ref else ''} | prompts = COCO val2017 "
          f"captions, one per image | {cmmd_label} | FID via clean-fid `mode=clean` | "
          f"{args.boot} bootstrap\n\n",
          "Pre-registered role: a **gate**, not a headline — an alignment gain bought by "
          "quality loss or mode collapse is not a pass.\n\n",
          "| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |\n",
          "|---|--:|--:|--:|--:|---|--:|--:|--:|\n"]
    for k, v in res.items():
        vb = v.get("cmmd_vs_base28")
        md.append(f"| {k} | {v['n']} | {v['fid']:.2f} | "
                  f"{v['fid_split_half'][0]:.2f} / {v['fid_split_half'][1]:.2f} | "
                  f"{v['cmmd']:.2f} | [{v['cmmd_ci'][0]:.2f}, {v['cmmd_ci'][1]:.2f}] | "
                  f"{v['precision']:.3f} | {v['recall']:.3f} | "
                  f"{'—' if vb is None else f'{vb:.2f}'} |\n")

    md.append(
        f"\n## Gate: fidelity not worse for {gate_treatment} than {gate_control}\n\n"
    )
    gates = {}
    for s in steps:
        a = res.get(f"{gate_treatment}@{s}")
        b = res.get(f"{gate_control}@{s}")
        if not (a and b):
            continue
        d_cmmd, d_fid = a["cmmd"] - b["cmmd"], a["fid"] - b["fid"]
        ok = d_cmmd <= args.cmmd_tol
        gates[s] = {"d_cmmd": d_cmmd, "d_fid": d_fid, "pass": bool(ok),
                    "d_precision": a["precision"] - b["precision"],
                    "d_recall": a["recall"] - b["recall"]}
        md.append(
                  f"- **{s}-step:** CMMD({gate_treatment})−CMMD({gate_control}) = "
                  f"**{d_cmmd:+.2f}** "
                  f"(tolerance +{args.cmmd_tol:g}) → **{'PASS' if ok else 'FAIL'}**; "
                  f"FID Δ {d_fid:+.2f}; precision Δ {gates[s]['d_precision']:+.3f}; "
                  f"recall Δ {gates[s]['d_recall']:+.3f}\n")
    if not gates:
        md.append(
            f"- {gate_treatment} and/or {gate_control} not present at any requested "
            "step count — gate not evaluated.\n"
        )
    md.append("\n## Read\n"
              "- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.\n"
              "- **recall down** ⇒ narrower support: the diversity failure mode; cross-check "
              "`phaseC/diversity_eval.py`.\n"
              "- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d "
              "are wide); it is a coarse sanity check only, not evidence of quality. See "
              "`phaseC/test_metrics.py`.\n"
              "- Compare model-to-model differences against the FID split-half spread before "
              "reading anything into them.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(md))
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"reference_n": len(coco), "square_ref": bool(args.square_ref), "cmmd": cmmd_label,
         "cmmd_impl": args.cmmd_impl,
         "gate_models": gate_models,
         "results": res, "gate": gates, "cmmd_tol": args.cmmd_tol}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
