#!/usr/bin/env python3
"""Is the official-evaluator argmax REAL or is it evaluator noise? (docs/bench/)

For a category, re-score the same N candidates per prompt with an INDEPENDENT judge (VQAScore,
clip-flant5-xxl, the oracle of the COCO caches, no relation to BLIP-VQA / UniDet) and ask:
  - does the official argmax pick also win under the independent judge?
  - how much of the independent judge's own best-of-N headroom does the official argmax recover?
A category whose selection is driven by detector noise scores at chance here (~1/N agreement,
~0 headroom recovery) even though its official headroom is large.

    RANK / WORLD_SIZE shard by prompt.   python eval/bench_independent_check.py --cats numeracy,color
    python eval/bench_independent_check.py --report                      -> <out>/report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pools/bench_train/compbench_prompts.json")
    ap.add_argument("--images", default="cache/bench/images/bench_teacher_k10")
    ap.add_argument("--scores", default="cache/bench/scores")
    ap.add_argument("--label", default="bench_teacher_k10")
    ap.add_argument("--cats", default="numeracy,color,spatial,texture")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--n_prompts", type=int, default=200, help="per category (the first n by idx, deterministic)")
    ap.add_argument("--out", default="out/bench_independent_check")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--report", action="store_true", help="aggregate the rows_rank*.json of a finished run")
    args = ap.parse_args()
    if args.report:
        report(args.out); return
    import torch
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    cats = [c for c in args.cats.split(",") if c]
    pool = {int(r["idx"]): r for r in json.load(open(args.pool))}
    official = {}
    for c in cats:
        f = f"{args.scores}/{args.label}_s{args.K}_{c}/scores.json"
        for r in json.load(open(f))["per_prompt"]:
            official[int(r["idx"])] = [float(x) for x in r["image_scores"]]
    # common/t2v_compat.py stubs the API scorers the vendored t2v_metrics imports unconditionally;
    # it MUST precede `import t2v_metrics`
    sys.path.insert(0, str(ROOT / "third_party" / "t2v_metrics"))
    from common import t2v_compat  # noqa: F401
    import t2v_metrics
    hub_cache = os.path.join(os.environ.get("HF_HOME", str(ROOT / "cache" / "huggingface")), "hub")
    scorer = t2v_metrics.VQAScore(model=args.vqa_model, device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}", cache_dir=hub_cache)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cats:
        idxs = sorted(i for i in official if pool[i]["category"] == c)[:args.n_prompts]
        mine = [i for n, i in enumerate(idxs) if n % world == rank]
        print(f"[r{rank}] {c}: {len(mine)}/{len(idxs)} prompts", flush=True)
        for n, i in enumerate(mine):
            paths = [f"{args.images}/p{i:05d}/s{args.K}/cand{j}.png" for j in range(args.N)]
            if not all(os.path.exists(p) for p in paths):
                continue
            with torch.no_grad():
                v = scorer(images=paths, texts=[pool[i]["prompt"]]).squeeze(1).float().cpu().numpy()
            rows.append({"idx": i, "category": c, "official": official[i], "vqa": v.tolist()})
            if (n + 1) % 25 == 0:
                print(f"[r{rank}] {c} {n + 1}/{len(mine)}", flush=True)
    json.dump(rows, open(out_dir / f"rows_rank{rank}.json", "w"))
    print(f"[r{rank}] wrote {len(rows)} rows", flush=True)


def report(out="out/bench_independent_check") -> None:
    rows = [r for f in glob.glob(f"{out}/rows_rank*.json") for r in json.load(open(f))]
    cats = sorted({r["category"] for r in rows})
    print(f"{'category':12s} {'n':>4s} {'agree':>7s} {'chance':>7s} {'VQA of official pick':>21s} {'VQA mean':>9s} {'VQA best':>9s} {'headroom recovered':>19s}")
    out_j = {}
    for c in cats:
        R = [r for r in rows if r["category"] == c]
        O = np.array([r["official"] for r in R]); V = np.array([r["vqa"] for r in R])
        a = O.argmax(1); n = len(R)
        pick = V[np.arange(n), a]
        rec = (pick.mean() - V.mean()) / max(V.max(1).mean() - V.mean(), 1e-9)
        out_j[c] = {"n": n, "agreement": float(np.mean(a == V.argmax(1))), "vqa_of_official_pick": float(pick.mean()),
                    "vqa_mean": float(V.mean()), "vqa_best": float(V.max(1).mean()), "headroom_recovered": float(rec)}
        print(f"{c:12s} {n:4d} {out_j[c]['agreement']:7.3f} {1/O.shape[1]:7.3f} {pick.mean():21.4f} {V.mean():9.4f} {V.max(1).mean():9.4f} {100*rec:18.1f}%")
    json.dump(out_j, open(f"{out}/report.json", "w"), indent=1)
    print(f"\nwrote {out}/report.json   (agreement ~ chance and ~0% recovery = the official argmax is detector noise)")


if __name__ == "__main__":
    main()
