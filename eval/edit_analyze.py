#!/usr/bin/env python3
"""Means, dominance search and paired tests for the SoftREPA-style editing tables (docs/editing/).

Per dataset, it computes per (model, setting) means of SoftREPA's five metrics (ImageReward, PickScore, CLIP and
HPS higher; LPIPS lower) and, on PIE, the official PIE-Bench metrics. Then, for every baseline operating point
(naive CD at each setting, and the teacher reference), it lists every setting of ours that is better on ALL FIVE
SoftREPA metrics, with a paired bootstrap over records (10,000 resamples, seed 0) for each metric of each pair.
Records are paired by id.

Per-record scores come from <out>/<dataset>/<model>/scores_<setting>.jsonl (eval/edit_score.py), or, when that tree
is absent, from the committed export <results>/scores_<dataset>.jsonl.gz. --export writes that export.

  python eval/edit_analyze.py [--out out/editing] [--results docs/editing/results] [--export]
"""
from __future__ import annotations

import argparse, gzip, json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SR = [("image_reward", +1), ("pickscore", +1), ("clip", +1), ("hps", +1), ("lpips_sr", -1)]
PIE = [("pie_structure_distance", -1), ("pie_psnr_unedit_part", +1), ("pie_lpips_unedit_part", -1),
       ("pie_mse_unedit_part", -1), ("pie_ssim_unedit_part", +1), ("pie_clip_similarity_target_image", +1),
       ("pie_clip_similarity_target_image_edit_part", +1), ("pie_psnr", +1), ("pie_lpips", -1), ("pie_ssim", +1)]
N_EXPECT = {"pie": 700, "div2k": 800, "cat2dog": 500}
BASELINES = ["naiveS4", "naive118k", "teacher"]
OURS = "ours118k"


def load(out: Path, results: Path, ds: str) -> dict:
    """{(model, setting): {id: row}}"""
    data = defaultdict(dict)
    if (out / ds).is_dir():
        for mdir in sorted((out / ds).iterdir()):
            for f in sorted(mdir.glob("scores_*.jsonl")) if mdir.is_dir() else []:
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    data[(mdir.name, f.stem[len("scores_"):])][r["id"]] = r
    else:
        with gzip.open(results / f"scores_{ds}.jsonl.gz", "rt") as fh:
            for line in fh:
                r = json.loads(line)
                data[(r.pop("model"), r.pop("setting"))][r["id"]] = r
    return dict(data)


def export(data: dict, results: Path, ds: str) -> None:
    results.mkdir(parents=True, exist_ok=True)
    with gzip.open(results / f"scores_{ds}.jsonl.gz", "wt") as fh:
        for (m, tag), rows in sorted(data.items()):
            for rid in sorted(rows):
                fh.write(json.dumps({"model": m, "setting": tag, **rows[rid]}) + "\n")


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi), float(max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0 / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "out/editing")
    ap.add_argument("--results", type=Path, default=REPO / "docs/editing/results")
    ap.add_argument("--export", action="store_true", help="write <results>/scores_<dataset>.jsonl.gz from --out")
    a = ap.parse_args()
    res = {}
    for ds in ["pie", "div2k", "cat2dog"]:
        data = load(a.out, a.results, ds)
        if a.export:
            export(data, a.results, ds)
        means = {}
        for (m, tag), rows in data.items():
            keys = SR + (PIE if ds == "pie" else [])
            means[f"{m}|{tag}"] = {"n": len(rows), **{k: float(np.nanmean([r[k] for r in rows.values()]))
                                                        for k, _ in keys if all(k in r for r in rows.values())}}
        complete = {k for k, v in means.items() if v["n"] == N_EXPECT[ds]}
        dom = []
        for bk in sorted(complete):
            bm, btag = bk.split("|")
            if bm not in BASELINES:
                continue
            for ok in sorted(complete):
                om, otag = ok.split("|")
                if om != OURS or not all(s * (means[ok][k] - means[bk][k]) > 0 for k, s in SR):
                    continue
                A, B = data[(om, otag)], data[(bm, btag)]
                ids = sorted(set(A) & set(B))
                tests = {}
                for k, s in SR:
                    d = s * np.array([A[i][k] - B[i][k] for i in ids], float)
                    lo, hi, p = boot(d)
                    tests[k] = {"delta_better": float(d.mean()), "ci": [lo, hi], "p": p}
                dom.append({"baseline": bk, "ours": ok, "n": len(ids), "tests": tests,
                            "all_significant": all(t["ci"][0] > 0 for t in tests.values())})
        res[ds] = {"means": means, "complete": sorted(complete), "dominating_pairs": dom}
        print(f"\n===== {ds}: {len(means)} arms scored ({len(complete)} complete)")
        for d in dom:
            print(f"  {d['ours']:22s} > {d['baseline']:27s} all 5 significant: {d['all_significant']}")
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
