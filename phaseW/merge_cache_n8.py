#!/usr/bin/env python3
"""Merge an N=4 candidate cache with its j=4..7 extension into an N=8 manifest.

The extension was generated with the same seed convention (manual_seed(seed + idx*1000 + j)), so
candidate j of the merged record re-rolls to the same image the scorer saw. Per-candidate lists
(scores) are concatenated in j order; argmax fields are recomputed over 8; `random_idx` is kept from
the ORIGINAL record (a draw among the first 4) so the naive arm trained on the N=4 cache remains
the exact control for an N=8 scored arm -- the naive arm is not re-run. Records missing in either
side are dropped and counted.

    python phaseW/merge_cache_n8.py --base phaseN/coco_selection_118k \
        --ext phaseN/coco_selection_118k_j4to7 --out phaseN/coco_selection_118k_n8
"""
from __future__ import annotations

import argparse, glob, hashlib, json
from pathlib import Path

import numpy as np


def load(d: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"{d}/selection_rank*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); out[int(r["idx"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--ext", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    base, ext = load(args.base), load(args.ext)
    n_base = int(next(iter(base.values()))["N"]); e0 = next(iter(ext.values()))
    j_start, n_tot = int(e0.get("j_start", n_base)), int(e0["N"])
    assert j_start == n_base and n_tot == 2 * n_base, (j_start, n_base, n_tot)
    list_keys = [k for k, v in e0.items() if isinstance(v, list)]
    merged, missing = [], 0
    for idx in sorted(base):
        if idx not in ext:
            missing += 1; continue
        b, e = base[idx], ext[idx]
        assert b["seed_base"] == e["seed_base"] and b["prompt"] == e["prompt"], idx
        r = dict(b); r["N"] = n_tot
        for k in list_keys:
            if k in b and isinstance(b[k], list):
                r[k] = list(b[k]) + list(e[k])
        for k in list(r):
            if k.endswith("_argmax_idx"):
                base_key = k[: -len("_argmax_idx")]
                if base_key in r and isinstance(r[base_key], list):
                    r[k] = int(np.nanargmax(np.asarray(r[base_key], dtype=float)))
        if "endpoint_vqa" in r and np.all(np.isfinite(r["endpoint_vqa"])):
            r["oracle_idx"] = int(np.argmax(r["endpoint_vqa"]))
        # random_idx deliberately kept from the base record (a draw among the first 4)
        merged.append(r)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for s in range(args.shards):
        with open(out / f"selection_rank{s}.jsonl", "w") as fh:
            for k, r in enumerate(merged):
                if k % args.shards == s:
                    fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(json.dumps([[r["idx"], r.get("dino_patch_cos_argmax_idx")] for r in merged], sort_keys=True).encode()).hexdigest()
    meta = {"n": len(merged), "N": n_tot, "base": args.base, "ext": args.ext, "missing_in_ext": missing,
            "random_idx": "kept from base (uniform over the first 4)", "dino_patch_argmax_manifest_sha256": sha}
    # offline headroom: best-of-8 vs best-of-4 under VQAScore, and DINO-patch recovery on both
    VQ = np.array([r["endpoint_vqa"] for r in merged if "endpoint_vqa" in r and np.all(np.isfinite(r["endpoint_vqa"]))])
    if len(VQ):
        rows = np.arange(len(VQ)); rnd = np.array([r["random_idx"] for r in merged if "endpoint_vqa" in r and np.all(np.isfinite(r["endpoint_vqa"]))])
        v_rand = VQ[rows, rnd].mean(); o4, o8 = VQ[:, :n_base].max(1).mean(), VQ.max(1).mean()
        DP = np.array([r["dino_patch_cos"] for r in merged if "endpoint_vqa" in r and np.all(np.isfinite(r["endpoint_vqa"]))])
        p4, p8 = DP[:, :n_base].argmax(1), DP.argmax(1)
        meta["headroom"] = {"random": float(v_rand), "oracle4": float(o4), "oracle8": float(o8),
                            "dino_pick4_vqa": float(VQ[rows, p4].mean()), "dino_pick8_vqa": float(VQ[rows, p8].mean()),
                            "dino4_pct_of_oracle4": float(100 * (VQ[rows, p4].mean() - v_rand) / (o4 - v_rand)),
                            "dino8_pct_of_oracle8": float(100 * (VQ[rows, p8].mean() - v_rand) / (o8 - v_rand))}
    json.dump(meta, open(out / "cache_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
