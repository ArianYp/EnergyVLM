#!/usr/bin/env python3
"""Three-seed table for the K=10 teacher grid and the grid-A sampler (docs/k10/; converged 3k
schedule, averaged checkpoints): per-seed CompBench / GenEval2, mean +- sd, seed-paired differences
(t over seeds) and the pooled per-prompt sign test. Finds eval dirs by label (latest with
alignment.json) in the experimental tree's phaseN/ records (common/artifacts.py).

    python eval/k10_seeds.py [--eval10] [--artifacts <tree>]    -> docs/k10/SEEDS.md (SEEDS10.md)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.artifacts import add_artifacts_arg, chdir_artifacts  # noqa: E402
from eval.nested_grid import load as load_job, paired  # noqa: E402

ARMS = {  # name -> label template
    "naive K=8": "W_B2-hp1_3k-hp1-avg_s{s}",
    "ours K=8 (paper)": "W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg_s{s}",
    "naive K=10": "W_B2-k10-hp1_3k-hp1-avg_s{s}",
    "ours K=10": "W_CD_dinop_hard-k10-hp1_3k-hp1-avg_s{s}",
    "naive K=10 + grid A": "W_B2-k10-hp1_3k-hp1-avg-gridA_s{s}",
    "ours K=10 + grid A": "W_CD_dinop_hard-k10-hp1_3k-hp1-avg-gridA_s{s}",
    "naive K=8 + grid A": "W_B2-hp1_3k-hp1-avg-gridA_s{s}",
    "ours K=8 + grid A": "W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg-gridA_s{s}",
}
CONTRASTS = [("ours K=10", "naive K=10"), ("ours K=10", "ours K=8 (paper)"), ("naive K=10", "naive K=8"),
             ("ours K=10 + grid A", "ours K=10"), ("naive K=10 + grid A", "naive K=10"), ("ours K=10 + grid A", "naive K=10 + grid A"),
             ("ours K=10 + grid A", "ours K=8 (paper)"), ("ours K=10 + grid A", "naive K=8")]


def find(label, prefix):
    ds = [d for d in glob.glob(f"phaseN/{prefix}_{label}_*") if os.path.exists(f"{d}/alignment.json")]
    ds = [d for d in ds if os.path.basename(d)[len(prefix) + 1:].rsplit("_", 1)[0] == label]
    if not ds:
        return None
    return int(sorted(ds, key=lambda d: int(d.rsplit("_", 1)[1]))[-1].rsplit("_", 1)[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    add_artifacts_arg(ap)
    ap.add_argument("--eval10", action="store_true", help="the official 10-images-per-prompt protocol (eval10_* dirs)")
    ap.add_argument("--out", default=None, help="default docs/k10/SEEDS.md (SEEDS10.md with --eval10)")
    args = ap.parse_args()
    chdir_artifacts(args.artifacts)
    prefix = "eval10" if args.eval10 else "eval"
    out = args.out or str(ROOT / "docs" / "k10" / ("SEEDS10.md" if args.eval10 else "SEEDS.md"))
    data = {}   # arm -> {seed: (cb dict, ge dict, alignment)}
    for arm, tpl in ARMS.items():
        data[arm] = {}
        for s in (0, 1, 2):
            j = find(tpl.format(s=s), prefix)
            if j is not None:
                data[arm][s] = load_job(j)
    L = [f"# K=10 and grid A over seeds (converged 3k schedule, averaged checkpoints; {'official 10 images/prompt' if args.eval10 else '1 image/prompt'})", "",
         "| arm | seeds | CompBench per seed | mean +- sd | GenEval2 per seed | mean +- sd |", "|---|---|---|---|---|---|"]
    for arm, d in data.items():
        if not d:
            continue
        cb = [d[s][2]["compbench_mean"] for s in sorted(d)]; ge = [d[s][2]["geneval2"] for s in sorted(d)]
        L.append(f"| {arm} | {','.join(str(s) for s in sorted(d))} | {' / '.join(f'{x:.4f}' for x in cb)} | {np.mean(cb):.4f} +- {np.std(cb, ddof=1) if len(cb) > 1 else 0:.4f} | "
                 f"{' / '.join(f'{x:.4f}' for x in ge)} | {np.mean(ge):.4f} +- {np.std(ge, ddof=1) if len(ge) > 1 else 0:.4f} |")
    L += ["", "## Contrasts (seed-paired where both seeds exist; pooled per-prompt sign test over the shared seeds)", "",
          "| contrast | seeds | dCB per seed | mean dCB | t-test p (seeds) | pooled win rate | pooled sign p | dGE2 mean |", "|---|---|---|---|---|---|---|---|"]
    from scipy import stats
    for a, b in CONTRASTS:
        ss = sorted(set(data.get(a, {})) & set(data.get(b, {})))
        if not ss:
            continue
        d_cb = [data[a][s][2]["compbench_mean"] - data[b][s][2]["compbench_mean"] for s in ss]
        d_ge = [data[a][s][2]["geneval2"] - data[b][s][2]["geneval2"] for s in ss]
        pooled_a = {(s,) + k: v for s in ss for k, v in data[a][s][0].items()}; pooled_b = {(s,) + k: v for s in ss for k, v in data[b][s][0].items()}
        p = paired(pooled_b, pooled_a)
        tp = stats.ttest_1samp(d_cb, 0).pvalue if len(d_cb) > 1 else float("nan")
        L.append(f"| {a} vs {b} | {','.join(map(str, ss))} | {' / '.join(f'{x:+.4f}' for x in d_cb)} | {np.mean(d_cb):+.4f} | {tp:.3g} | {p['win']:.3f} | {p['sign_p']:.2g} | {np.mean(d_ge):+.4f} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n"); print("\n".join(L)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
