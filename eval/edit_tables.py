#!/usr/bin/env python3
"""Write the two editing tables of docs/editing/ from per-record scores (no hand-copied numbers).

  Table 1  editing_table_softrepa5.tex  SoftREPA Table 5 layout: PIE-Bench, DIV2K, Cat2Dog x {naive, ours}
  Table 2  editing_table_piebench.tex   SoftREPA Table 2 layout: official PIE-Bench metrics
Also writes <results>/table_pair_means.json with the numbers used.

The pair is naive CD at NMAX 7/8 with standard target CFG 2, against ours at NMAX 5/8 with standard CFG 3; it is
the pair that wins all five SoftREPA metrics, significantly, on all three datasets (analysis.json). The CFG column
uses SoftREPA's convention: their sampler computes v + w (v - v_null) = standard w + 1, so 2 -> 1 and 3 -> 2.

  python eval/edit_tables.py [--out out/editing] [--results docs/editing/results] [--dest docs/editing]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from edit_analyze import load  # noqa: E402  (same loader: out/ tree or the committed .jsonl.gz export)

REPO = Path(__file__).resolve().parents[1]
NAIVE, OURS = "naiveS4", "ours118k"
PAIR = {NAIVE: "T8_n7_s1_t2", OURS: "T8_n5_s1_t3"}
SET = {NAIVE: ("7", "1"), OURS: ("5", "2")}          # NMAX of 8; target CFG in SoftREPA's convention
NAME = {NAIVE: "Naive CD (4-step student)", OURS: "Ours (4-step student)"}
BLOCK = [("pie", "PIE-Bench (700 images)"), ("div2k", "DIV2K (800 images)"), ("cat2dog", "Cat2Dog (500 images)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "out/editing")
    ap.add_argument("--results", type=Path, default=REPO / "docs/editing/results")
    ap.add_argument("--dest", type=Path, default=REPO / "docs/editing")
    a = ap.parse_args()
    M = {}
    for ds, _ in BLOCK:
        data = load(a.out, a.results, ds)
        M[ds] = {}
        for m in (NAIVE, OURS):
            R = data[(m, PAIR[m])]
            keys = [k for k in next(iter(R.values())) if k != "id"]
            M[ds][m] = {"n": len(R), **{k: float(np.nanmean([r[k] for r in R.values()])) for k in keys}}
    (a.results / "table_pair_means.json").write_text(json.dumps(M, indent=1))

    cols = [("image_reward", "{:.3f}"), ("pickscore", "{:.3f}"), ("clip", "{:.3f}"), ("hps", "{:.3f}"),
            ("lpips_sr", "{:.3f}")]
    L = [r"\begin{tabular}{l c c c c c c c}", r"\toprule",
         r"Model & NMAX & \shortstack{target\\CFG} & \shortstack{ImageReward\\$\uparrow$} & "
         r"\shortstack{PickScore\\$\uparrow$} & \shortstack{CLIP\\$\uparrow$} & \shortstack{HPS\\$\uparrow$} & "
         r"\shortstack{LPIPS\\$\downarrow$} \\", r"\midrule"]
    for i, (ds, title) in enumerate(BLOCK):
        if i:
            L.append(r"\midrule")
        L.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        for m in (NAIVE, OURS):
            v = [f.format(M[ds][m][k]) for k, f in cols]
            if m == OURS:
                v = [rf"\textbf{{{x}}}" for x in v]
            L.append(f"{NAME[m]} & {SET[m][0]} & {SET[m][1]} & " + " & ".join(v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (a.dest / "editing_table_softrepa5.tex").write_text("\n".join(L) + "\n")

    p = M["pie"]
    t2 = [("image_reward", 100), ("pickscore", 1), ("hps", 100), ("pie_clip_similarity_target_image_edit_part", 1),
          ("pie_clip_similarity_target_image", 1), ("pie_structure_distance", 1000), ("pie_psnr_unedit_part", 1),
          ("pie_lpips_unedit_part", 100), ("pie_ssim_unedit_part", 100)]
    L = [r"\begin{tabular}{l c c c c c c c c c}", r"\toprule",
         r" & \multicolumn{3}{c}{Human Preference} & \multicolumn{2}{c}{Text Alignment} & Structure & "
         r"\multicolumn{3}{c}{Background Preservation} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(lr){8-10}",
         r"Model & \shortstack{Image-\\Reward $\uparrow$} & \shortstack{Pick-\\Score $\uparrow$} & HPS $\uparrow$ & "
         r"\shortstack{CLIP/\\Edited $\uparrow$} & \shortstack{CLIP/\\Whole $\uparrow$} & Distance $\downarrow$ & "
         r"PSNR $\uparrow$ & LPIPS $\downarrow$ & SSIM $\uparrow$ \\", r"\midrule"]
    for m in (NAIVE, OURS):
        v = [f"{p[m][k] * s:.2f}" for k, s in t2]
        if m == OURS:
            v = [rf"\textbf{{{x}}}" for x in v]
        L.append(f"{NAME[m]} & " + " & ".join(v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (a.dest / "editing_table_piebench.tex").write_text("\n".join(L) + "\n")
    print((a.dest / "editing_table_softrepa5.tex").read_text())
    print((a.dest / "editing_table_piebench.tex").read_text())


if __name__ == "__main__":
    main()
