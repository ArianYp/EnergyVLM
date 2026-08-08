# Phase-FP random orientation minus M1 — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0141 | [+0.0049, +0.0235] | 0.9979 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0063 | [-0.0074, +0.0204] | 0.8173 |
| **3-in-1** | complex | 300 | -0.0030 | [-0.0087, +0.0027] | 0.1463 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0007 | [-0.0006, +0.0019] | 0.8407 |
| pooled primary | 7 categories | — | +0.0083 | [+0.0011, +0.0157] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0078** [-0.0092, +0.0243]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | -0.0095 | [-0.0275, +0.0091] |
| color | BLIP-VQA | 300 | +0.0062 | [-0.0079, +0.0211] |
| complex | 3-in-1 | 300 | -0.0030 | [-0.0087, +0.0027] |
| non_spatial | CLIPScore | 300 | +0.0007 | [-0.0006, +0.0019] |
| numeracy | UniDet | 300 | +0.0195 | [-0.0032, +0.0431] |
| shape | BLIP-VQA | 300 | +0.0131 | [-0.0050, +0.0312] |
| spatial | UniDet | 298 | +0.0089 | [-0.0204, +0.0377] |
| texture | BLIP-VQA | 300 | +0.0229 | [+0.0072, +0.0386] |
