# Phase-FP correct minus random orientation (identical fixed pairs) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0633 | [+0.0521, +0.0744] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0525 | [+0.0363, +0.0687] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0295 | [+0.0198, +0.0389] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0004 | [-0.0010, +0.0019] | 0.7179 |
| pooled primary | 7 categories | — | +0.0538 | [+0.0452, +0.0624] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0108** [-0.0087, +0.0304]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0520 | [+0.0324, +0.0715] |
| color | BLIP-VQA | 300 | +0.0547 | [+0.0388, +0.0710] |
| complex | 3-in-1 | 300 | +0.0295 | [+0.0198, +0.0389] |
| non_spatial | CLIPScore | 300 | +0.0004 | [-0.0010, +0.0019] |
| numeracy | UniDet | 300 | +0.0326 | [+0.0061, +0.0587] |
| shape | BLIP-VQA | 300 | +0.0847 | [+0.0614, +0.1088] |
| spatial | UniDet | 298 | +0.0728 | [+0.0371, +0.1101] |
| texture | BLIP-VQA | 300 | +0.0505 | [+0.0342, +0.0672] |
