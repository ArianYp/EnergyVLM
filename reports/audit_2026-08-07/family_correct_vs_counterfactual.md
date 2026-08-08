# Correct minus counterfactual pair construction (job 100593) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0405 | [+0.0301, +0.0513] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0689 | [+0.0524, +0.0860] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0161 | [+0.0091, +0.0231] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0021 | [+0.0006, +0.0035] | 0.9980 |
| pooled primary | 7 categories | — | +0.0492 | [+0.0406, +0.0579] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **-0.0284** [-0.0488, -0.0086]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0648 | [+0.0457, +0.0841] |
| color | BLIP-VQA | 300 | +0.0364 | [+0.0195, +0.0547] |
| complex | 3-in-1 | 300 | +0.0161 | [+0.0091, +0.0231] |
| non_spatial | CLIPScore | 300 | +0.0021 | [+0.0006, +0.0035] |
| numeracy | UniDet | 300 | +0.0275 | [+0.0015, +0.0535] |
| shape | BLIP-VQA | 300 | +0.0449 | [+0.0249, +0.0647] |
| spatial | UniDet | 298 | +0.1145 | [+0.0766, +0.1549] |
| texture | BLIP-VQA | 300 | +0.0403 | [+0.0229, +0.0580] |
