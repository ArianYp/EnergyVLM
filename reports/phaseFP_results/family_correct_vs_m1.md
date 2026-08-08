# Phase-FP correct fixed-pair arm minus M1 (same prompt manifest) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0774 | [+0.0657, +0.0892] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0588 | [+0.0427, +0.0750] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0265 | [+0.0173, +0.0359] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0011 | [-0.0005, +0.0027] | 0.9103 |
| pooled primary | 7 categories | — | +0.0621 | [+0.0535, +0.0708] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0186** [-0.0014, +0.0385]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0425 | [+0.0231, +0.0630] |
| color | BLIP-VQA | 300 | +0.0610 | [+0.0442, +0.0790] |
| complex | 3-in-1 | 300 | +0.0265 | [+0.0173, +0.0359] |
| non_spatial | CLIPScore | 300 | +0.0011 | [-0.0005, +0.0027] |
| numeracy | UniDet | 300 | +0.0521 | [+0.0263, +0.0780] |
| shape | BLIP-VQA | 300 | +0.0978 | [+0.0751, +0.1211] |
| spatial | UniDet | 298 | +0.0818 | [+0.0458, +0.1182] |
| texture | BLIP-VQA | 300 | +0.0734 | [+0.0537, +0.0936] |
