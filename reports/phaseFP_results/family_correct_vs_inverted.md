# CorrectFixed minus InvertedFixed (Phase-FP) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.1220 | [+0.1071, +0.1370] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0772 | [+0.0583, +0.0960] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0425 | [+0.0317, +0.0535] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0057 | [+0.0037, +0.0077] | 1.0000 |
| pooled primary | 7 categories | — | +0.0914 | [+0.0812, +0.1017] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0448** [+0.0210, +0.0692]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0699 | [+0.0485, +0.0916] |
| color | BLIP-VQA | 300 | +0.1149 | [+0.0893, +0.1421] |
| complex | 3-in-1 | 300 | +0.0425 | [+0.0317, +0.0535] |
| non_spatial | CLIPScore | 300 | +0.0057 | [+0.0037, +0.0077] |
| numeracy | UniDet | 300 | +0.0513 | [+0.0244, +0.0796] |
| shape | BLIP-VQA | 300 | +0.1297 | [+0.1028, +0.1565] |
| spatial | UniDet | 298 | +0.1103 | [+0.0672, +0.1539] |
| texture | BLIP-VQA | 300 | +0.1213 | [+0.0961, +0.1470] |
