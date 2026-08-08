# InvertedFixed minus M1Fixed (Phase-FP) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | -0.0446 | [-0.0561, -0.0331] | 0.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | -0.0184 | [-0.0321, -0.0047] | 0.0046 |
| **3-in-1** | complex | 300 | -0.0160 | [-0.0232, -0.0089] | 0.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | -0.0046 | [-0.0063, -0.0030] | 0.0000 |
| pooled primary | 7 categories | — | -0.0293 | [-0.0370, -0.0218] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **-0.0262** [-0.0441, -0.0080]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | -0.0274 | [-0.0435, -0.0108] |
| color | BLIP-VQA | 300 | -0.0540 | [-0.0748, -0.0349] |
| complex | 3-in-1 | 300 | -0.0160 | [-0.0232, -0.0089] |
| non_spatial | CLIPScore | 300 | -0.0046 | [-0.0063, -0.0030] |
| numeracy | UniDet | 300 | +0.0008 | [-0.0225, +0.0242] |
| shape | BLIP-VQA | 300 | -0.0319 | [-0.0520, -0.0114] |
| spatial | UniDet | 298 | -0.0285 | [-0.0583, +0.0011] |
| texture | BLIP-VQA | 300 | -0.0479 | [-0.0676, -0.0296] |
