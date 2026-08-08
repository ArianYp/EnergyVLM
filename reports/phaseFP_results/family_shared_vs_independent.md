# Shared minus independent pair noise (task N endpoint half) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0180 | [+0.0094, +0.0269] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0149 | [-0.0006, +0.0305] | 0.9700 |
| **3-in-1** | complex | 300 | +0.0001 | [-0.0067, +0.0066] | 0.5167 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | -0.0008 | [-0.0021, +0.0005] | 0.1184 |
| pooled primary | 7 categories | — | +0.0141 | [+0.0066, +0.0218] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0031** [-0.0149, +0.0214]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0156 | [-0.0041, +0.0348] |
| color | BLIP-VQA | 300 | +0.0214 | [+0.0081, +0.0359] |
| complex | 3-in-1 | 300 | +0.0001 | [-0.0067, +0.0066] |
| non_spatial | CLIPScore | 300 | -0.0008 | [-0.0021, +0.0005] |
| numeracy | UniDet | 300 | +0.0149 | [-0.0101, +0.0408] |
| shape | BLIP-VQA | 300 | +0.0138 | [-0.0039, +0.0320] |
| spatial | UniDet | 298 | +0.0143 | [-0.0194, +0.0489] |
| texture | BLIP-VQA | 300 | +0.0187 | [+0.0054, +0.0322] |
