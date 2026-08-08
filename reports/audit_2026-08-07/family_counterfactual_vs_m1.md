# CounterfactualPreference minus M1 (cross-job, identical prompt manifest) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0383 | [+0.0272, +0.0494] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | -0.0070 | [-0.0211, +0.0070] | 0.1709 |
| **3-in-1** | complex | 300 | +0.0126 | [+0.0046, +0.0208] | 0.9991 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | -0.0007 | [-0.0022, +0.0008] | 0.1633 |
| pooled primary | 7 categories | — | +0.0152 | [+0.0076, +0.0230] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0453** [+0.0275, +0.0634]. The two disjoint families disagree in sign.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | -0.0201 | [-0.0366, -0.0037] |
| color | BLIP-VQA | 300 | +0.0239 | [+0.0048, +0.0432] |
| complex | 3-in-1 | 300 | +0.0126 | [+0.0046, +0.0208] |
| non_spatial | CLIPScore | 300 | -0.0007 | [-0.0022, +0.0008] |
| numeracy | UniDet | 300 | +0.0215 | [-0.0019, +0.0450] |
| shape | BLIP-VQA | 300 | +0.0598 | [+0.0387, +0.0820] |
| spatial | UniDet | 298 | -0.0224 | [-0.0539, +0.0085] |
| texture | BLIP-VQA | 300 | +0.0312 | [+0.0146, +0.0481] |
