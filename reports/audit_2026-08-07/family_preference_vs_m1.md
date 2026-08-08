# Preference minus M1 (job 99955) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0583 | [+0.0478, +0.0691] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0315 | [+0.0154, +0.0477] | 0.9999 |
| **3-in-1** | complex | 300 | +0.0243 | [+0.0153, +0.0333] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0007 | [-0.0009, +0.0023] | 0.7961 |
| pooled primary | 7 categories | — | +0.0420 | [+0.0335, +0.0504] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0268** [+0.0070, +0.0460]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | -0.0019 | [-0.0195, +0.0160] |
| color | BLIP-VQA | 300 | +0.0461 | [+0.0304, +0.0628] |
| complex | 3-in-1 | 300 | +0.0243 | [+0.0153, +0.0333] |
| non_spatial | CLIPScore | 300 | +0.0007 | [-0.0009, +0.0023] |
| numeracy | UniDet | 300 | +0.0271 | [+0.0032, +0.0514] |
| shape | BLIP-VQA | 300 | +0.0762 | [+0.0548, +0.0979] |
| spatial | UniDet | 298 | +0.0693 | [+0.0319, +0.1080] |
| texture | BLIP-VQA | 300 | +0.0527 | [+0.0345, +0.0711] |
