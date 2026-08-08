# PreferenceREPA minus Preference (job 99955) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0169 | [+0.0093, +0.0248] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0017 | [-0.0122, +0.0159] | 0.5937 |
| **3-in-1** | complex | 300 | +0.0025 | [-0.0032, +0.0084] | 0.8013 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | -0.0006 | [-0.0020, +0.0008] | 0.1980 |
| pooled primary | 7 categories | — | +0.0083 | [+0.0015, +0.0154] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0153** [-0.0009, +0.0315]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0258 | [+0.0101, +0.0427] |
| color | BLIP-VQA | 300 | +0.0101 | [-0.0022, +0.0230] |
| complex | 3-in-1 | 300 | +0.0025 | [-0.0032, +0.0084] |
| non_spatial | CLIPScore | 300 | -0.0006 | [-0.0020, +0.0008] |
| numeracy | UniDet | 300 | +0.0024 | [-0.0201, +0.0239] |
| shape | BLIP-VQA | 300 | +0.0166 | [+0.0021, +0.0317] |
| spatial | UniDet | 298 | -0.0233 | [-0.0564, +0.0087] |
| texture | BLIP-VQA | 300 | +0.0241 | [+0.0110, +0.0384] |
