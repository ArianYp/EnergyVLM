# Placebo minus M1 (job 99955) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | -0.0025 | [-0.0112, +0.0062] | 0.2966 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | -0.0061 | [-0.0199, +0.0079] | 0.2008 |
| **3-in-1** | complex | 300 | -0.0017 | [-0.0070, +0.0038] | 0.2661 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | -0.0013 | [-0.0026, -0.0000] | 0.0249 |
| pooled primary | 7 categories | — | -0.0039 | [-0.0110, +0.0033] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0036** [-0.0132, +0.0200]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | -0.0079 | [-0.0265, +0.0108] |
| color | BLIP-VQA | 300 | -0.0175 | [-0.0313, -0.0043] |
| complex | 3-in-1 | 300 | -0.0017 | [-0.0070, +0.0038] |
| non_spatial | CLIPScore | 300 | -0.0013 | [-0.0026, -0.0000] |
| numeracy | UniDet | 300 | +0.0062 | [-0.0145, +0.0278] |
| shape | BLIP-VQA | 300 | +0.0066 | [-0.0098, +0.0225] |
| spatial | UniDet | 298 | -0.0165 | [-0.0474, +0.0145] |
| texture | BLIP-VQA | 300 | +0.0035 | [-0.0129, +0.0194] |
