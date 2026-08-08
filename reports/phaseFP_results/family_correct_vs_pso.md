# our recipe minus offline-PSO configuration — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | -0.0183 | [-0.0265, -0.0104] | 0.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0053 | [-0.0105, +0.0206] | 0.7543 |
| **3-in-1** | complex | 300 | -0.0114 | [-0.0191, -0.0041] | 0.0007 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0027 | [+0.0012, +0.0042] | 0.9998 |
| pooled primary | 7 categories | — | -0.0072 | [-0.0149, +0.0003] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **-0.0236** [-0.0412, -0.0061]. The two disjoint families disagree in sign.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0076 | [-0.0112, +0.0267] |
| color | BLIP-VQA | 300 | -0.0085 | [-0.0188, +0.0024] |
| complex | 3-in-1 | 300 | -0.0114 | [-0.0191, -0.0041] |
| non_spatial | CLIPScore | 300 | +0.0027 | [+0.0012, +0.0042] |
| numeracy | UniDet | 300 | +0.0338 | [+0.0088, +0.0579] |
| shape | BLIP-VQA | 300 | -0.0275 | [-0.0445, -0.0111] |
| spatial | UniDet | 298 | -0.0255 | [-0.0597, +0.0086] |
| texture | BLIP-VQA | 300 | -0.0190 | [-0.0334, -0.0050] |
