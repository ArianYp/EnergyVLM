# Preference minus Placebo (job 99955) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0608 | [+0.0490, +0.0728] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0376 | [+0.0205, +0.0547] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0260 | [+0.0165, +0.0355] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0020 | [+0.0002, +0.0038] | 0.9864 |
| pooled primary | 7 categories | — | +0.0459 | [+0.0368, +0.0550] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0232** [+0.0023, +0.0438]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0059 | [-0.0150, +0.0262] |
| color | BLIP-VQA | 300 | +0.0636 | [+0.0443, +0.0836] |
| complex | 3-in-1 | 300 | +0.0260 | [+0.0165, +0.0355] |
| non_spatial | CLIPScore | 300 | +0.0020 | [+0.0002, +0.0038] |
| numeracy | UniDet | 300 | +0.0210 | [-0.0038, +0.0464] |
| shape | BLIP-VQA | 300 | +0.0697 | [+0.0480, +0.0919] |
| spatial | UniDet | 298 | +0.0858 | [+0.0457, +0.1266] |
| texture | BLIP-VQA | 300 | +0.0492 | [+0.0293, +0.0694] |
