# CorrectPreference minus M1 (cross-job, identical prompt manifest) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0789 | [+0.0669, +0.0910] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0620 | [+0.0462, +0.0780] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0288 | [+0.0198, +0.0381] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0013 | [-0.0002, +0.0029] | 0.9519 |
| pooled primary | 7 categories | — | +0.0645 | [+0.0559, +0.0732] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0169** [-0.0031, +0.0368]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0448 | [+0.0245, +0.0657] |
| color | BLIP-VQA | 300 | +0.0603 | [+0.0437, +0.0783] |
| complex | 3-in-1 | 300 | +0.0288 | [+0.0198, +0.0381] |
| non_spatial | CLIPScore | 300 | +0.0013 | [-0.0002, +0.0029] |
| numeracy | UniDet | 300 | +0.0490 | [+0.0227, +0.0755] |
| shape | BLIP-VQA | 300 | +0.1047 | [+0.0812, +0.1289] |
| spatial | UniDet | 298 | +0.0921 | [+0.0582, +0.1278] |
| texture | BLIP-VQA | 300 | +0.0715 | [+0.0515, +0.0918] |
