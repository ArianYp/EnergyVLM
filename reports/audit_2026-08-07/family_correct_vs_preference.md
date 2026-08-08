# CorrectPreference minus Preference (cross-job, identical prompt manifest) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0205 | [+0.0118, +0.0294] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0305 | [+0.0161, +0.0451] | 0.9999 |
| **3-in-1** | complex | 300 | +0.0045 | [-0.0008, +0.0098] | 0.9528 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0006 | [-0.0007, +0.0019] | 0.8211 |
| pooled primary | 7 categories | — | +0.0225 | [+0.0152, +0.0300] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **-0.0100** [-0.0270, +0.0073]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0467 | [+0.0296, +0.0641] |
| color | BLIP-VQA | 300 | +0.0142 | [+0.0028, +0.0265] |
| complex | 3-in-1 | 300 | +0.0045 | [-0.0008, +0.0098] |
| non_spatial | CLIPScore | 300 | +0.0006 | [-0.0007, +0.0019] |
| numeracy | UniDet | 300 | +0.0219 | [-0.0010, +0.0450] |
| shape | BLIP-VQA | 300 | +0.0285 | [+0.0114, +0.0462] |
| spatial | UniDet | 298 | +0.0229 | [-0.0098, +0.0573] |
| texture | BLIP-VQA | 300 | +0.0189 | [+0.0035, +0.0351] |
