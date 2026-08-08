# PreferenceREPA minus M1 (job 99955) — CompBench by evaluator family

Paired prompt bootstrap, 10000 draws, seed 20260807. Prompts are resampled within category; the family effect is the equal-weighted mean of its category means.

| evaluator family | categories | prompts | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **BLIP-VQA** | color, shape, texture | 900 | +0.0753 | [+0.0646, +0.0864] | 1.0000 |
| **UniDet** | spatial, 3d_spatial, numeracy | 898 | +0.0332 | [+0.0182, +0.0482] | 1.0000 |
| **3-in-1** | complex | 300 | +0.0267 | [+0.0176, +0.0360] | 1.0000 |
| **CLIPScore** *(secondary)* | non_spatial | 300 | +0.0001 | [-0.0016, +0.0018] | 0.5287 |
| pooled primary | 7 categories | — | +0.0503 | [+0.0422, +0.0583] | — |

## Architecture disagreement

BLIP-VQA minus UniDet: **+0.0421** [+0.0236, +0.0605]. Both disjoint families point the same way.

The UniDet family shares no architecture with the VQAScore selector that produced the training labels, so a same-signed UniDet effect is the part of this contrast that is not explainable by scorer circularity.

## Per category

| category | family | n | Δ | 95% CI |
|---|---|---:|---:|---|
| 3d_spatial | UniDet | 300 | +0.0239 | [+0.0058, +0.0423] |
| color | BLIP-VQA | 300 | +0.0562 | [+0.0395, +0.0742] |
| complex | 3-in-1 | 300 | +0.0267 | [+0.0176, +0.0360] |
| non_spatial | CLIPScore | 300 | +0.0001 | [-0.0016, +0.0018] |
| numeracy | UniDet | 300 | +0.0296 | [+0.0049, +0.0537] |
| shape | BLIP-VQA | 300 | +0.0928 | [+0.0711, +0.1148] |
| spatial | UniDet | 298 | +0.0460 | [+0.0136, +0.0799] |
| texture | BLIP-VQA | 300 | +0.0768 | [+0.0597, +0.0947] |
