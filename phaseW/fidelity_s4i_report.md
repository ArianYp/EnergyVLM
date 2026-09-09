# Selection-rule arms (3k pool, raw finals), per seed — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)

reference **5000** COCO val2017 real images (center-cropped square) | prompts = COCO val2017 captions, one per image | CMMD (reference: openai/clip-vit-large-patch14-336, crop+bicubic 336, unit-norm, sigma 10) | FID via clean-fid `mode=clean` | 1000 bootstrap

Pre-registered role: a **gate**, not a headline — an alignment gain bought by quality loss or mode collapse is not a pass.

| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| S4_B2-avglast3_s3@4 | 5000 | 29.25 | 35.53 / 35.59 | 0.84 | [0.83, 0.85] | 0.471 | 0.059 | — |
| S4_B2-avglast3_s4@4 | 5000 | 30.75 | 36.63 / 37.45 | 0.88 | [0.87, 0.89] | 0.482 | 0.053 | — |
| S4_B2-rewXi-avglast3_s0@4 | 5000 | 30.29 | 36.81 / 36.28 | 0.73 | [0.73, 0.74] | 0.514 | 0.080 | — |
| S4_B2-rewXi-avglast3_s1@4 | 5000 | 30.57 | 36.55 / 37.05 | 0.76 | [0.75, 0.77] | 0.500 | 0.070 | — |
| S4_B2-rewXi-avglast3_s2@4 | 5000 | 29.74 | 36.18 / 35.72 | 0.78 | [0.77, 0.79] | 0.489 | 0.071 | — |
| S4_B2-rewXi-avglast3_s3@4 | 5000 | 28.89 | 34.99 / 35.33 | 0.72 | [0.72, 0.73] | 0.510 | 0.077 | — |
| S4_B2-rewXi-avglast3_s4@4 | 5000 | 30.65 | 36.96 / 36.71 | 0.76 | [0.76, 0.77] | 0.502 | 0.072 | — |
| S4_CD_dinop_hard-avglast3_s3@4 | 5000 | 28.74 | 34.96 / 34.80 | 0.79 | [0.78, 0.80] | 0.489 | 0.069 | — |
| S4_CD_dinop_hard-avglast3_s4@4 | 5000 | 30.86 | 37.33 / 36.69 | 0.78 | [0.77, 0.79] | 0.526 | 0.062 | — |
| S4_CD_dinop_hard-rewXi-avglast3_s0@4 | 5000 | 30.08 | 35.95 / 36.65 | 0.74 | [0.73, 0.75] | 0.525 | 0.081 | — |
| S4_CD_dinop_hard-rewXi-avglast3_s1@4 | 5000 | 30.08 | 35.87 / 36.71 | 0.73 | [0.72, 0.73] | 0.512 | 0.083 | — |
| S4_CD_dinop_hard-rewXi-avglast3_s2@4 | 5000 | 30.22 | 36.33 / 36.39 | 0.70 | [0.69, 0.71] | 0.529 | 0.083 | — |
| S4_CD_dinop_hard-rewXi-avglast3_s3@4 | 5000 | 28.23 | 34.63 / 34.10 | 0.72 | [0.71, 0.73] | 0.511 | 0.089 | — |
| S4_CD_dinop_hard-rewXi-avglast3_s4@4 | 5000 | 30.69 | 36.92 / 36.75 | 0.74 | [0.73, 0.74] | 0.531 | 0.075 | — |
| T_B4_s0@4 | 5000 | 29.04 | 35.09 / 35.54 | 0.86 | [0.85, 0.87] | 0.483 | 0.058 | — |
| T_B2_s0@4 | 5000 | 33.32 | 39.61 / 39.56 | 0.95 | [0.94, 0.96] | 0.444 | 0.041 | — |

## Gate: fidelity not worse for T_B4_s0 than T_B2_s0

- **4-step:** CMMD(T_B4_s0)−CMMD(T_B2_s0) = **-0.09** (tolerance +2) → **PASS**; FID Δ -4.28; precision Δ +0.039; recall Δ +0.018

## Read
- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.
- **recall down** ⇒ narrower support: the diversity failure mode; cross-check `phaseC/diversity_eval.py`.
- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d are wide); it is a coarse sanity check only, not evidence of quality. See `phaseC/test_metrics.py`.
- Compare model-to-model differences against the FID split-half spread before reading anything into them.
