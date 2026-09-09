# Selection-rule arms (3k pool, raw finals), per seed — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)

reference **5000** COCO val2017 real images (center-cropped square) | prompts = COCO val2017 captions, one per image | CMMD (reference: openai/clip-vit-large-patch14-336, crop+bicubic 336, unit-norm, sigma 10) | FID via clean-fid `mode=clean` | 1000 bootstrap

Pre-registered role: a **gate**, not a headline — an alignment gain bought by quality loss or mode collapse is not a pass.

| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| S4_CD_dinop_hard-rewRi-e25-avglast3_s0@4 | 5000 | 31.08 | 37.32 / 37.16 | 0.82 | [0.82, 0.83] | 0.481 | 0.066 | — |
| S4_CD_dinop_hard-rewRi-e25-avglast3_s1@4 | 5000 | 30.82 | 36.49 / 37.60 | 0.86 | [0.85, 0.87] | 0.464 | 0.058 | — |
| S4_CD_dinop_hard-rewRi-e25-avglast3_s2@4 | 5000 | 30.81 | 37.37 / 36.77 | 0.82 | [0.81, 0.83] | 0.474 | 0.064 | — |
| S4_CD_dinop_hard-rewRi-s16-avglast3_s0@4 | 5000 | 31.14 | 37.40 / 37.33 | 0.79 | [0.78, 0.80] | 0.489 | 0.068 | — |
| S4_CD_dinop_hard-rewRi-s16-avglast3_s1@4 | 5000 | 30.75 | 36.94 / 36.94 | 0.84 | [0.83, 0.85] | 0.467 | 0.066 | — |
| S4_CD_dinop_hard-rewRi-s16-avglast3_s2@4 | 5000 | 30.78 | 36.77 / 37.08 | 0.82 | [0.81, 0.83] | 0.467 | 0.068 | — |
| S4_CD_dinop_hard-rewXi-bf16-avglast3_s0@4 | 5000 | 29.71 | 36.09 / 35.82 | 0.73 | [0.72, 0.74] | 0.527 | 0.086 | — |
| S4_CD_dinop_hard-rewXi-bf16-avglast3_s1@4 | 5000 | 29.79 | 35.93 / 36.04 | 0.72 | [0.71, 0.72] | 0.510 | 0.078 | — |
| S4_CD_dinop_hard-rewXi-bf16-avglast3_s2@4 | 5000 | 30.09 | 36.48 / 36.05 | 0.71 | [0.70, 0.71] | 0.525 | 0.081 | — |
| S4_CD_dinop_hard-rewXi-bilinear-avglast3_s0@4 | 5000 | 30.34 | 36.23 / 36.77 | 0.73 | [0.72, 0.74] | 0.518 | 0.082 | — |
| S4_CD_dinop_hard-rewXi-bilinear-avglast3_s1@4 | 5000 | 29.98 | 36.03 / 36.51 | 0.72 | [0.71, 0.73] | 0.513 | 0.090 | — |
| S4_CD_dinop_hard-rewXi-bilinear-avglast3_s2@4 | 5000 | 30.14 | 36.30 / 36.25 | 0.72 | [0.71, 0.73] | 0.517 | 0.085 | — |
| S4_CD_dinop_hard-rewXi-l31-avglast3_s0@4 | 5000 | 30.49 | 36.79 / 36.55 | 0.68 | [0.67, 0.69] | 0.548 | 0.089 | — |
| S4_CD_dinop_hard-rewXi-l31-avglast3_s1@4 | 5000 | 30.52 | 36.66 / 36.75 | 0.69 | [0.68, 0.70] | 0.512 | 0.094 | — |
| S4_CD_dinop_hard-rewXi-l31-avglast3_s2@4 | 5000 | 30.87 | 36.89 / 37.11 | 0.69 | [0.68, 0.69] | 0.541 | 0.091 | — |
| S4_CD_dinop_hard-rewXi-l62-avglast3_s0@4 | 5000 | 30.60 | 36.83 / 36.77 | 0.64 | [0.64, 0.65] | 0.570 | 0.103 | — |
| S4_CD_dinop_hard-rewXi-l62-avglast3_s1@4 | 5000 | 31.44 | 38.04 / 37.18 | 0.62 | [0.61, 0.63] | 0.553 | 0.108 | — |
| S4_CD_dinop_hard-rewXi-l62-avglast3_s2@4 | 5000 | 32.15 | 38.44 / 38.13 | 0.64 | [0.64, 0.65] | 0.559 | 0.091 | — |
| S4_CD_dinop_hard-rewXi-l7.75-avglast3_s0@4 | 5000 | 29.92 | 36.51 / 35.85 | 0.76 | [0.75, 0.76] | 0.500 | 0.076 | — |
| S4_CD_dinop_hard-rewXi-l7.75-avglast3_s1@4 | 5000 | 29.92 | 36.20 / 36.12 | 0.76 | [0.75, 0.77] | 0.494 | 0.076 | — |
| S4_CD_dinop_hard-rewXi-l7.75-avglast3_s2@4 | 5000 | 30.24 | 36.36 / 36.51 | 0.76 | [0.76, 0.77] | 0.501 | 0.074 | — |
| S4_CD_dinop_hard-rewXi-noisiest-avglast3_s0@4 | 5000 | 30.96 | 37.73 / 36.45 | 0.69 | [0.69, 0.70] | 0.533 | 0.085 | — |
| S4_CD_dinop_hard-rewXi-noisiest-avglast3_s1@4 | 5000 | 31.17 | 37.42 / 37.00 | 0.71 | [0.71, 0.72] | 0.514 | 0.087 | — |
| S4_CD_dinop_hard-rewXi-noisiest-avglast3_s2@4 | 5000 | 32.49 | 38.31 / 38.84 | 0.69 | [0.68, 0.70] | 0.545 | 0.086 | — |
| S4_CD_dinop_hard-rewXi-R1-avglast3_s0@4 | 5000 | 30.33 | 36.21 / 36.63 | 0.77 | [0.76, 0.78] | 0.499 | 0.080 | — |
| S4_CD_dinop_hard-rewXi-R1-avglast3_s1@4 | 5000 | 30.06 | 35.62 / 36.85 | 0.75 | [0.74, 0.76] | 0.498 | 0.081 | — |
| S4_CD_dinop_hard-rewXi-R1-avglast3_s2@4 | 5000 | 29.97 | 35.81 / 36.40 | 0.72 | [0.71, 0.72] | 0.508 | 0.086 | — |
| S4_CD_dinop_hard-rewXi-R5-avglast3_s0@4 | 5000 | 30.61 | 37.02 / 36.55 | 0.70 | [0.70, 0.71] | 0.525 | 0.088 | — |
| S4_CD_dinop_hard-rewXi-R5-avglast3_s1@4 | 5000 | 30.76 | 36.56 / 37.35 | 0.70 | [0.69, 0.70] | 0.513 | 0.093 | — |
| S4_CD_dinop_hard-rewXi-R5-avglast3_s2@4 | 5000 | 30.94 | 37.19 / 36.94 | 0.68 | [0.68, 0.69] | 0.531 | 0.086 | — |
| T_B4_s0@4 | 5000 | 29.04 | 35.40 / 35.21 | 0.86 | [0.85, 0.87] | 0.483 | 0.058 | — |
| T_B2_s0@4 | 5000 | 33.32 | 39.96 / 39.28 | 0.95 | [0.94, 0.96] | 0.444 | 0.041 | — |

## Gate: fidelity not worse for T_B4_s0 than T_B2_s0

- **4-step:** CMMD(T_B4_s0)−CMMD(T_B2_s0) = **-0.09** (tolerance +2) → **PASS**; FID Δ -4.28; precision Δ +0.039; recall Δ +0.018

## Read
- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.
- **recall down** ⇒ narrower support: the diversity failure mode; cross-check `phaseC/diversity_eval.py`.
- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d are wide); it is a coarse sanity check only, not evidence of quality. See `phaseC/test_metrics.py`.
- Compare model-to-model differences against the FID split-half spread before reading anything into them.
