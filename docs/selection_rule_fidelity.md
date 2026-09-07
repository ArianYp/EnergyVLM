# Selection-rule arms (3k pool, raw finals), per seed — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)

reference **5000** COCO val2017 real images (center-cropped square) | prompts = COCO val2017 captions, one per image | CMMD (reference: openai/clip-vit-large-patch14-336, crop+bicubic 336, unit-norm, sigma 10) | FID via clean-fid `mode=clean` | 1000 bootstrap

Pre-registered role: a **gate**, not a headline — an alignment gain bought by quality loss or mode collapse is not a pass.

| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| S4_B2-avglast3_s0@4 | 5000 | 30.58 | 36.91 / 36.70 | 0.79 | [0.78, 0.80] | 0.507 | 0.056 | — |
| S4_B2-avglast3_s1@4 | 5000 | 30.45 | 36.35 / 37.07 | 0.85 | [0.84, 0.86] | 0.474 | 0.055 | — |
| S4_B2-avglast3_s2@4 | 5000 | 30.24 | 36.61 / 36.41 | 0.87 | [0.86, 0.88] | 0.457 | 0.051 | — |
| S4_B2_s0@4 | 5000 | 32.91 | 39.10 / 39.29 | 0.91 | [0.90, 0.92] | 0.456 | 0.045 | — |
| S4_B2_s1@4 | 5000 | 33.16 | 39.65 / 39.16 | 0.96 | [0.95, 0.96] | 0.436 | 0.046 | — |
| S4_B2_s2@4 | 5000 | 31.17 | 37.61 / 37.39 | 0.97 | [0.96, 0.98] | 0.414 | 0.037 | — |
| S4_CD_dinop_catfreeze-T0.04-avglast3_s0@4 | 5000 | 30.98 | 37.21 / 37.27 | 0.84 | [0.83, 0.85] | 0.478 | 0.062 | — |
| S4_CD_dinop_catfreeze-T0.04-avglast3_s1@4 | 5000 | 31.11 | 37.14 / 37.43 | 0.81 | [0.80, 0.82] | 0.484 | 0.066 | — |
| S4_CD_dinop_catfreeze-T0.04-avglast3_s2@4 | 5000 | 31.05 | 37.38 / 36.98 | 0.84 | [0.83, 0.85] | 0.474 | 0.061 | — |
| S4_CD_dinop_catfreeze-T0.04_s0@4 | 5000 | 32.64 | 38.67 / 39.17 | 0.89 | [0.88, 0.90] | 0.445 | 0.052 | — |
| S4_CD_dinop_catfreeze-T0.04_s1@4 | 5000 | 33.32 | 39.18 / 39.99 | 0.98 | [0.97, 0.99] | 0.444 | 0.039 | — |
| S4_CD_dinop_catfreeze-T0.04_s2@4 | 5000 | 32.37 | 38.83 / 38.32 | 0.90 | [0.89, 0.91] | 0.443 | 0.047 | — |
| S4_CD_dinop_cat-T0.04-avglast3_s0@4 | 5000 | 29.83 | 36.17 / 35.88 | 0.81 | [0.80, 0.82] | 0.484 | 0.062 | — |
| S4_CD_dinop_cat-T0.04-avglast3_s1@4 | 5000 | 30.21 | 36.37 / 36.49 | 0.84 | [0.83, 0.85] | 0.457 | 0.059 | — |
| S4_CD_dinop_cat-T0.04-avglast3_s2@4 | 5000 | 30.86 | 36.82 / 37.20 | 0.81 | [0.80, 0.82] | 0.486 | 0.062 | — |
| S4_CD_dinop_cat-T0.04_s0@4 | 5000 | 32.90 | 39.00 / 39.31 | 0.90 | [0.90, 0.91] | 0.464 | 0.040 | — |
| S4_CD_dinop_cat-T0.04_s1@4 | 5000 | 32.01 | 39.07 / 37.70 | 1.02 | [1.01, 1.03] | 0.434 | 0.040 | — |
| S4_CD_dinop_cat-T0.04_s2@4 | 5000 | 31.40 | 37.92 / 37.45 | 0.93 | [0.92, 0.95] | 0.449 | 0.043 | — |
| S4_CD_dinop_full-T0.04-avglast3_s0@4 | 5000 | 30.37 | 36.85 / 36.39 | 0.79 | [0.78, 0.80] | 0.508 | 0.066 | — |
| S4_CD_dinop_full-T0.04-avglast3_s1@4 | 5000 | 29.81 | 36.13 / 36.07 | 0.80 | [0.80, 0.81] | 0.489 | 0.069 | — |
| S4_CD_dinop_full-T0.04-avglast3_s2@4 | 5000 | 30.75 | 37.03 / 36.83 | 0.79 | [0.78, 0.80] | 0.507 | 0.059 | — |
| S4_CD_dinop_full-T0.04_s0@4 | 5000 | 30.97 | 37.73 / 36.82 | 0.85 | [0.84, 0.86] | 0.488 | 0.058 | — |
| S4_CD_dinop_full-T0.04_s1@4 | 5000 | 29.43 | 35.57 / 35.70 | 0.87 | [0.86, 0.88] | 0.461 | 0.051 | — |
| S4_CD_dinop_full-T0.04_s2@4 | 5000 | 31.22 | 37.01 / 37.78 | 0.85 | [0.84, 0.86] | 0.504 | 0.050 | — |
| S4_CD_dinop_hard-avglast3_s0@4 | 5000 | 30.53 | 36.61 / 36.77 | 0.83 | [0.82, 0.84] | 0.489 | 0.062 | — |
| S4_CD_dinop_hard-avglast3_s1@4 | 5000 | 29.85 | 35.66 / 36.40 | 0.80 | [0.79, 0.81] | 0.495 | 0.067 | — |
| S4_CD_dinop_hard-avglast3_s2@4 | 5000 | 29.99 | 35.79 / 36.53 | 0.77 | [0.77, 0.78] | 0.484 | 0.070 | — |
| S4_CD_dinop_hard_s0@4 | 5000 | 31.90 | 37.99 / 38.45 | 0.89 | [0.89, 0.90] | 0.455 | 0.052 | — |
| S4_CD_dinop_hard_s1@4 | 5000 | 30.57 | 36.96 / 36.66 | 0.94 | [0.93, 0.95] | 0.450 | 0.047 | — |
| S4_CD_dinop_hard_s2@4 | 5000 | 29.62 | 35.57 / 35.94 | 0.83 | [0.82, 0.84] | 0.485 | 0.060 | — |
| T_B4_s0@4 | 5000 | 29.04 | 35.40 / 35.21 | 0.86 | [0.85, 0.87] | 0.483 | 0.058 | — |
| T_B2_s0@4 | 5000 | 33.32 | 39.96 / 39.28 | 0.95 | [0.94, 0.96] | 0.444 | 0.041 | — |

## Gate: fidelity not worse for T_B4_s0 than T_B2_s0

- **4-step:** CMMD(T_B4_s0)−CMMD(T_B2_s0) = **-0.09** (tolerance +2) → **PASS**; FID Δ -4.28; precision Δ +0.039; recall Δ +0.018

## Read
- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.
- **recall down** ⇒ narrower support: the diversity failure mode; cross-check `phaseC/diversity_eval.py`.
- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d are wide); it is a coarse sanity check only, not evidence of quality. See `phaseC/test_metrics.py`.
- Compare model-to-model differences against the FID split-half spread before reading anything into them.
