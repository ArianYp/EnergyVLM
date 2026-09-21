# Nested grid A vs scheduler grid (converged 3k averaged students, seed 0) — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)

reference **5000** COCO val2017 real images (center-cropped square) | prompts = COCO val2017 captions, one per image | CMMD (reference: openai/clip-vit-large-patch14-336, crop+bicubic 336, unit-norm, sigma 10) | FID via clean-fid `mode=clean` | 1000 bootstrap

Pre-registered role: a **gate**, not a headline — an alignment gain bought by quality loss or mode collapse is not a pass.

| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| W_B2-hp1_3k-avg_s0@4 | 5000 | 28.63 | 34.84 / 34.75 | 0.64 | [0.64, 0.65] | 0.537 | 0.112 | — |
| W_B2-hp1_3k-avg-gridA_s0@4 | 5000 | 27.99 | 34.05 / 34.35 | 0.58 | [0.58, 0.59] | 0.582 | 0.140 | — |
| W_CD_dinop_hard-rewRi-s16-hp1_3k-avg_s0@4 | 5000 | 28.45 | 34.70 / 34.72 | 0.67 | [0.66, 0.67] | 0.526 | 0.111 | — |
| W_CD_dinop_hard-rewRi-s16-hp1_3k-avg-gridA_s0@4 | 5000 | 28.01 | 33.78 / 34.65 | 0.59 | [0.59, 0.60] | 0.575 | 0.132 | — |
| T_B4_s0@4 | 5000 | 29.04 | 35.68 / 35.08 | 0.86 | [0.85, 0.87] | 0.483 | 0.058 | — |
| T_B2_s0@4 | 5000 | 33.32 | 39.34 / 39.85 | 0.95 | [0.94, 0.96] | 0.444 | 0.041 | — |

## Gate: fidelity not worse for T_B4_s0 than T_B2_s0

- **4-step:** CMMD(T_B4_s0)−CMMD(T_B2_s0) = **-0.09** (tolerance +2) → **PASS**; FID Δ -4.28; precision Δ +0.039; recall Δ +0.018

## Read
- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.
- **recall down** ⇒ narrower support: the diversity failure mode; cross-check `phaseC/diversity_eval.py`.
- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d are wide); it is a coarse sanity check only, not evidence of quality. See `phaseC/test_metrics.py`.
- Compare model-to-model differences against the FID split-half spread before reading anything into them.
