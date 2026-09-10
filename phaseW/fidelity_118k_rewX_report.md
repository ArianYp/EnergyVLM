# 118k scale: random vs argmax vs argmax+exact-reward, 3 seeds, weight-averaged — publication-scale fidelity (FID + CMMD vs COCO-val2017 real)

reference **5000** COCO val2017 real images (center-cropped square) | prompts = COCO val2017 captions, one per image | CMMD (reference: openai/clip-vit-large-patch14-336, crop+bicubic 336, unit-norm, sigma 10) | FID via clean-fid `mode=clean` | 1000 bootstrap

Pre-registered role: a **gate**, not a headline — an alignment gain bought by quality loss or mode collapse is not a pass.

| model@steps | n | FID ↓ | FID split-half | CMMD ↓ | CMMD 95% CI | precision ↑ | recall ↑ | CMMD vs base@28 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| W_B2_118k-avglast5_s0@4 | 5000 | 31.08 | 37.50 / 37.21 | 0.84 | [0.83, 0.85] | 0.485 | 0.059 | — |
| W_B2_118k-avglast5_s1@4 | 5000 | 30.97 | 36.83 / 37.84 | 0.84 | [0.83, 0.85] | 0.489 | 0.066 | — |
| W_B2_118k-avglast5_s2@4 | 5000 | 31.44 | 37.76 / 37.74 | 0.83 | [0.83, 0.84] | 0.503 | 0.062 | — |
| W_CD_dinop_hard_118k-avglast5_s0@4 | 5000 | 30.98 | 37.08 / 37.51 | 0.78 | [0.78, 0.79] | 0.522 | 0.071 | — |
| W_CD_dinop_hard_118k-avglast5_s1@4 | 5000 | 30.83 | 37.05 / 37.11 | 0.79 | [0.78, 0.80] | 0.511 | 0.068 | — |
| W_CD_dinop_hard_118k-avglast5_s2@4 | 5000 | 31.47 | 37.43 / 37.94 | 0.78 | [0.77, 0.78] | 0.522 | 0.063 | — |
| W_CD_dinop_hard_118k-rewX-avglast5_s0@4 | 5000 | 30.45 | 36.57 / 36.76 | 0.69 | [0.68, 0.70] | 0.559 | 0.085 | — |
| W_CD_dinop_hard_118k-rewX-avglast5_s1@4 | 5000 | 30.66 | 36.63 / 36.99 | 0.68 | [0.68, 0.69] | 0.572 | 0.091 | — |
| W_CD_dinop_hard_118k-rewX-avglast5_s2@4 | 5000 | 31.25 | 37.66 / 37.12 | 0.68 | [0.67, 0.69] | 0.558 | 0.086 | — |

## Gate: fidelity not worse for W_CD_dinop_hard_118k-rewX-avglast5_s0 than W_CD_dinop_hard_118k-avglast5_s0

- **4-step:** CMMD(W_CD_dinop_hard_118k-rewX-avglast5_s0)−CMMD(W_CD_dinop_hard_118k-avglast5_s0) = **-0.09** (tolerance +2) → **PASS**; FID Δ -0.52; precision Δ +0.038; recall Δ +0.014

## Read
- **CMMD/FID up** ⇒ quality loss. FID/CMMD, not precision, is the quality signal.
- **recall down** ⇒ narrower support: the diversity failure mode; cross-check `phaseC/diversity_eval.py`.
- **precision saturates near 1** at this feature dimension (k-NN radii in 768-d are wide); it is a coarse sanity check only, not evidence of quality. See `phaseC/test_metrics.py`.
- Compare model-to-model differences against the FID split-half spread before reading anything into them.
