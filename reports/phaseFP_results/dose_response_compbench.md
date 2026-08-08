# Phase-FP fixed-pair dose response

Paired prompt bootstrap, 10000 draws, seed 20260807. Equal-category-weighted over the seven primary CompBench categories.

## Preregistered primary: dose slope

**slope = -0.13972**, 95% CI [-0.15719, -0.12210], P(slope < 0) = 1.0000

Gate (95% interval entirely below zero): **PASS**

Probability of the full ordering correct > counterfactual > random > inverted: **0.9254**

## Absolute score by arm

| arm | dose | CompBench primary | 95% CI |
|---|---:|---:|---|
| `correct` | 0.0000 | 0.55697 | [0.54599, 0.56846] |
| `counterfactual` | 0.1082 | 0.50913 | [0.49808, 0.52024] |
| `random` | 0.2866 | 0.50314 | [0.49175, 0.51460] |
| `inverted` | 0.5729 | 0.46556 | [0.45388, 0.47733] |

## Contrasts against `correct`

| contrast | dose gap | Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| `correct_minus_counterfactual` | 0.1082 | +0.04784 | [+0.03923, +0.05625] | 1.0000 |
| `correct_minus_random` | 0.2866 | +0.05383 | [+0.04520, +0.06235] | 1.0000 |
| `correct_minus_inverted` | 0.5729 | +0.09141 | [+0.08095, +0.10197] | 1.0000 |

## By evaluator family

UniDet shares no architecture with the VQAScore selector that produced the training labels, so its column is the part of any effect not explainable by scorer circularity.

| family | dose slope | 95% CI | P(<0) | correct − counterfactual | 95% CI |
|---|---:|---|---:|---:|---|
| **BLIP-VQA** | -0.20075 | [-0.22691, -0.17447] | 1.0000 | +0.04083 | [+0.03098, +0.05102] |
| **UniDet** | -0.10138 | [-0.13198, -0.07102] | 1.0000 | +0.06599 | [+0.04936, +0.08278] |
| **3-in-1** | -0.07167 | [-0.09046, -0.05261] | 1.0000 | +0.01439 | [+0.00732, +0.02130] |
| **CLIPScore** | -0.01009 | [-0.01356, -0.00661] | 1.0000 | +0.00005 | [-0.00124, +0.00131] |

## Per category (absolute)

| category | n | `correct` | `counterfactual` | `random` | `inverted` |
|---|---:|---:|---:|---:|---:|
| color | 300 | 0.86666 | 0.82268 | 0.81191 | 0.75174 |
| shape | 300 | 0.61506 | 0.57344 | 0.53039 | 0.48539 |
| texture | 300 | 0.76478 | 0.72790 | 0.71429 | 0.64350 |
| spatial | 298 | 0.32011 | 0.22705 | 0.24727 | 0.20983 |
| 3d_spatial | 300 | 0.35869 | 0.30116 | 0.30670 | 0.28878 |
| numeracy | 300 | 0.58581 | 0.53842 | 0.55322 | 0.53453 |
| complex | 300 | 0.38769 | 0.37330 | 0.35821 | 0.34516 |
| non_spatial | 300 | 0.31459 | 0.31454 | 0.31415 | 0.30890 |
