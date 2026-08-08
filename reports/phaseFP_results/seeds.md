# Fixed-pair dose response across training seeds

Seeds analyzed: s1, s2, s3

Per-seed intervals are **prompt** bootstraps. The across-seed row describes **training**
randomness. They are different sources of variation and are not combined.

## Dose slope (preregistered primary)

| seed | slope | 95% CI (prompts) | ordering probability |
|---|---:|---|---:|
| s1 | -0.13972 | [-0.15731, -0.12259] | 0.9254 |
| s2 | -0.09591 | [-0.11090, -0.08038] | 0.9968 |
| s3 | -0.13591 | [-0.15175, -0.12004] | 0.9998 |
| **across seeds** | **-0.12385** | range [-0.13972, -0.09591] | 0.9740 |

All seeds same sign: **True**. Across-seed range 0.04381.

## Absolute score by arm and seed

| seed | `correct` | `counterfactual` | `random` | `inverted` |
|---|---:|---:|---:|---:|
| s1 | 0.55697 | 0.50913 | 0.50314 | 0.46556 |
| s2 | 0.54191 | 0.50983 | 0.49894 | 0.48028 |
| s3 | 0.55947 | 0.51884 | 0.50435 | 0.47330 |

## Contrasts against `correct`

| contrast | s1 | s2 | s3 | mean | range | same sign |
|---|---:|---:|---:|---:|---:|:-:|
| `correct_minus_counterfactual` | +0.04784 | +0.03208 | +0.04063 | +0.04018 | 0.01576 | yes |
| `correct_minus_inverted` | +0.09141 | +0.06163 | +0.08616 | +0.07973 | 0.02978 | yes |
| `correct_minus_random` | +0.05383 | +0.04297 | +0.05511 | +0.05064 | 0.01214 | yes |

## Dose slope by evaluator family

| family | s1 | s2 | s3 | mean | same sign |
|---|---:|---:|---:|---:|:-:|
| BLIP-VQA | -0.20075 | -0.14837 | -0.18195 | -0.17702 | yes |
| UniDet | -0.10138 | -0.05752 | -0.11352 | -0.09081 | yes |
| 3-in-1 | -0.07167 | -0.05371 | -0.06496 | -0.06344 | yes |
