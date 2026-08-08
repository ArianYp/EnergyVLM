# Preference-logit distribution and beta calibration

All completed Phase-I runs used beta = 100. The loss is `-log sigmoid(-beta * (Delta_theta - Delta_0))`, so the quantity that actually sets the gradient regime is `beta * |Delta_theta - Delta_0|`.

## Where beta = 100 puts the sigmoid

| run | median \|logit\| | frac \|logit\|>4 (saturated) | frac \|logit\|<0.5 (inert) | responsive | mean sigmoid' |
|---|---:|---:|---:|---:|---:|
| `preference` | 0.268 | 0.060 | 0.450 | 0.490 | 0.1914 |
| `placebo` | 0.023 | 0.017 | 0.623 | 0.360 | 0.2185 |
| `preference_repa` | 0.245 | 0.050 | 0.483 | 0.467 | 0.1919 |
| `correct_matched` | 0.285 | 0.077 | 0.423 | 0.500 | 0.1828 |
| `counterfactual` | 0.181 | 0.027 | 0.557 | 0.417 | 0.2049 |

## Scale-free margin and implied beta

| run | median \|gap\| | q75 \|gap\| | beta for median \|logit\|=1 | beta for median \|logit\|=2 | beta for q75 \|logit\|=4 |
|---|---:|---:|---:|---:|---:|
| `preference` | 0.00567 | 0.01376 | 176.3 | 352.7 | 290.7 |
| `placebo` | 0.00315 | 0.00802 | 317.2 | 634.4 | 498.8 |
| `preference_repa` | 0.00580 | 0.01413 | 172.4 | 344.9 | 283.0 |
| `correct_matched` | 0.00667 | 0.01578 | 149.9 | 299.8 | 253.5 |
| `counterfactual` | 0.00424 | 0.00993 | 235.6 | 471.2 | 402.7 |

## Does the loss improve winners or degrade losers?

Both quantities are measured against the frozen M1 reference, so positive values mean the student moved away from M1 in the direction the loss requests.

| run | winner error improvement | loser error increase | winner share | verdict |
|---|---:|---:|---:|---|
| `preference` | -0.00778 | +0.01461 | 0.403 | balanced |
| `placebo` | -0.00408 | +0.00461 | 0.477 | balanced |
| `preference_repa` | -0.00951 | +0.01706 | 0.419 | balanced |
| `correct_matched` | -0.00928 | +0.01834 | 0.431 | balanced |
| `counterfactual` | -0.00467 | +0.00818 | 0.449 | balanced |

## Gradient and drift

| run | grad norm (median) | clipping rate | final L2 drift from M1 | anchor loss (median) |
|---|---:|---:|---:|---:|
| `preference` | 57.5077 | 1.000 | 3.6160 | 0.00870 |
| `placebo` | 68.3274 | 1.000 | 3.5220 | 0.00384 |
| `preference_repa` | 50.4813 | 1.000 | 3.6693 | 0.00771 |
| `correct_matched` | 60.7029 | 1.000 | 3.6295 | 0.00766 |
| `counterfactual` | 62.7748 | 1.000 | 3.5867 | 0.00568 |
