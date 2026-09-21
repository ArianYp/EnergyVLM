# Performance tables (2026-09-17)

All 4-step students of SD3.5-Medium, 3k COCO captions, sampled at guidance 1 on the scheduler grid; CompBench = mean of 8 categories (300 held-out prompts each, one image per prompt); GenEval2 = Soft-TIFA geometric mean (800 prompts). Deltas are BEST minus the row. Old schedule = constant LR 2.8e-5, 6,000 updates; converged = cosine LR 1e-5, batch 16, 3,000 updates, window K={4..7} (K=8) or {6..9} (K=10).

## Summary

| model | seeds | CompBench | d vs BEST | GenEval2 | d vs BEST |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 5 | 0.4533 | +0.0417 | 0.2110 | +0.0147 |
| naive CD, old schedule, avg last 3 | 5 | 0.4751 | +0.0199 | 0.2247 | +0.0010 |
| naive CD, converged schedule, K=8, avg | 1 | 0.4860 | +0.0091 | 0.2296 | -0.0039 |
| scored + projector reward, K=8, avg (paper) | 1 | 0.4877 | +0.0074 | 0.2296 | -0.0039 |
| BEST: scored + projector reward, K=10, avg | 1 | 0.4951 | +0.0000 | 0.2257 | +0.0000 |

## T2I-CompBench per category (absolute)

| model | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.7803 | 0.5342 | 0.6920 | 0.1743 | 0.2682 | 0.5116 | 0.3080 | 0.3583 |
| naive CD, old schedule, avg last 3 | 0.8086 | 0.5517 | 0.7151 | 0.2110 | 0.3030 | 0.5313 | 0.3104 | 0.3700 |
| naive CD, converged schedule, K=8, avg | 0.8074 | 0.5589 | 0.7155 | 0.2411 | 0.3318 | 0.5465 | 0.3119 | 0.3748 |
| scored + projector reward, K=8, avg (paper) | 0.8114 | 0.5689 | 0.7225 | 0.2136 | 0.3279 | 0.5621 | 0.3125 | 0.3825 |
| BEST: scored + projector reward, K=10, avg | 0.8093 | 0.5703 | 0.7405 | 0.2429 | 0.3332 | 0.5722 | 0.3142 | 0.3779 |

## T2I-CompBench per category (BEST minus row)

| model | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | +0.0291 | +0.0361 | +0.0485 | +0.0685 | +0.0650 | +0.0606 | +0.0062 | +0.0196 |
| naive CD, old schedule, avg last 3 | +0.0007 | +0.0187 | +0.0254 | +0.0319 | +0.0302 | +0.0409 | +0.0038 | +0.0079 |
| naive CD, converged schedule, K=8, avg | +0.0019 | +0.0114 | +0.0250 | +0.0018 | +0.0014 | +0.0257 | +0.0023 | +0.0030 |
| scored + projector reward, K=8, avg (paper) | -0.0021 | +0.0015 | +0.0180 | +0.0292 | +0.0052 | +0.0102 | +0.0017 | -0.0046 |

## GenEval2 per skill (absolute; Soft-TIFA per-atom mean by skill)

| model | attribute | count | object | position | verb |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.6961 | 0.3425 | 0.8118 | 0.4934 | 0.2485 |
| naive CD, old schedule, avg last 3 | 0.7077 | 0.3587 | 0.8368 | 0.4829 | 0.2347 |
| naive CD, converged schedule, K=8, avg | 0.7138 | 0.3923 | 0.8604 | 0.4670 | 0.2704 |
| scored + projector reward, K=8, avg (paper) | 0.7065 | 0.4021 | 0.8517 | 0.4801 | 0.2510 |
| BEST: scored + projector reward, K=10, avg | 0.7307 | 0.4076 | 0.8608 | 0.4944 | 0.2693 |

## GenEval2 per skill (BEST minus row)

| model | attribute | count | object | position | verb |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | +0.0346 | +0.0651 | +0.0490 | +0.0010 | +0.0208 |
| naive CD, old schedule, avg last 3 | +0.0230 | +0.0489 | +0.0240 | +0.0115 | +0.0346 |
| naive CD, converged schedule, K=8, avg | +0.0169 | +0.0153 | +0.0004 | +0.0274 | -0.0010 |
| scored + projector reward, K=8, avg (paper) | +0.0242 | +0.0056 | +0.0091 | +0.0143 | +0.0184 |

## GenEval2 per prompt complexity (number of scored atoms; absolute, then BEST minus row)

| model | 3-4 atoms | 5-6 atoms | 7-8 atoms | 9+ atoms |
|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.3520 | 0.2883 | 0.1900 | 0.1396 |
| naive CD, old schedule, avg last 3 | 0.4189 | 0.3023 | 0.1981 | 0.1433 |
| naive CD, converged schedule, K=8, avg | 0.4100 | 0.3326 | 0.1936 | 0.1409 |
| scored + projector reward, K=8, avg (paper) | 0.4552 | 0.3178 | 0.2015 | 0.1347 |
| BEST: scored + projector reward, K=10, avg | 0.4055 | 0.3282 | 0.1882 | 0.1384 |
| **delta rows** |  |  |  |  |
| naive CD, old schedule, raw final (unconverged) | +0.0535 | +0.0399 | -0.0017 | -0.0012 |
| naive CD, old schedule, avg last 3 | -0.0134 | +0.0259 | -0.0099 | -0.0049 |
| naive CD, converged schedule, K=8, avg | -0.0044 | -0.0044 | -0.0054 | -0.0025 |
| scored + projector reward, K=8, avg (paper) | -0.0496 | +0.0104 | -0.0133 | +0.0037 |
