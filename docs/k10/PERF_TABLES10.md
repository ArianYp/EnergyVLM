# Performance tables (2026-09-17)

All 4-step students of SD3.5-Medium, 3k COCO captions, sampled at guidance 1 on the scheduler grid; CompBench = mean of 8 categories (300 held-out prompts each, the official TEN images per prompt (the unconverged raw-final row stays at one image)); GenEval2 = Soft-TIFA geometric mean (800 prompts). Deltas are BEST minus the row. Old schedule = constant LR 2.8e-5, 6,000 updates; converged = cosine LR 1e-5, batch 16, 3,000 updates, window K={4..7} (K=8) or {6..9} (K=10).

## Summary

| model | seeds | CompBench | d vs BEST | GenEval2 | d vs BEST |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 5 | 0.4533 | +0.0491 | 0.2110 | +0.0182 |
| naive CD, old schedule, avg last 3 | 3 | 0.4774 | +0.0250 | 0.2253 | +0.0038 |
| naive CD, converged schedule, K=8, avg | 3 | 0.4862 | +0.0163 | 0.2271 | +0.0021 |
| scored + projector reward, K=8, avg (paper) | 3 | 0.4899 | +0.0125 | 0.2311 | -0.0020 |
| scored + projector reward, K=10, avg | 3 | 0.4961 | +0.0063 | 0.2311 | -0.0020 |
| BEST: scored + projector reward, K=10 + grid A, avg | 3 | 0.5024 | +0.0000 | 0.2291 | +0.0000 |

## T2I-CompBench per category (absolute)

| model | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.7803 | 0.5342 | 0.6920 | 0.1743 | 0.2682 | 0.5116 | 0.3080 | 0.3583 |
| naive CD, old schedule, avg last 3 | 0.8017 | 0.5509 | 0.7173 | 0.2178 | 0.3129 | 0.5379 | 0.3111 | 0.3699 |
| naive CD, converged schedule, K=8, avg | 0.7990 | 0.5606 | 0.7340 | 0.2174 | 0.3340 | 0.5542 | 0.3127 | 0.3773 |
| scored + projector reward, K=8, avg (paper) | 0.8048 | 0.5659 | 0.7321 | 0.2288 | 0.3370 | 0.5582 | 0.3129 | 0.3799 |
| scored + projector reward, K=10, avg | 0.7934 | 0.5681 | 0.7458 | 0.2464 | 0.3484 | 0.5725 | 0.3143 | 0.3800 |
| BEST: scored + projector reward, K=10 + grid A, avg | 0.7981 | 0.5727 | 0.7481 | 0.2639 | 0.3615 | 0.5820 | 0.3123 | 0.3809 |

## T2I-CompBench per category (BEST minus row)

| model | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | +0.0179 | +0.0385 | +0.0562 | +0.0896 | +0.0933 | +0.0704 | +0.0043 | +0.0226 |
| naive CD, old schedule, avg last 3 | -0.0036 | +0.0218 | +0.0309 | +0.0462 | +0.0486 | +0.0441 | +0.0012 | +0.0110 |
| naive CD, converged schedule, K=8, avg | -0.0009 | +0.0120 | +0.0141 | +0.0465 | +0.0274 | +0.0278 | -0.0004 | +0.0036 |
| scored + projector reward, K=8, avg (paper) | -0.0067 | +0.0068 | +0.0161 | +0.0352 | +0.0245 | +0.0239 | -0.0005 | +0.0010 |
| scored + projector reward, K=10, avg | +0.0047 | +0.0046 | +0.0023 | +0.0175 | +0.0131 | +0.0095 | -0.0019 | +0.0009 |

## GenEval2 per skill (absolute; Soft-TIFA per-atom mean by skill)

| model | attribute | count | object | position | verb |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.6961 | 0.3425 | 0.8118 | 0.4934 | 0.2485 |
| naive CD, old schedule, avg last 3 | 0.7076 | 0.3619 | 0.8419 | 0.4735 | 0.2607 |
| naive CD, converged schedule, K=8, avg | 0.7088 | 0.3933 | 0.8593 | 0.4794 | 0.2577 |
| scored + projector reward, K=8, avg (paper) | 0.7072 | 0.3948 | 0.8531 | 0.4779 | 0.2587 |
| scored + projector reward, K=10, avg | 0.7277 | 0.4178 | 0.8681 | 0.4964 | 0.2824 |
| BEST: scored + projector reward, K=10 + grid A, avg | 0.7282 | 0.4309 | 0.8808 | 0.4850 | 0.2503 |

## GenEval2 per skill (BEST minus row)

| model | attribute | count | object | position | verb |
|---|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | +0.0320 | +0.0884 | +0.0690 | -0.0084 | +0.0018 |
| naive CD, old schedule, avg last 3 | +0.0205 | +0.0690 | +0.0390 | +0.0114 | -0.0105 |
| naive CD, converged schedule, K=8, avg | +0.0193 | +0.0376 | +0.0215 | +0.0055 | -0.0074 |
| scored + projector reward, K=8, avg (paper) | +0.0210 | +0.0361 | +0.0277 | +0.0071 | -0.0084 |
| scored + projector reward, K=10, avg | +0.0004 | +0.0131 | +0.0128 | -0.0114 | -0.0322 |

## GenEval2 per prompt complexity (number of scored atoms; absolute, then BEST minus row)

| model | 3-4 atoms | 5-6 atoms | 7-8 atoms | 9+ atoms |
|---|---|---|---|---|
| naive CD, old schedule, raw final (unconverged) | 0.3520 | 0.2883 | 0.1900 | 0.1396 |
| naive CD, old schedule, avg last 3 | 0.4075 | 0.3110 | 0.1972 | 0.1427 |
| naive CD, converged schedule, K=8, avg | 0.3926 | 0.3251 | 0.2029 | 0.1385 |
| scored + projector reward, K=8, avg (paper) | 0.4485 | 0.3161 | 0.2073 | 0.1378 |
| scored + projector reward, K=10, avg | 0.3728 | 0.3438 | 0.2008 | 0.1426 |
| BEST: scored + projector reward, K=10 + grid A, avg | 0.3781 | 0.3368 | 0.2065 | 0.1377 |
| **delta rows** |  |  |  |  |
| naive CD, old schedule, raw final (unconverged) | +0.0260 | +0.0485 | +0.0165 | -0.0019 |
| naive CD, old schedule, avg last 3 | -0.0294 | +0.0258 | +0.0093 | -0.0050 |
| naive CD, converged schedule, K=8, avg | -0.0146 | +0.0117 | +0.0036 | -0.0008 |
| scored + projector reward, K=8, avg (paper) | -0.0704 | +0.0207 | -0.0008 | -0.0002 |
| scored + projector reward, K=10, avg | +0.0052 | -0.0070 | +0.0056 | -0.0049 |
