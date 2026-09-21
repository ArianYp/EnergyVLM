# Nested-grid evaluation (2026-09-17)

Converged 3k averaged students (seed 0) sampled at 4 steps on sub-grids of the 8-step training grid. Paired per prompt against the same checkpoint on the scheduler's 4-step grid (the reported numbers). Single-run floor for an unpaired contrast is 0.0065 CompBench; here the pairing removes the prompt variance, so the sign / Wilcoxon tests are the relevant ones.

Grids:

- scheduler: sigma = 1, 0.858, 0.602, 0.009, 0 (states 0,-,-,7: shares only the endpoints)
- A: sigma = 1, 0.883, 0.694, 0.338, 0 (states 0,2,4,6,8: one coarse last step)
- B: sigma = 1, 0.883, 0.548, 0.009, 0 (states 0,2,5,7,8: closest to deployment)
- C: sigma = 1, 0.883, 0.694, 0.009, 0 (states 0,2,4,7,8)

| model | grid | CompBench | dCB vs scheduler | win rate | sign p | Wilcoxon p | GenEval2 | dGE2 | sign p | d color | d shape | d texture | d spatial | d 3d_spatial | d numeracy | d non_spatial | d complex |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | scheduler | 0.4860 | – | – | – | – | 0.2296 | – | – | – | – | – | – | – | – | – | – |
| naive | A | 0.4949 | +0.0089 | 0.526 | 0.021 | 0.00011 | 0.2271 | -0.0025 | 0.5 | +0.0064* | +0.0070 | +0.0086 | -0.0095 | +0.0146* | +0.0443* | -0.0018* | +0.0015 |
| naive | B | 0.4847 | -0.0013 | 0.505 | 0.7 | 0.28 | 0.2275 | -0.0020 | 0.75 | +0.0009 | -0.0002 | -0.0026 | -0.0099 | -0.0050 | +0.0015 | +0.0002 | +0.0047 |
| naive | C | 0.4872 | +0.0012 | 0.503 | 0.84 | 0.14 | 0.2273 | -0.0022 | 0.00077 | -0.0008 | +0.0016 | +0.0028 | -0.0161 | +0.0018 | +0.0162 | +0.0004 | +0.0037 |
| ours | scheduler | 0.4877 | – | – | – | – | 0.2296 | – | – | – | – | – | – | – | – | – | – |
| ours | A | 0.4926 | +0.0049 | 0.523 | 0.039 | 0.0087 | 0.2278 | -0.0018 | 0.0064 | +0.0069* | +0.0018 | +0.0047* | +0.0203 | +0.0094* | -0.0019 | -0.0016* | -0.0005 |
| ours | B | 0.4852 | -0.0024 | 0.500 | 1 | 0.58 | 0.2326 | +0.0030 | 0.31 | -0.0019 | -0.0045 | -0.0078 | +0.0037 | -0.0045 | -0.0062 | +0.0001 | +0.0015 |
| ours | C | 0.4888 | +0.0011 | 0.502 | 0.91 | 0.97 | 0.2240 | -0.0056 | 0.037 | -0.0042 | -0.0052 | +0.0061 | +0.0098 | +0.0045 | -0.0001 | +0.0009 | -0.0028 |

`*` = per-category sign test p < 0.05. Win rate = share of non-tied prompts where the sub-grid scores higher.
