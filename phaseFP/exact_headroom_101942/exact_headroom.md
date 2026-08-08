# Exact best-of-4 headroom (per-candidate CompBench expectations)

2098 prompt-category rows; every candidate slot scored, so each policy is an exact per-prompt expectation rather than a single draw.

| policy | CompBench primary |
|---|---:|
| best-of-4 (VQAScore argmax) | 0.53078 |
| uniform over all 4 | 0.47226 |
| uniform over the 3 non-oracle | 0.45276 |

- **Headroom vs uniform (unbiased): +0.05852 [+0.05148, +0.06542]**
- Headroom vs non-oracle (biased): +0.07803 [+0.06865, +0.08723]

Mixture-identity residual: -1.11e-16 (should be ~0).
Control bias: exact N/(N-1) = 1.3333; observed 1.3333.

Excluding the oracle from the control pool inflates measured headroom by exactly 33.3% — this is algebra, not an empirical finding.
