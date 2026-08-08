# Diversity price of teacher best-of-8 selection

1200 prompts from the frozen-teacher `cfg7_s8` bank, 4 images per prompt per policy, same DINO/LPIPS estimators as the student diversity gate.

| metric | teacher best-of-K | teacher random | price (best − random) |
|---|---:|---:|---|
| DINO diversity | 0.00150 | 0.36774 | **-0.36624** [-0.37522, -0.35743] |
| LPIPS diversity | 0.00391 | 0.58880 | **-0.58488** [-0.59433, -0.57571] |
| mean VQAScore | 0.90121 | 0.74760 | +0.15361 |
| distinct images per prompt | 1.010 | 3.328 | -2.317 |

## How to use this number

The oracle policy the student is asked to amortize is itself less diverse than random teacher sampling by the amount above, measured in the same units as the student gate. A student whose diversity loss is no larger than this price has not become worse than what it imitates; a student that loses more has paid a cost the policy does not explain. Report both comparisons rather than the M1 comparison alone.

The analytic best-of-4 KL displacement log 4 - 3/4 ~ 0.636 nats is in different units and cannot be compared with these values.
