# Diversity price of teacher best-of-4 selection

1200 prompts from the frozen-teacher `cfg7_s8` bank, 4 images per prompt per policy, same DINO/LPIPS estimators as the student diversity gate.

| metric | teacher best-of-K | teacher random | price (best − random) |
|---|---:|---:|---|
| DINO diversity | 0.24711 | 0.37383 | **-0.12672** [-0.13583, -0.11756] |
| LPIPS diversity | 0.43089 | 0.59087 | **-0.15998** [-0.17142, -0.14815] |
| mean VQAScore | 0.87530 | 0.74739 | +0.12791 |
| distinct images per prompt | 2.388 | 3.339 | -0.951 |

## How to use this number

The oracle policy the student is asked to amortize is itself less diverse than random teacher sampling by the amount above, measured in the same units as the student gate. A student whose diversity loss is no larger than this price has not become worse than what it imitates; a student that loses more has paid a cost the policy does not explain. Report both comparisons rather than the M1 comparison alone.

The analytic best-of-4 KL displacement log 4 - 3/4 ~ 0.636 nats is in different units and cannot be compared with these values.
