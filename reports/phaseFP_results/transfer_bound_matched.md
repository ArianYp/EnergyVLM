# Amortization ratio for the selected-trajectory null

Method: joint bootstrap on shared prompts.

## Can the two contrasts be bootstrapped jointly?

Prompts are matched on **text**, not on the integer `idx` field, which is assigned per job and is not comparable across jobs.

Shared prompts across the two evaluations: **2098**.

The populations overlap, so numerator and denominator are resampled together and their correlation is captured.

## Components

| quantity | contrast | prompts | Δ | 95% CI |
|---|---|---:|---:|---|
| numerator (transfer) | B4 minus B2 @ step 4 | 2098 | +0.00045 | [-0.00889, +0.00975] |
| denominator (oracle) | TeacherBest minus TeacherUniform @ step 8 | 2098 | +0.05287 | [+0.04336, +0.06247] |

## Ratio

- Point estimate: **0.0085**
- 95% interval: **[-0.1720, 0.1863]**
- One-sided 95% upper bound: **0.1575**

### Comparison with the shortcut quoted in the handoff

- upper transfer bound / oracle point: 0.1845
- upper transfer bound / oracle lower bound: 0.2249

The shortcut divides an interval endpoint by a point estimate and is therefore neither a bound on the ratio nor a confidence statement about it. The ratio interval above is the quantity that can be cited.
