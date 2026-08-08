# Amortization ratio for the selected-trajectory null

Method: independent bootstraps; the two contrasts share no prompts, so no joint resampling exists and independence is a design property, not an assumption.

## Can the two contrasts be bootstrapped jointly?

Prompts are matched on **text**, not on the integer `idx` field, which is assigned per job and is not comparable across jobs.

Shared prompts across the two evaluations: **0**.

The populations are **disjoint**. No joint or matched bootstrap exists in the current artifacts, so the ratio below combines two independent bootstraps. That handles sampling error correctly, but the numerator and denominator still describe different prompt populations. Closing that gap requires re-estimating the teacher oracle on the same held-out split as the transfer contrast (roadmap task T), not a different statistical procedure.

## Components

| quantity | contrast | prompts | Δ | 95% CI |
|---|---|---:|---:|---|
| numerator (transfer) | B4 minus B2 @ step 4 | 2098 | +0.00045 | [-0.00889, +0.00975] |
| denominator (oracle) | oracle minus random @ step 8 | 1050 | +0.07172 | [+0.05709, +0.08655] |

## Ratio

- Point estimate: **0.0062**
- 95% interval: **[-0.1265, 0.1390]**
- One-sided 95% upper bound: **0.1168**

### Comparison with the shortcut quoted in the handoff

- upper transfer bound / oracle point: 0.1360
- upper transfer bound / oracle lower bound: 0.1709

The shortcut divides an interval endpoint by a point estimate and is therefore neither a bound on the ratio nor a confidence statement about it. The ratio interval above is the quantity that can be cited.
