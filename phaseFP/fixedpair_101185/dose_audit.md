# Fixed-pair dose audit

Cache: `phaseFP/fixedpair_101185`  
Records: **5042**; reversal subset: **1373** (27.23%)

## Effective label disagreement versus the correct arm

| arm | fraction of records with a different preferred image |
|---|---:|
| `correct` | 0.0000 |
| `counterfactual` | 0.2723 |
| `random` | 0.5000 |
| `inverted` | 1.0000 |

## Assigned original-prompt score margin

| population | arm | mean | 95% CI | median |
|---|---|---:|---|---:|
| all (n=5042) | `correct` | +0.2865 | [+0.2794, +0.2934] | +0.2080 |
| all (n=5042) | `counterfactual` | +0.1783 | [+0.1690, +0.1878] | +0.1308 |
| all (n=5042) | `random` | -0.0002 | [-0.0108, +0.0104] | +0.0012 |
| all (n=5042) | `inverted` | -0.2865 | [-0.2934, -0.2794] | -0.2079 |
| reversal_only (n=1373) | `correct` | +0.1986 | [+0.1866, +0.2110] | +0.0983 |
| reversal_only (n=1373) | `counterfactual` | -0.1986 | [-0.2110, -0.1866] | -0.0983 |
| reversal_only (n=1373) | `random` | -0.0062 | [-0.0219, +0.0102] | +0.0034 |
| reversal_only (n=1373) | `inverted` | -0.1986 | [-0.2110, -0.1866] | -0.0983 |
| non_reversal (n=3669) | `correct` | +0.3193 | [+0.3112, +0.3275] | +0.2548 |
| non_reversal (n=3669) | `counterfactual` | +0.3193 | [+0.3112, +0.3275] | +0.2548 |
| non_reversal (n=3669) | `random` | +0.0021 | [-0.0111, +0.0152] | -0.0032 |
| non_reversal (n=3669) | `inverted` | -0.3193 | [-0.3275, -0.3112] | -0.2548 |

## Reversal rate by prompt category

| category | records | reversal rate |
|---|---:|---:|
| 3d_spatial | 700 | 0.2714 |
| color | 677 | 0.3914 |
| complex | 550 | 0.2382 |
| non_spatial | 336 | 0.3274 |
| numeracy | 699 | 0.1373 |
| shape | 685 | 0.0920 |
| spatial | 700 | 0.5771 |
| texture | 695 | 0.1640 |

## Reversal rate by edit family

| edit family | records | reversal rate |
|---|---:|---:|
| 3d_spatial | 707 | 0.2687 |
| color | 827 | 0.3519 |
| count | 702 | 0.1368 |
| shape | 732 | 0.0929 |
| spatial | 941 | 0.5324 |
| texture | 810 | 0.1481 |
| verb | 323 | 0.3313 |

## Are reversals concentrated on near-ties?

- Mean original-prompt pair margin, reversed records: 0.1986
- Mean original-prompt pair margin, non-reversed records: 0.3193
- Difference: -0.1207
- Mean |counterfactual margin| on reversed pairs: 0.1622
