# Preference signal versus noise level

Per-example rows from `telemetry.jsonl`. The gap is the reference-relative quantity the loss drives negative: `(e_theta+ - e_theta-) - (e_0+ - e_0-)`.

## `shared` (12,000 examples)

- corr(sigma, |gap|), raw: **+0.2635**
- corr(sigma, |gap| / branch error scale): **+0.2210** [+0.2085, +0.2336]

The scale-normalised correlation is the one that tests coalescence: the raw correlation would be positive merely because flow errors grow with sigma.

| sigma band | n | mean gap | 95% CI | mean \|gap\| | **\|gap\| / error scale** | winner share | branch error scale |
|---|---:|---:|---|---:|---:|---:|---:|
| [0.0, 0.2) | 956 | -0.003678 | [-0.005181, -0.002326] | 0.009419 | **0.0102** | 0.447 | 0.4671 |
| [0.2, 0.4) | 1245 | -0.002276 | [-0.002659, -0.001908] | 0.004390 | **0.0150** | 0.430 | 0.1515 |
| [0.4, 0.6) | 1836 | -0.002535 | [-0.002853, -0.002234] | 0.004277 | **0.0198** | 0.399 | 0.1123 |
| [0.6, 0.8) | 2824 | -0.003102 | [-0.003411, -0.002797] | 0.005225 | **0.0229** | 0.396 | 0.1175 |
| [0.8, 1.0) | 5139 | -0.013401 | [-0.014909, -0.011919] | 0.027558 | **0.0410** | 0.404 | 0.3308 |

Read the **bold** column, not raw |gap|. Raw |gap| is U-shaped because flow errors are large at both ends of the sigma range; dividing by the branch error scale removes that and leaves the quantity the coalescence hypothesis predicts.

