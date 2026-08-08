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

## `independent` (12,000 examples)

- corr(sigma, |gap|), raw: **+0.2538**
- corr(sigma, |gap| / branch error scale): **+0.2260** [+0.2140, +0.2378]

The scale-normalised correlation is the one that tests coalescence: the raw correlation would be positive merely because flow errors grow with sigma.

| sigma band | n | mean gap | 95% CI | mean \|gap\| | **\|gap\| / error scale** | winner share | branch error scale |
|---|---:|---:|---|---:|---:|---:|---:|
| [0.0, 0.2) | 943 | -0.003274 | [-0.004421, -0.002139] | 0.009064 | **0.0099** | 0.457 | 0.4668 |
| [0.2, 0.4) | 1219 | -0.002129 | [-0.002494, -0.001781] | 0.004157 | **0.0143** | 0.440 | 0.1500 |
| [0.4, 0.6) | 1853 | -0.002272 | [-0.002579, -0.001970] | 0.004298 | **0.0195** | 0.423 | 0.1142 |
| [0.6, 0.8) | 2790 | -0.003282 | [-0.003619, -0.002960] | 0.005522 | **0.0242** | 0.397 | 0.1179 |
| [0.8, 1.0) | 5195 | -0.011396 | [-0.013158, -0.009708] | 0.028676 | **0.0431** | 0.402 | 0.3255 |

Read the **bold** column, not raw |gap|. Raw |gap| is U-shaped because flow errors are large at both ends of the sigma range; dividing by the branch error scale removes that and leaves the quantity the coalescence hypothesis predicts.

## Shared versus independent corruption noise

- scale-normalised corr(sigma, |gap|): shared +0.2210 vs independent +0.2260
- overall mean gap: shared -0.007386 vs independent -0.006521

Coalescence is supported only if the shared condition shows a materially stronger sigma dependence AND wins on the primary endpoint. A flat profile in both retires the mechanism.
