# Selection x reward factorial (S4i)

## raw final checkpoints

### CompBench

| cell | seeds | mean +- sd | per seed |
|---|---|---|---|
| M_00 random pick, no reward | 5 | 0.4533 +- 0.0131 | 0.4491 / 0.4393 / 0.4612 / 0.4453 / 0.4719 |
| M_01 random pick + exact reward | 5 | 0.4663 +- 0.0104 | 0.4562 / 0.4578 / 0.4790 / 0.4628 / 0.4755 |
| M_10 argmax, no reward | 5 | 0.4687 +- 0.0099 | 0.4690 / 0.4591 / 0.4825 / 0.4736 / 0.4593 |
| M_11 argmax + exact reward | 5 | 0.4712 +- 0.0125 | 0.4723 / 0.4498 / 0.4798 / 0.4806 / 0.4733 |

| contrast | seed-paired mean +- sd [95% CI] (p, n) |
|---|---|
| reward | random (M_01 - M_00) | +0.0129 +- 0.0070 [+0.0042, +0.0216] (p=0.015, n=5) |
| reward | argmax (M_11 - M_10) | +0.0025 +- 0.0089 [-0.0086, +0.0135] (p=0.571, n=5) |
| selection | no reward (M_10 - M_00) | +0.0154 +- 0.0160 [-0.0045, +0.0352] (p=0.098, n=5) |
| selection | reward (M_11 - M_01) | +0.0049 +- 0.0115 [-0.0094, +0.0192] (p=0.394, n=5) |
| interaction I = (M_11-M_10)-(M_01-M_00) | -0.0105 +- 0.0148 [-0.0289, 0.0079] (p=0.189, n=5) |
| unpaired OLS: selection 0.0154 (se 0.0073), reward 0.0129 (se 0.0073), interaction -0.0105 (se 0.0103, p=0.327), n runs 20 | |

### GenEval2

| cell | seeds | mean +- sd | per seed |
|---|---|---|---|
| M_00 random pick, no reward | 5 | 21.10 +- 1.69 | 23.42 / 21.27 / 18.65 / 20.98 / 21.15 |
| M_01 random pick + exact reward | 5 | 22.25 +- 1.00 | 23.21 / 22.75 / 21.01 / 21.34 / 22.94 |
| M_10 argmax, no reward | 5 | 22.53 +- 1.02 | 20.91 / 23.02 / 23.46 / 23.08 / 22.20 |
| M_11 argmax + exact reward | 5 | 23.11 +- 1.01 | 22.22 / 21.84 / 24.04 / 23.93 / 23.52 |

| contrast | seed-paired mean +- sd [95% CI] (p, n) |
|---|---|
| reward | random (M_01 - M_00) | +1.15 +- 1.05 [-0.16, +2.46] (p=0.071, n=5) |
| reward | argmax (M_11 - M_10) | +0.58 +- 1.03 [-0.70, +1.85] (p=0.280, n=5) |
| selection | no reward (M_10 - M_00) | +1.44 +- 2.63 [-1.82, +4.70] (p=0.288, n=5) |
| selection | reward (M_11 - M_01) | +0.86 +- 1.89 [-1.49, +3.21] (p=0.368, n=5) |
| interaction I = (M_11-M_10)-(M_01-M_00) | -0.58 +- 1.68 [-2.67, 1.51] (p=0.485, n=5) |
| unpaired OLS: selection 1.44 (se 0.77), reward 1.15 (se 0.77), interaction -0.58 (se 1.09, p=0.602), n runs 20 | |

## averaged checkpoints (2k/4k/6k)

### CompBench

| cell | seeds | mean +- sd | per seed |
|---|---|---|---|
| M_00 random pick, no reward | 5 | 0.4751 +- 0.0032 | 0.4756 / 0.4724 / 0.4732 / 0.4740 / 0.4804 |
| M_01 random pick + exact reward | 5 | 0.4819 +- 0.0060 | 0.4728 / 0.4847 / 0.4788 / 0.4855 / 0.4876 |
| M_10 argmax, no reward | 5 | 0.4868 +- 0.0031 | 0.4849 / 0.4861 / 0.4909 / 0.4890 / 0.4833 |
| M_11 argmax + exact reward | 5 | 0.4923 +- 0.0021 | 0.4906 / 0.4944 / 0.4938 / 0.4931 / 0.4897 |

| contrast | seed-paired mean +- sd [95% CI] (p, n) |
|---|---|
| reward | random (M_01 - M_00) | +0.0068 +- 0.0060 [-0.0008, +0.0143] (p=0.067, n=5) |
| reward | argmax (M_11 - M_10) | +0.0055 +- 0.0021 [+0.0029, +0.0081] (p=0.004, n=5) |
| selection | no reward (M_10 - M_00) | +0.0117 +- 0.0058 [+0.0045, +0.0189] (p=0.011, n=5) |
| selection | reward (M_11 - M_01) | +0.0104 +- 0.0062 [+0.0028, +0.0181] (p=0.019, n=5) |
| interaction I = (M_11-M_10)-(M_01-M_00) | -0.0013 +- 0.0060 [-0.0087, 0.0062] (p=0.661, n=5) |
| unpaired OLS: selection 0.0117 (se 0.0025), reward 0.0068 (se 0.0025), interaction -0.0013 (se 0.0035, p=0.720), n runs 20 | |

### GenEval2

| cell | seeds | mean +- sd | per seed |
|---|---|---|---|
| M_00 random pick, no reward | 5 | 22.47 +- 0.68 | 23.63 / 21.93 / 22.04 / 22.35 / 22.38 |
| M_01 random pick + exact reward | 5 | 23.06 +- 0.65 | 23.22 / 23.21 / 22.98 / 22.06 / 23.85 |
| M_10 argmax, no reward | 5 | 23.92 +- 1.29 | 24.17 / 25.01 / 22.00 / 25.10 / 23.32 |
| M_11 argmax + exact reward | 5 | 24.41 +- 0.77 | 24.83 / 23.47 / 23.73 / 25.24 / 24.77 |

| contrast | seed-paired mean +- sd [95% CI] (p, n) |
|---|---|
| reward | random (M_01 - M_00) | +0.60 +- 0.89 [-0.50, +1.70] (p=0.206, n=5) |
| reward | argmax (M_11 - M_10) | +0.49 +- 1.30 [-1.12, +2.10] (p=0.448, n=5) |
| selection | no reward (M_10 - M_00) | +1.46 +- 1.38 [-0.26, +3.17] (p=0.078, n=5) |
| selection | reward (M_11 - M_01) | +1.35 +- 1.14 [-0.06, +2.76] (p=0.057, n=5) |
| interaction I = (M_11-M_10)-(M_01-M_00) | -0.11 +- 1.57 [-2.06, 1.84] (p=0.883, n=5) |
| unpaired OLS: selection 1.46 (se 0.56), reward 0.60 (se 0.56), interaction -0.11 (se 0.79, p=0.892), n runs 20 | |

