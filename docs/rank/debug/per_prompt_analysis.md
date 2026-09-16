# Per-prompt analysis of the hp1 3k students (naive / ours / rank negatives)

Read-only re-analysis of the CompBench (`compbench_scores/*/scores.json`) and GenEval2 (`geneval2_scores/*/scores.json`) per-prompt records under `phaseN/eval_*`, using `iclr2027/verify_numbers.py:load_dir`. Nothing is taken from a summary table.

* CompBench: **2398 prompts** (8 categories x 300), 1 image per prompt, BLIP-VQA/UniDet/CLIP per category.
* GenEval2: **800 prompts**, soft-TIFA geometric mean over atoms.
* Ranking-negative target categories: `color, numeracy, spatial, 3d_spatial, texture, shape`.

## 1. Runs and headline scores

| run | CompBench | GenEval2 | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex | dir |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive_s0_avg | 0.4860 | 22.96 | 0.8074 | 0.5589 | 0.7155 | 0.2411 | 0.3318 | 0.5465 | 0.3119 | 0.3748 | `eval_W_B2-hp1_3k-hp1-avg_s0_145251` |
| naive_s0_raw | 0.4846 | 22.28 | 0.8033 | 0.5565 | 0.7186 | 0.2297 | 0.3277 | 0.5516 | 0.3119 | 0.3778 | `eval_W_B2-hp1_3k-hp1_s0_145250` |
| naive_s2_avg | 0.4852 | 22.38 | 0.8146 | 0.5581 | 0.7212 | 0.2163 | 0.3236 | 0.5603 | 0.3116 | 0.3759 | `eval_W_B2-hp1_3k-hp1-avg_s2_151360` |
| naive_s2_raw | 0.4858 | 22.68 | 0.8136 | 0.5595 | 0.7231 | 0.2171 | 0.3287 | 0.5556 | 0.3118 | 0.3772 | `eval_W_B2-hp1_3k-hp1_s2_151358` |
| ours_s0_avg | 0.4877 | 22.96 | 0.8114 | 0.5689 | 0.7225 | 0.2136 | 0.3279 | 0.5621 | 0.3125 | 0.3825 | `eval_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg_s0_145253` |
| ours_s0_raw | 0.4878 | 22.82 | 0.8098 | 0.5711 | 0.7219 | 0.2167 | 0.3295 | 0.5572 | 0.3126 | 0.3832 | `eval_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1_s0_145252` |
| ours_s2_avg | 0.4884 | 23.57 | 0.8165 | 0.5586 | 0.7259 | 0.2194 | 0.3300 | 0.5607 | 0.3126 | 0.3833 | `eval_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1-avg_s2_151364` |
| ours_s2_raw | 0.4878 | 24.31 | 0.8126 | 0.5635 | 0.7262 | 0.2141 | 0.3275 | 0.5641 | 0.3126 | 0.3819 | `eval_W_CD_dinop_hard-rewRi-s16-hp1_3k-hp1_s2_151362` |
| rankA_s0_avg *(partial)* | 0.4058 | -- | 0.7320 | 0.5390 | 0.6475 | 0.1238 | 0.1841 | 0.3827 | 0.3083 | 0.3289 | `eval_W_CD_dinop_hard-rewRi-s16-rankA-hp1_3k-hp1-avg_s0_151319` |
| rankA_s0_raw | 0.4014 | 16.14 | 0.7295 | 0.5342 | 0.6348 | 0.1266 | 0.1778 | 0.3757 | 0.3077 | 0.3247 | `eval_W_CD_dinop_hard-rewRi-s16-rankA-hp1_3k-hp1_s0_151317` |
| rankB1_s0_avg | 0.4867 | 23.33 | 0.8046 | 0.5609 | 0.7330 | 0.2164 | 0.3228 | 0.5612 | 0.3130 | 0.3819 | `eval_W_CD_dinop_hard-rewRi-s16-rankB1-hp1_3k-hp1-avg_s0_151305` |
| rankB1_s0_raw | 0.4866 | 23.45 | 0.8039 | 0.5593 | 0.7308 | 0.2088 | 0.3243 | 0.5714 | 0.3128 | 0.3817 | `eval_W_CD_dinop_hard-rewRi-s16-rankB1-hp1_3k-hp1_s0_151303` |
| rankB2_s1_avg | 0.4884 | 23.70 | 0.8046 | 0.5695 | 0.7254 | 0.2186 | 0.3295 | 0.5629 | 0.3129 | 0.3838 | `eval_W_CD_dinop_hard-rewRi-s16-rankB2-hp1_3k-hp1-avg_s1_151331` |
| rankB2_s1_raw | 0.4873 | 24.07 | 0.8097 | 0.5731 | 0.7255 | 0.2158 | 0.3336 | 0.5455 | 0.3130 | 0.3827 | `eval_W_CD_dinop_hard-rewRi-s16-rankB2-hp1_3k-hp1_s1_151329` |
| rankB2_s2_avg | 0.4866 | 23.44 | 0.8153 | 0.5621 | 0.7221 | 0.2081 | 0.3275 | 0.5621 | 0.3126 | 0.3832 | `eval_W_CD_dinop_hard-rewRi-s16-rankB2-hp1_3k-hp1-avg_s2_151340` |
| rankB2_s2_raw | 0.4882 | 22.85 | 0.8171 | 0.5596 | 0.7248 | 0.2101 | 0.3226 | 0.5744 | 0.3128 | 0.3842 | `eval_W_CD_dinop_hard-rewRi-s16-rankB2-hp1_3k-hp1_s2_151338` |

`rankA_s0_avg` was still being evaluated: 7/8 CompBench categories (no `complex`), no GenEval2. Its rows below are restricted to the categories it has, so its "overall" mean is **not** comparable to the 8-category means of the other runs.

Spread of the 8 fully-evaluated distinct models on CompBench: 0.4014 to 0.4884.

## 2. Noise floor: paired differences between runs that should be equivalent

Two kinds of control: (a) **checkpoint averaging** (raw final vs average of the last checkpoints of the *same* training run) and (b) **training seed** (same recipe, different seed). `sd` is the sd of the per-prompt difference; `r` is the Pearson correlation of the two runs' per-prompt scores; `>0.1` is the fraction of prompts whose score moves by more than 0.1.

| control pair | kind | mean diff | sd | >0.1 | >0.2 | r (pearson) | rho (spearman) | wilcoxon p | sign p |
|---|---|---|---|---|---|---|---|---|---|
| naive_s0_raw - naive_s0_avg | ckpt-avg | -0.0014 | 0.0872 | 0.063 | 0.032 | 0.962 | 0.960 | 0.0602 | 0.1392 |
| ours_s0_raw - ours_s0_avg | ckpt-avg | 0.0001 | 0.0727 | 0.049 | 0.021 | 0.974 | 0.972 | 0.7320 | 1.0000 |
| naive_s2_raw - naive_s2_avg | ckpt-avg | 0.0006 | 0.0820 | 0.063 | 0.029 | 0.966 | 0.966 | 0.8795 | 0.8556 |
| ours_s2_raw - ours_s2_avg | ckpt-avg | -0.0006 | 0.0898 | 0.063 | 0.032 | 0.960 | 0.959 | 0.3668 | 0.5701 |
| naive_s2_raw - naive_s0_raw | seed | 0.0012 | 0.1233 | 0.141 | 0.075 | 0.924 | 0.919 | 0.0643 | 0.1269 |
| naive_s2_avg - naive_s0_avg | seed | -0.0008 | 0.1234 | 0.147 | 0.070 | 0.924 | 0.919 | 0.2131 | 0.5152 |
| ours_s2_raw - ours_s0_raw | seed | 0.0001 | 0.1185 | 0.131 | 0.065 | 0.931 | 0.927 | 0.7847 | 0.5894 |
| ours_s2_avg - ours_s0_avg | seed | 0.0007 | 0.1144 | 0.123 | 0.060 | 0.936 | 0.931 | 0.2683 | 0.2596 |
| rankB2_s2_raw - rankB2_s1_raw | seed | 0.0009 | 0.1220 | 0.145 | 0.069 | 0.927 | 0.925 | 0.7142 | 0.7202 |
| rankB2_s2_avg - rankB2_s1_avg | seed | -0.0018 | 0.1214 | 0.135 | 0.066 | 0.927 | 0.925 | 0.2656 | 0.4181 |
| rankB1_s0_raw - rankB1_s0_avg | ckpt-avg | -0.0001 | 0.0798 | 0.053 | 0.027 | 0.969 | 0.968 | 0.3755 | 0.2939 |
| rankA_s0_raw - rankA_s0_avg | ckpt-avg | -0.0044 | 0.1267 | 0.129 | 0.073 | 0.920 | 0.917 | 0.0002 | 0.0012 |

GenEval2 noise floor (same pairs, per-prompt soft-TIFA score, 0-1 scale):

| control pair | mean diff x100 | sd | >0.1 | r | wilcoxon p | sign p |
|---|---|---|---|---|---|---|
| naive_s0_raw - naive_s0_avg | -0.677 | 0.1103 | 0.149 | 0.935 | 0.2937 | 0.9718 |
| ours_s0_raw - ours_s0_avg | -0.137 | 0.1098 | 0.131 | 0.937 | 0.9047 | 0.6974 |
| naive_s2_raw - naive_s2_avg | 0.301 | 0.1199 | 0.161 | 0.922 | 0.3862 | 0.2159 |
| ours_s2_raw - ours_s2_avg | 0.736 | 0.1035 | 0.130 | 0.947 | 0.8479 | 0.9718 |
| naive_s2_raw - naive_s0_raw | 0.399 | 0.1796 | 0.287 | 0.826 | 0.5119 | 0.5018 |
| naive_s2_avg - naive_s0_avg | -0.579 | 0.1830 | 0.287 | 0.820 | 0.4663 | 0.8597 |
| ours_s2_raw - ours_s0_raw | 1.485 | 0.1892 | 0.255 | 0.821 | 0.0726 | 0.5018 |
| ours_s2_avg - ours_s0_avg | 0.612 | 0.1747 | 0.236 | 0.843 | 0.2965 | 0.5018 |
| rankB2_s2_raw - rankB2_s1_raw | -1.226 | 0.1758 | 0.256 | 0.841 | 0.0533 | 0.1679 |
| rankB2_s2_avg - rankB2_s1_avg | -0.263 | 0.1794 | 0.266 | 0.835 | 0.1550 | 0.0098 |
| rankB1_s0_raw - rankB1_s0_avg | 0.118 | 0.1369 | 0.139 | 0.905 | 0.9607 | 0.5478 |

**Read of the noise floor.** Per-prompt CompBench scores decorrelate badly between any two runs (median r = 0.933), the per-prompt difference has sd ~= 0.114 against a within-run score sd of 0.318, and 0.126 of prompts (median) move by more than 0.1 between two runs that differ only by seed or by checkpoint averaging. The image-level noise is therefore comparable in size to the whole between-prompt signal; only the *mean* is stable.

### 2b. Similarity structure: which runs look alike, prompt by prompt

sd of the per-prompt difference between every pair of fully evaluated runs (lower = more alike). `rankA` is excluded from the family summaries because it is a different model altogether.

| | naive_s0_avg | naive_s0_raw | naive_s2_avg | naive_s2_raw | ours_s0_avg | ours_s0_raw | ours_s2_avg | ours_s2_raw | rankA_s0_raw | rankB1_s0_avg | rankB1_s0_raw | rankB2_s1_avg | rankB2_s1_raw | rankB2_s2_avg | rankB2_s2_raw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **naive_s0_avg** | 0.000 | 0.087 | 0.123 | 0.128 | 0.155 | 0.154 | 0.149 | 0.158 | 0.247 | 0.155 | 0.157 | 0.156 | 0.149 | 0.155 | 0.154 |
| **naive_s0_raw** | 0.087 | 0.000 | 0.124 | 0.123 | 0.152 | 0.155 | 0.148 | 0.157 | 0.248 | 0.156 | 0.154 | 0.155 | 0.149 | 0.155 | 0.154 |
| **naive_s2_avg** | 0.123 | 0.124 | 0.000 | 0.082 | 0.150 | 0.150 | 0.143 | 0.150 | 0.247 | 0.158 | 0.157 | 0.152 | 0.145 | 0.148 | 0.147 |
| **naive_s2_raw** | 0.128 | 0.123 | 0.082 | 0.000 | 0.152 | 0.151 | 0.146 | 0.151 | 0.247 | 0.160 | 0.159 | 0.152 | 0.145 | 0.149 | 0.150 |
| **ours_s0_avg** | 0.155 | 0.152 | 0.150 | 0.152 | 0.000 | 0.073 | 0.114 | 0.115 | 0.245 | 0.122 | 0.119 | 0.122 | 0.118 | 0.116 | 0.117 |
| **ours_s0_raw** | 0.154 | 0.155 | 0.150 | 0.151 | 0.073 | 0.000 | 0.117 | 0.118 | 0.247 | 0.127 | 0.120 | 0.122 | 0.124 | 0.116 | 0.118 |
| **ours_s2_avg** | 0.149 | 0.148 | 0.143 | 0.146 | 0.114 | 0.117 | 0.000 | 0.090 | 0.244 | 0.128 | 0.126 | 0.125 | 0.122 | 0.121 | 0.123 |
| **ours_s2_raw** | 0.158 | 0.157 | 0.150 | 0.151 | 0.115 | 0.118 | 0.090 | 0.000 | 0.245 | 0.127 | 0.124 | 0.122 | 0.122 | 0.123 | 0.126 |
| **rankA_s0_raw** | 0.247 | 0.248 | 0.247 | 0.247 | 0.245 | 0.247 | 0.244 | 0.245 | 0.000 | 0.249 | 0.252 | 0.247 | 0.245 | 0.245 | 0.245 |
| **rankB1_s0_avg** | 0.155 | 0.156 | 0.158 | 0.160 | 0.122 | 0.127 | 0.128 | 0.127 | 0.249 | 0.000 | 0.080 | 0.125 | 0.126 | 0.134 | 0.130 |
| **rankB1_s0_raw** | 0.157 | 0.154 | 0.157 | 0.159 | 0.119 | 0.120 | 0.126 | 0.124 | 0.252 | 0.080 | 0.000 | 0.130 | 0.126 | 0.134 | 0.132 |
| **rankB2_s1_avg** | 0.156 | 0.155 | 0.152 | 0.152 | 0.122 | 0.122 | 0.125 | 0.122 | 0.247 | 0.125 | 0.130 | 0.000 | 0.078 | 0.121 | 0.124 |
| **rankB2_s1_raw** | 0.149 | 0.149 | 0.145 | 0.145 | 0.118 | 0.124 | 0.122 | 0.122 | 0.245 | 0.126 | 0.126 | 0.078 | 0.000 | 0.122 | 0.122 |
| **rankB2_s2_avg** | 0.155 | 0.155 | 0.148 | 0.149 | 0.116 | 0.116 | 0.121 | 0.123 | 0.245 | 0.134 | 0.134 | 0.121 | 0.122 | 0.000 | 0.080 |
| **rankB2_s2_raw** | 0.154 | 0.154 | 0.147 | 0.150 | 0.117 | 0.118 | 0.123 | 0.126 | 0.245 | 0.130 | 0.132 | 0.124 | 0.122 | 0.080 | 0.000 |

| pair type | n pairs | median | min | max |
|---|---|---|---|---|
| same training run, raw vs checkpoint-average | 7 | 0.080 | 0.073 | 0.090 |
| two different naive runs | 4 | 0.124 | 0.123 | 0.128 |
| two different reward-trained runs (ours / rankB1 / rankB2) | 40 | 0.122 | 0.114 | 0.134 |
| a reward-trained run vs a naive run | 40 | 0.152 | 0.143 | 0.160 |

**This is the one clean structural result.** Two *different* reward-trained runs disagree per prompt with sd 0.122 (0.114-0.134) and two naive runs with 0.124 (0.123-0.128), but a reward run against a naive run disagrees with sd 0.152 (0.143-0.160) -- the two ranges do not overlap at all (40 cross pairs all above every one of the 44 within-family pairs). The reward recipe therefore does change *which* prompts a student gets right, even though the category and overall means are inside 0.003. rankB1 and rankB2 sit squarely inside the reward family: they are not distinguishable from `ours`.

## 3a. Every run vs `ours_s0_avg` (CompBench)

| run | mean diff | sd | sem | >0.1 | n+/n- | wilcoxon p | sign p | paired t p |
|---|---|---|---|---|---|---|---|---|
| naive_s0_avg | -0.0017 | 0.155 | 0.0032 | 0.204 | 937/1069 | 0.0028 | 0.0034 | 0.5938 |
| naive_s0_raw | -0.0031 | 0.152 | 0.0031 | 0.196 | 925/1081 | 0.0014 | 0.0005 | 0.3258 |
| naive_s2_avg | -0.0025 | 0.150 | 0.0031 | 0.197 | 946/1057 | 0.0446 | 0.0140 | 0.4188 |
| naive_s2_raw | -0.0018 | 0.152 | 0.0031 | 0.201 | 964/1046 | 0.0379 | 0.0708 | 0.5557 |
| ours_s0_raw | 0.0001 | 0.073 | 0.0015 | 0.049 | 954/953 | 0.7320 | 1.0000 | 0.9547 |
| ours_s2_avg | 0.0007 | 0.114 | 0.0023 | 0.123 | 1009/958 | 0.2683 | 0.2596 | 0.7653 |
| ours_s2_raw | 0.0001 | 0.115 | 0.0024 | 0.127 | 990/985 | 0.6603 | 0.9283 | 0.9521 |
| rankA_s0_avg *(8 cats)* | -0.0819 | 0.242 | 0.0049 | 0.393 | 642/1421 | 3.7e-80 | 2.3e-67 | 2.3e-58 |
| rankA_s0_raw | -0.0863 | 0.245 | 0.0050 | 0.403 | 634/1431 | 3.2e-85 | 1.9e-70 | 9.1e-63 |
| rankB1_s0_avg | -0.0009 | 0.122 | 0.0025 | 0.128 | 984/991 | 0.4900 | 0.8926 | 0.7037 |
| rankB1_s0_raw | -0.0010 | 0.119 | 0.0024 | 0.128 | 950/1024 | 0.0837 | 0.1003 | 0.6652 |
| rankB2_s1_avg | 0.0007 | 0.122 | 0.0025 | 0.137 | 997/977 | 0.9254 | 0.6689 | 0.7667 |
| rankB2_s1_raw | -0.0003 | 0.118 | 0.0024 | 0.134 | 993/980 | 0.9888 | 0.7870 | 0.8940 |
| rankB2_s2_avg | -0.0010 | 0.116 | 0.0024 | 0.132 | 976/993 | 0.4261 | 0.7184 | 0.6628 |
| rankB2_s2_raw | 0.0005 | 0.117 | 0.0024 | 0.138 | 962/1003 | 0.5219 | 0.3669 | 0.8207 |

GenEval2 vs `ours_s0_avg` (per-prompt score, x100):

| run | mean diff x100 | sd | n+/n- | wilcoxon p | sign p | object (atom) | count (atom) | attribute (atom) | position (atom) | verb (atom) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_s0_avg | -0.004 | 0.230 | 379/421 | 0.8711 | 0.1471 | 0.87 (0.8979) | -0.97 (0.3273) | 0.73 (0.7422) | -1.31 (0.0098) | 1.94 (0.2964) |
| naive_s0_raw | -0.681 | 0.231 | 383/417 | 0.6618 | 0.2433 | 0.71 (0.8235) | -0.86 (0.4937) | 0.60 (0.3896) | -0.78 (0.1851) | 1.98 (0.6558) |
| naive_s2_avg | -0.583 | 0.234 | 404/396 | 0.8418 | 0.8045 | 0.76 (0.2601) | -1.29 (0.8390) | 0.29 (0.3770) | 0.42 (0.8343) | 0.54 (0.5509) |
| naive_s2_raw | -0.282 | 0.231 | 397/403 | 0.9606 | 0.8597 | 1.41 (0.0165) | -1.12 (0.6665) | 1.51 (0.0548) | 0.17 (0.2574) | 1.73 (0.8514) |
| ours_s0_raw | -0.137 | 0.110 | 394/406 | 0.9047 | 0.6974 | 0.08 (0.4472) | -0.32 (0.1721) | 0.06 (0.2142) | -0.10 (0.8711) | 0.51 (0.7418) |
| ours_s2_avg | 0.612 | 0.175 | 410/390 | 0.2965 | 0.5018 | -0.01 (0.5096) | -0.86 (0.8744) | 0.21 (0.8518) | 0.00 (0.6837) | 0.60 (0.2694) |
| ours_s2_raw | 1.349 | 0.187 | 404/396 | 0.2168 | 0.8045 | 0.41 (0.1071) | 0.17 (0.5883) | -0.02 (0.3270) | 0.38 (0.7368) | -1.07 (0.1729) |
| rankA_s0_raw | -6.822 | 0.323 | 344/456 | 4.0e-08 | 8.5e-05 | -20.79 (3.5e-111) | -13.26 (1.5e-21) | -10.26 (2.9e-24) | -8.90 (2.2e-06) | -13.99 (0.0142) |
| rankB1_s0_avg | 0.370 | 0.177 | 410/390 | 0.4133 | 0.5018 | 0.14 (0.1094) | 0.34 (0.6996) | -0.15 (0.5520) | 0.50 (0.3761) | 1.24 (0.3999) |
| rankB1_s0_raw | 0.488 | 0.193 | 408/392 | 0.5036 | 0.5959 | 0.02 (0.7695) | 0.28 (0.7039) | -0.56 (0.4487) | -0.20 (0.8612) | 0.68 (0.5225) |
| rankB2_s1_avg | 0.742 | 0.181 | 421/379 | 0.1937 | 0.1471 | 0.61 (0.0438) | -0.17 (0.4406) | 0.91 (0.0406) | 1.15 (0.2261) | 2.19 (0.6809) |
| rankB2_s1_raw | 1.113 | 0.185 | 417/383 | 0.0658 | 0.2433 | 0.75 (0.0102) | 0.02 (0.4176) | 1.18 (0.0234) | 0.79 (0.1114) | 0.42 (0.8078) |
| rankB2_s2_avg | 0.479 | 0.189 | 395/405 | 0.5599 | 0.7504 | 0.92 (0.2963) | -0.12 (0.7278) | -0.04 (0.2311) | 0.30 (0.3869) | 0.07 (0.9228) |
| rankB2_s2_raw | -0.113 | 0.180 | 393/407 | 0.5956 | 0.6458 | 0.42 (0.6086) | -0.47 (0.6473) | -0.41 (0.9219) | 0.16 (0.5873) | 2.51 (0.3424) |

Per-category CompBench mean difference vs `ours_s0_avg` (Wilcoxon p in brackets, n=300 per cell):

| run | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive_s0_avg | -0.0040 (0.0197) | -0.0099 (0.5122) | -0.0069 (0.0730) | +0.0274 (0.2987) | +0.0039 (0.9218) | -0.0156 (0.1761) | -0.0006 (0.5724) | -0.0077 (0.0864) |
| naive_s0_raw | -0.0081 (0.0053) | -0.0123 (0.1924) | -0.0038 (0.3338) | +0.0160 (0.9969) | -0.0003 (0.8331) | -0.0105 (0.2999) | -0.0005 (0.4024) | -0.0048 (0.0946) |
| naive_s2_avg | +0.0032 (0.0501) | -0.0107 (0.1982) | -0.0013 (0.8240) | +0.0026 (0.7536) | -0.0043 (0.9264) | -0.0018 (0.9428) | -0.0009 (0.3798) | -0.0066 (0.0759) |
| naive_s2_raw | +0.0023 (0.0556) | -0.0093 (0.2006) | +0.0007 (0.7551) | +0.0035 (0.5857) | +0.0008 (0.6759) | -0.0065 (0.7774) | -0.0007 (0.5682) | -0.0053 (0.1666) |
| ours_s0_raw | -0.0016 (0.6114) | +0.0023 (0.8907) | -0.0005 (0.9087) | +0.0031 (0.6010) | +0.0015 (0.2456) | -0.0049 (0.6139) | +0.0001 (0.1441) | +0.0007 (0.3853) |
| ours_s2_avg | +0.0052 (0.1615) | -0.0103 (0.5438) | +0.0034 (0.1826) | +0.0058 (0.7096) | +0.0021 (0.3547) | -0.0014 (0.9147) | +0.0001 (0.9954) | +0.0007 (0.7165) |
| ours_s2_raw | +0.0012 (0.8586) | -0.0054 (0.3119) | +0.0037 (0.6222) | +0.0005 (0.9172) | -0.0004 (0.6153) | +0.0020 (0.9012) | +0.0001 (0.8077) | -0.0006 (0.8528) |
| rankA_s0_avg | -0.0794 (5.0e-10) | -0.0298 (0.0520) | -0.0749 (4.8e-11) | -0.0899 (6.5e-05) | -0.1439 (3.6e-28) | -0.1793 (9.9e-21) | -0.0042 (0.0056) | -0.0536 (2.1e-17) |
| rankA_s0_raw | -0.0819 (4.2e-10) | -0.0346 (0.0152) | -0.0877 (9.9e-13) | -0.0871 (0.0002) | -0.1501 (2.1e-29) | -0.1864 (3.5e-22) | -0.0048 (0.0008) | -0.0578 (9.4e-18) |
| rankB1_s0_avg | -0.0068 (0.0344) | -0.0080 (0.3373) | +0.0105 (0.2178) | +0.0028 (0.8317) | -0.0051 (0.4448) | -0.0008 (0.9855) | +0.0005 (0.2123) | -0.0007 (0.7665) |
| rankB1_s0_raw | -0.0075 (0.1172) | -0.0095 (0.0739) | +0.0084 (0.6765) | -0.0049 (0.3624) | -0.0037 (0.0868) | +0.0093 (0.3080) | +0.0003 (0.3425) | -0.0009 (0.7956) |
| rankB2_s1_avg | -0.0068 (0.1494) | +0.0007 (0.5819) | +0.0030 (0.8437) | +0.0049 (0.4537) | +0.0016 (0.6669) | +0.0009 (0.9988) | +0.0004 (0.5423) | +0.0013 (0.6693) |
| rankB2_s1_raw | -0.0017 (0.3479) | +0.0042 (0.9931) | +0.0030 (0.7533) | +0.0021 (0.4593) | +0.0056 (0.2616) | -0.0166 (0.0631) | +0.0006 (0.2019) | +0.0002 (0.9846) |
| rankB2_s2_avg | +0.0039 (0.8607) | -0.0068 (0.1103) | -0.0003 (0.9525) | -0.0055 (0.8134) | -0.0004 (0.1808) | +0.0000 (0.8508) | +0.0001 (0.7314) | +0.0006 (0.8092) |
| rankB2_s2_raw | +0.0058 (0.7218) | -0.0092 (0.1072) | +0.0024 (0.8076) | -0.0035 (0.9602) | -0.0053 (0.3189) | +0.0123 (0.2839) | +0.0004 (0.6335) | +0.0016 (0.7010) |

## 3b. Every run vs `naive_s0_avg` (CompBench)

| run | mean diff | sd | sem | >0.1 | n+/n- | wilcoxon p | sign p | paired t p |
|---|---|---|---|---|---|---|---|---|
| naive_s0_raw | -0.0014 | 0.087 | 0.0018 | 0.063 | 933/999 | 0.0602 | 0.1392 | 0.4441 |
| naive_s2_avg | -0.0008 | 0.123 | 0.0025 | 0.147 | 1008/978 | 0.2131 | 0.5152 | 0.7548 |
| naive_s2_raw | -0.0001 | 0.128 | 0.0026 | 0.148 | 1016/963 | 0.0901 | 0.2424 | 0.9566 |
| ours_s0_avg | 0.0017 | 0.155 | 0.0032 | 0.204 | 1069/937 | 0.0028 | 0.0034 | 0.5938 |
| ours_s0_raw | 0.0018 | 0.154 | 0.0031 | 0.204 | 1094/917 | 0.0013 | 8.6e-05 | 0.5730 |
| ours_s2_avg | 0.0024 | 0.149 | 0.0030 | 0.207 | 1087/931 | 0.0011 | 0.0006 | 0.4328 |
| ours_s2_raw | 0.0018 | 0.158 | 0.0032 | 0.212 | 1084/933 | 0.0005 | 0.0008 | 0.5712 |
| rankA_s0_avg *(8 cats)* | -0.0802 | 0.243 | 0.0050 | 0.407 | 702/1374 | 1.7e-69 | 6.0e-50 | 1.3e-55 |
| rankA_s0_raw | -0.0846 | 0.247 | 0.0050 | 0.413 | 671/1406 | 4.0e-76 | 1.4e-59 | 8.1e-60 |
| rankB1_s0_avg | 0.0007 | 0.155 | 0.0032 | 0.212 | 1059/955 | 0.0240 | 0.0217 | 0.8149 |
| rankB1_s0_raw | 0.0006 | 0.157 | 0.0032 | 0.211 | 1074/937 | 0.0044 | 0.0024 | 0.8420 |
| rankB2_s1_avg | 0.0024 | 0.156 | 0.0032 | 0.211 | 1090/945 | 0.0026 | 0.0014 | 0.4450 |
| rankB2_s1_raw | 0.0014 | 0.149 | 0.0030 | 0.205 | 1063/947 | 0.0123 | 0.0103 | 0.6524 |
| rankB2_s2_avg | 0.0007 | 0.155 | 0.0032 | 0.204 | 1072/949 | 0.0049 | 0.0066 | 0.8361 |
| rankB2_s2_raw | 0.0022 | 0.154 | 0.0032 | 0.208 | 1071/953 | 0.0046 | 0.0093 | 0.4790 |

GenEval2 vs `naive_s0_avg` (per-prompt score, x100):

| run | mean diff x100 | sd | n+/n- | wilcoxon p | sign p | object (atom) | count (atom) | attribute (atom) | position (atom) | verb (atom) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_s0_raw | -0.677 | 0.110 | 401/399 | 0.2937 | 0.9718 | -0.17 (0.5756) | 0.11 (0.1784) | -0.13 (0.9933) | 0.53 (0.0637) | 0.04 (0.3579) |
| naive_s2_avg | -0.579 | 0.183 | 403/397 | 0.4663 | 0.8597 | -0.11 (0.1305) | -0.31 (0.5115) | -0.44 (0.5337) | 1.73 (0.0463) | -1.40 (0.5310) |
| naive_s2_raw | -0.278 | 0.182 | 412/388 | 0.8677 | 0.4161 | 0.53 (0.0016) | -0.15 (0.8095) | 0.77 (0.0360) | 1.48 (0.0471) | -0.21 (0.7746) |
| ours_s0_avg | 0.004 | 0.230 | 421/379 | 0.8711 | 0.1471 | -0.87 (0.8979) | 0.97 (0.3273) | -0.73 (0.7422) | 1.31 (0.0098) | -1.94 (0.2964) |
| ours_s0_raw | -0.133 | 0.228 | 414/386 | 0.6712 | 0.3398 | -0.79 (0.8301) | 0.66 (0.5554) | -0.67 (0.8009) | 1.21 (0.0563) | -1.42 (0.7812) |
| ours_s2_avg | 0.616 | 0.234 | 420/380 | 0.5840 | 0.1679 | -0.88 (0.7712) | 0.11 (0.6185) | -0.52 (0.8326) | 1.31 (0.1087) | -1.34 (0.8717) |
| ours_s2_raw | 1.352 | 0.226 | 417/383 | 0.5506 | 0.2433 | -0.46 (0.8863) | 1.15 (0.2851) | -0.75 (0.8524) | 1.69 (0.1172) | -3.01 (0.3106) |
| rankA_s0_raw | -6.818 | 0.320 | 341/459 | 1.5e-09 | 3.4e-05 | -21.66 (3.0e-112) | -12.28 (1.5e-19) | -10.99 (1.5e-23) | -7.59 (0.0012) | -15.93 (0.0831) |
| rankB1_s0_avg | 0.374 | 0.230 | 416/384 | 0.6831 | 0.2731 | -0.73 (0.2129) | 1.31 (0.1848) | -0.89 (0.2247) | 1.81 (0.0015) | -0.70 (0.8111) |
| rankB1_s0_raw | 0.492 | 0.232 | 428/372 | 0.3605 | 0.0518 | -0.86 (0.3776) | 1.25 (0.1271) | -1.29 (0.4146) | 1.11 (0.0464) | -1.26 (0.7095) |
| rankB2_s1_avg | 0.745 | 0.240 | 408/392 | 0.8499 | 0.5959 | -0.26 (0.3745) | 0.80 (0.2573) | 0.18 (0.3408) | 2.46 (0.0170) | 0.26 (0.3716) |
| rankB2_s1_raw | 1.117 | 0.240 | 420/380 | 0.3645 | 0.1679 | -0.13 (0.3105) | 0.99 (0.1576) | 0.45 (0.3717) | 2.10 (0.0160) | -1.52 (0.7224) |
| rankB2_s2_avg | 0.482 | 0.227 | 411/389 | 0.9157 | 0.4578 | 0.04 (0.6235) | 0.86 (0.3176) | -0.77 (0.8381) | 1.61 (0.0106) | -1.87 (0.8011) |
| rankB2_s2_raw | -0.109 | 0.224 | 394/406 | 0.2687 | 0.6974 | -0.45 (0.6711) | 0.50 (0.8121) | -1.14 (0.6436) | 1.47 (0.0534) | 0.57 (0.6281) |

Per-category CompBench mean difference vs `naive_s0_avg` (Wilcoxon p in brackets, n=300 per cell):

| run | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive_s0_raw | -0.0041 (0.0013) | -0.0024 (0.0859) | +0.0031 (0.0712) | -0.0114 (0.1763) | -0.0042 (0.4216) | +0.0051 (0.4042) | +0.0000 (0.9158) | +0.0029 (0.7150) |
| naive_s2_avg | +0.0072 (0.0555) | -0.0008 (0.8634) | +0.0056 (0.0526) | -0.0248 (0.0958) | -0.0082 (0.6805) | +0.0138 (0.2004) | -0.0003 (0.3482) | +0.0011 (0.9968) |
| naive_s2_raw | +0.0062 (0.0791) | +0.0006 (0.6191) | +0.0076 (0.0592) | -0.0240 (0.2086) | -0.0031 (0.5497) | +0.0091 (0.3174) | -0.0001 (0.7386) | +0.0024 (0.3379) |
| ours_s0_avg | +0.0040 (0.0197) | +0.0099 (0.5122) | +0.0069 (0.0730) | -0.0274 (0.2987) | -0.0039 (0.9218) | +0.0156 (0.1761) | +0.0006 (0.5724) | +0.0077 (0.0864) |
| ours_s0_raw | +0.0024 (0.0114) | +0.0122 (0.2894) | +0.0064 (0.0812) | -0.0244 (0.7693) | -0.0023 (0.6814) | +0.0107 (0.2662) | +0.0007 (0.2492) | +0.0084 (0.0599) |
| ours_s2_avg | +0.0091 (0.0006) | -0.0003 (0.9525) | +0.0103 (0.0095) | -0.0217 (0.2185) | -0.0018 (0.9956) | +0.0142 (0.2311) | +0.0007 (0.4363) | +0.0084 (0.0380) |
| ours_s2_raw | +0.0052 (0.0071) | +0.0045 (0.5814) | +0.0107 (0.0118) | -0.0270 (0.4807) | -0.0043 (0.9646) | +0.0176 (0.1482) | +0.0007 (0.5941) | +0.0071 (0.0546) |
| rankA_s0_avg | -0.0754 (4.0e-07) | -0.0199 (0.1102) | -0.0680 (9.7e-10) | -0.1173 (5.8e-07) | -0.1478 (5.0e-27) | -0.1638 (1.9e-18) | -0.0036 (0.0268) | -0.0459 (3.3e-13) |
| rankA_s0_raw | -0.0780 (2.8e-08) | -0.0247 (0.0409) | -0.0807 (1.4e-10) | -0.1145 (2.8e-06) | -0.1540 (3.9e-28) | -0.1708 (1.8e-19) | -0.0042 (0.0073) | -0.0501 (7.9e-15) |
| rankB1_s0_avg | -0.0028 (0.7045) | +0.0019 (0.8432) | +0.0175 (0.0250) | -0.0247 (0.4395) | -0.0090 (0.9500) | +0.0147 (0.1331) | +0.0011 (0.1187) | +0.0070 (0.1311) |
| rankB1_s0_raw | -0.0035 (0.3021) | +0.0004 (0.7037) | +0.0153 (0.0323) | -0.0323 (0.3950) | -0.0075 (0.6677) | +0.0249 (0.0108) | +0.0009 (0.1837) | +0.0068 (0.0947) |
| rankB2_s1_avg | -0.0028 (0.3484) | +0.0106 (0.2003) | +0.0099 (0.1405) | -0.0225 (0.9172) | -0.0023 (0.7225) | +0.0164 (0.2051) | +0.0010 (0.2687) | +0.0090 (0.1146) |
| rankB2_s1_raw | +0.0023 (0.2252) | +0.0142 (0.1098) | +0.0099 (0.4016) | -0.0253 (0.4350) | +0.0018 (0.2973) | -0.0010 (0.7890) | +0.0011 (0.1030) | +0.0078 (0.1466) |
| rankB2_s2_avg | +0.0079 (0.0178) | +0.0032 (0.9764) | +0.0066 (0.1262) | -0.0329 (0.3149) | -0.0043 (0.8541) | +0.0156 (0.1499) | +0.0007 (0.3598) | +0.0083 (0.0454) |
| rankB2_s2_raw | +0.0097 (0.0500) | +0.0007 (0.9308) | +0.0093 (0.0724) | -0.0310 (0.2364) | -0.0092 (0.8325) | +0.0279 (0.0173) | +0.0009 (0.1665) | +0.0093 (0.0643) |

## 3c. Sign-test effect size: excess prompts won against each reference

`excess` = (prompts where the run scores higher) - (prompts where it scores lower). The mean difference is swamped by large symmetric per-prompt swings; the *direction* is not.

vs `naive_s0_avg`:

| run | family | n+ | n- | excess | win rate | sign p |
|---|---|---|---|---|---|---|
| naive_s0_raw | naive | 933 | 999 | -66 | 0.483 | 0.1392 |
| naive_s2_avg | naive | 1008 | 978 | +30 | 0.508 | 0.5152 |
| naive_s2_raw | naive | 1016 | 963 | +53 | 0.513 | 0.2424 |
| ours_s0_avg | reward | 1069 | 937 | +132 | 0.533 | 0.0034 |
| ours_s0_raw | reward | 1094 | 917 | +177 | 0.544 | 8.6e-05 |
| ours_s2_avg | reward | 1087 | 931 | +156 | 0.539 | 0.0006 |
| ours_s2_raw | reward | 1084 | 933 | +151 | 0.537 | 0.0008 |
| rankA_s0_avg | rankA | 702 | 1374 | -672 | 0.338 | 6.0e-50 |
| rankA_s0_raw | rankA | 671 | 1406 | -735 | 0.323 | 1.4e-59 |
| rankB1_s0_avg | reward | 1059 | 955 | +104 | 0.526 | 0.0217 |
| rankB1_s0_raw | reward | 1074 | 937 | +137 | 0.534 | 0.0024 |
| rankB2_s1_avg | reward | 1090 | 945 | +145 | 0.536 | 0.0014 |
| rankB2_s1_raw | reward | 1063 | 947 | +116 | 0.529 | 0.0103 |
| rankB2_s2_avg | reward | 1072 | 949 | +123 | 0.530 | 0.0066 |
| rankB2_s2_raw | reward | 1071 | 953 | +118 | 0.529 | 0.0093 |

vs `ours_s0_avg`:

| run | family | n+ | n- | excess | win rate | sign p |
|---|---|---|---|---|---|---|
| naive_s0_avg | naive | 937 | 1069 | -132 | 0.467 | 0.0034 |
| naive_s0_raw | naive | 925 | 1081 | -156 | 0.461 | 0.0005 |
| naive_s2_avg | naive | 946 | 1057 | -111 | 0.472 | 0.0140 |
| naive_s2_raw | naive | 964 | 1046 | -82 | 0.480 | 0.0708 |
| ours_s0_raw | reward | 954 | 953 | +1 | 0.500 | 1.0000 |
| ours_s2_avg | reward | 1009 | 958 | +51 | 0.513 | 0.2596 |
| ours_s2_raw | reward | 990 | 985 | +5 | 0.501 | 0.9283 |
| rankA_s0_avg | rankA | 642 | 1421 | -779 | 0.311 | 2.3e-67 |
| rankA_s0_raw | rankA | 634 | 1431 | -797 | 0.307 | 1.9e-70 |
| rankB1_s0_avg | reward | 984 | 991 | -7 | 0.498 | 0.8926 |
| rankB1_s0_raw | reward | 950 | 1024 | -74 | 0.481 | 0.1003 |
| rankB2_s1_avg | reward | 997 | 977 | +20 | 0.505 | 0.6689 |
| rankB2_s1_raw | reward | 993 | 980 | +13 | 0.503 | 0.7870 |
| rankB2_s2_avg | reward | 976 | 993 | -17 | 0.496 | 0.7184 |
| rankB2_s2_raw | reward | 962 | 1003 | -41 | 0.490 | 0.3669 |

## 3d. Pooled over seeds (avg checkpoints only)

Averaging the per-prompt score over the two training seeds of each recipe halves the run noise. `CONTROL_crossed_pools` pools one naive seed with one ours seed on each side, so it has exactly the same amount of seed and checkpoint noise but zero recipe contrast -- it is the null for these rows.

| pooled comparison | mean x | mean y | mean diff | sem | n+/n- | wilcoxon p | sign p | t p | GE2 diff x100 (p) |
|---|---|---|---|---|---|---|---|---|---|
| ours - naive | 0.4882 | 0.4858 | 0.0024 | 0.0025 | 1101/957 | 0.0081 | 0.0016 | 0.3350 | 0.600 (0.6112) |
| rankB1 - naive | 0.4869 | 0.4858 | 0.0011 | 0.0029 | 1071/987 | 0.1457 | 0.0673 | 0.6990 | 0.663 (0.7626) |
| rankB2 - naive | 0.4877 | 0.4858 | 0.0019 | 0.0026 | 1076/996 | 0.0462 | 0.0826 | 0.4519 | 0.904 (0.6243) |
| rankB1 - ours | 0.4869 | 0.4882 | -0.0013 | 0.0023 | 1004/1026 | 0.4519 | 0.6412 | 0.5690 | 0.064 (0.1950) |
| rankB2 - ours | 0.4877 | 0.4882 | -0.0005 | 0.0018 | 1002/1021 | 0.7895 | 0.6890 | 0.7809 | 0.304 (0.6478) |
| CONTROL_crossed_pools | 0.4874 | 0.4867 | 0.0007 | 0.0017 | 1018/1021 | 0.5596 | 0.9647 | 0.6653 | 0.596 (0.0744) |

Per-category pooled mean difference, `diff (paired-t p / Wilcoxon p)`. CompBench's headline number is a *mean*, so the t column is the one that matters for the reported score; the Wilcoxon column says whether the shift is broad or carried by a minority of prompts.

| pooled comparison | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| ours - naive | +0.0030 (0.6569 / 0.0045) | +0.0052 (0.4937 / 0.6725) | +0.0058 (0.4286 / 0.0304) | -0.0121 (0.2554 / 0.5030) | +0.0012 (0.8542 / 0.5612) | +0.0080 (0.3719 / 0.1852) | +0.0008 (0.2046 / 0.2105) | +0.0075 (0.0336 / 0.1549) |
| rankB1 - naive | -0.0064 (0.4167 / 0.5533) | +0.0023 (0.7715 / 0.6075) | +0.0147 (0.0788 / 0.0678) | -0.0123 (0.3678 / 0.7922) | -0.0049 (0.4790 / 0.2583) | +0.0078 (0.4554 / 0.4545) | +0.0012 (0.0551 / 0.0465) | +0.0065 (0.0896 / 0.1379) |
| rankB2 - naive | -0.0010 (0.8670 / 0.2928) | +0.0073 (0.2641 / 0.5177) | +0.0054 (0.4896 / 0.3151) | -0.0153 (0.1885 / 0.6779) | +0.0008 (0.9047 / 0.7696) | +0.0091 (0.3364 / 0.1502) | +0.0010 (0.1127 / 0.1837) | +0.0081 (0.0127 / 0.0578) |
| rankB1 - ours | -0.0094 (0.0591 / 0.0044) | -0.0029 (0.6400 / 0.6451) | +0.0088 (0.0946 / 0.4154) | -0.0001 (0.9906 / 0.6069) | -0.0061 (0.3620 / 0.2161) | -0.0001 (0.9879 / 0.8447) | +0.0005 (0.2954 / 0.2282) | -0.0010 (0.5583 / 0.8719) |
| rankB2 - ours | -0.0040 (0.2409 / 0.0293) | +0.0021 (0.6539 / 0.7292) | -0.0004 (0.9236 / 0.3050) | -0.0032 (0.7167 / 0.7831) | -0.0005 (0.9271 / 0.9722) | +0.0011 (0.8764 / 0.6891) | +0.0002 (0.5972 / 0.7649) | +0.0006 (0.7188 / 0.1837) |
| CONTROL_crossed_pools | -0.0010 (0.7717 / 0.5858) | -0.0047 (0.2461 / 0.5078) | -0.0011 (0.7976 / 0.6804) | +0.0153 (0.0526 / 0.1862) | +0.0051 (0.3615 / 0.8144) | -0.0076 (0.2596 / 0.1757) | +0.0002 (0.5378 / 0.4146) | -0.0002 (0.8973 / 0.9820) |

Two cells stand out and both are reproduced by all three reward recipes while the crossed-pool control sits at zero:

* **complex**: ours +0.0075 (t p 0.034), rankB2 +0.0081 (t p 0.013), rankB1 +0.0065 (t p 0.090) against naive; control -0.0002 (t p 0.90). A mean shift carried by a minority of prompts (Wilcoxon 0.06-0.15).
* **color**: ours +0.0030 against naive with Wilcoxon p 0.0045 and sign p 0.0022 but t p 0.66 -- a broad, tiny, many-prompt nudge, the mirror image of complex. rankB1 is the exception (-0.0064, and -0.0094 against ours, Wilcoxon p 0.0044).
* **spatial** looks like the largest cell everywhere (-0.012 to -0.015) but the crossed-pool control produces +0.0153 on the same category, larger and opposite in sign: 2D-spatial simply has the worst run-to-run scatter (control sd 0.0086) and `naive_s0_avg` drew a high value (0.2411 vs 0.2163 at seed 2).

## 4. Do the differences concentrate in the targeted categories?

`cos` is the cosine between the mean-centred 8-category difference vector and the mean-centred indicator of the six targeted categories (+1 = the arm moves exactly the targeted categories, -1 = exactly the untargeted ones, 0 = no relation). `L2` is the norm of the category-difference vector; the control pairs give the scale it has to beat. `target - other` is the difference in mean CompBench between the targeted and untargeted categories.

Per-category sd across the control pairs (the per-category noise floor):

| color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|
| 0.0055 | 0.0057 | 0.0050 | 0.0085 | 0.0047 | 0.0099 | 0.0002 | 0.0018 |

Control-pair category-vector L2 norms: median 0.0141, max 0.0350.

| comparison | L2 | cos w/ target mask | target mean | other mean | target - other | largest cat |
|---|---|---|---|---|---|---|
| naive_s0_avg - ours_s0_avg | 0.0351 | 0.115 | -0.0009 | -0.0041 | 0.0032 | spatial |
| naive_s0_raw - ours_s0_avg | 0.0249 | -0.027 | -0.0032 | -0.0026 | -0.0005 | spatial |
| naive_s2_avg - ours_s0_avg | 0.0142 | 0.166 | -0.0021 | -0.0037 | 0.0017 | shape |
| naive_s2_raw - ours_s0_avg | 0.0133 | 0.156 | -0.0014 | -0.0030 | 0.0016 | shape |
| ours_s0_raw - ours_s0_avg | 0.0066 | -0.077 | -0.0000 | 0.0004 | -0.0004 | numeracy |
| ours_s2_avg - ours_s0_avg | 0.0136 | 0.032 | 0.0008 | 0.0004 | 0.0003 | shape |
| ours_s2_raw - ours_s0_avg | 0.0070 | 0.089 | 0.0003 | -0.0002 | 0.0005 | shape |
| rankA_s0_avg - ours_s0_avg | 0.2768 | -0.571 | -0.0995 | -0.0289 | -0.0707 | numeracy |
| rankA_s0_raw - ours_s0_avg | 0.2895 | -0.577 | -0.1046 | -0.0313 | -0.0734 | numeracy |
| rankB1_s0_avg - ours_s0_avg | 0.0160 | -0.091 | -0.0012 | -0.0001 | -0.0012 | texture |
| rankB1_s0_raw - ours_s0_avg | 0.0185 | -0.070 | -0.0013 | -0.0003 | -0.0010 | shape |
| rankB2_s1_avg - ours_s0_avg | 0.0092 | -0.021 | 0.0007 | 0.0009 | -0.0002 | color |
| rankB2_s1_raw - ours_s0_avg | 0.0185 | -0.060 | -0.0005 | 0.0004 | -0.0009 | numeracy |
| rankB2_s2_avg - ours_s0_avg | 0.0096 | -0.253 | -0.0015 | 0.0004 | -0.0019 | shape |
| rankB2_s2_raw - ours_s0_avg | 0.0179 | -0.042 | 0.0004 | 0.0010 | -0.0006 | numeracy |
| naive_s0_raw - naive_s0_avg | 0.0146 | -0.329 | -0.0023 | 0.0015 | -0.0038 | spatial |
| naive_s2_avg - naive_s0_avg | 0.0309 | -0.064 | -0.0012 | 0.0004 | -0.0016 | spatial |
| naive_s2_raw - naive_s0_avg | 0.0277 | -0.076 | -0.0006 | 0.0011 | -0.0017 | spatial |
| ours_s0_avg - naive_s0_avg | 0.0351 | -0.115 | 0.0009 | 0.0041 | -0.0032 | spatial |
| ours_s0_raw - naive_s0_avg | 0.0313 | -0.146 | 0.0009 | 0.0045 | -0.0037 | spatial |
| ours_s2_avg - naive_s0_avg | 0.0306 | -0.120 | 0.0017 | 0.0046 | -0.0029 | spatial |
| ours_s2_raw - naive_s0_avg | 0.0356 | -0.096 | 0.0011 | 0.0039 | -0.0027 | spatial |
| rankA_s0_avg - naive_s0_avg | 0.2743 | -0.587 | -0.0987 | -0.0248 | -0.0739 | numeracy |
| rankA_s0_raw - naive_s0_avg | 0.2859 | -0.600 | -0.1038 | -0.0272 | -0.0766 | numeracy |
| rankB1_s0_avg - naive_s0_avg | 0.0357 | -0.153 | -0.0004 | 0.0041 | -0.0044 | spatial |
| rankB1_s0_raw - naive_s0_avg | 0.0449 | -0.118 | -0.0004 | 0.0038 | -0.0043 | spatial |
| rankB2_s1_avg - naive_s0_avg | 0.0329 | -0.130 | 0.0016 | 0.0050 | -0.0034 | spatial |
| rankB2_s1_raw - naive_s0_avg | 0.0318 | -0.162 | 0.0003 | 0.0045 | -0.0041 | spatial |
| rankB2_s2_avg - naive_s0_avg | 0.0391 | -0.162 | -0.0006 | 0.0045 | -0.0051 | spatial |
| rankB2_s2_raw - naive_s0_avg | 0.0457 | -0.105 | 0.0013 | 0.0051 | -0.0039 | spatial |
| NOISE:naive_s0_raw - naive_s0_avg | 0.0146 | -0.329 | -0.0023 | 0.0015 | -0.0038 | spatial |
| NOISE:ours_s0_raw - ours_s0_avg | 0.0066 | -0.077 | -0.0000 | 0.0004 | -0.0004 | numeracy |
| NOISE:naive_s2_raw - naive_s2_avg | 0.0076 | -0.018 | 0.0006 | 0.0007 | -0.0001 | 3d_spatial |
| NOISE:ours_s2_raw - ours_s2_avg | 0.0094 | 0.020 | -0.0005 | -0.0007 | 0.0002 | spatial |
| NOISE:naive_s2_raw - naive_s0_raw | 0.0177 | 0.147 | 0.0017 | -0.0004 | 0.0021 | spatial |
| NOISE:naive_s2_avg - naive_s0_avg | 0.0309 | -0.064 | -0.0012 | 0.0004 | -0.0016 | spatial |
| NOISE:ours_s2_raw - ours_s0_raw | 0.0120 | 0.094 | 0.0003 | -0.0006 | 0.0009 | shape |
| NOISE:ours_s2_avg - ours_s0_avg | 0.0136 | 0.032 | 0.0008 | 0.0004 | 0.0003 | shape |
| NOISE:rankB2_s2_raw - rankB2_s1_raw | 0.0350 | 0.010 | 0.0009 | 0.0006 | 0.0003 | numeracy |
| NOISE:rankB2_s2_avg - rankB2_s1_avg | 0.0172 | -0.129 | -0.0022 | -0.0005 | -0.0017 | color |
| NOISE:rankB1_s0_raw - rankB1_s0_avg | 0.0131 | 0.013 | -0.0001 | -0.0002 | 0.0001 | numeracy |
| NOISE:rankA_s0_raw - rankA_s0_avg | 0.0175 | -0.270 | -0.0051 | -0.0024 | -0.0027 | texture |

Per-category z-scores (category difference / per-category control sd); |z| > 2 would be a category moving beyond the control scatter:

| comparison | color | shape | texture | spatial | 3d_spatial | numeracy | non_spatial | complex |
|---|---|---|---|---|---|---|---|---|
| naive_s0_avg - ours_s0_avg | -0.72 | -1.74 | -1.40 | 3.21 | 0.83 | -1.56 | -2.51 | -4.23 |
| naive_s0_raw - ours_s0_avg | -1.48 | -2.16 | -0.77 | 1.87 | -0.06 | -1.05 | -2.36 | -2.62 |
| naive_s2_avg - ours_s0_avg | 0.58 | -1.88 | -0.26 | 0.31 | -0.92 | -0.18 | -3.91 | -3.62 |
| naive_s2_raw - ours_s0_avg | 0.41 | -1.63 | 0.13 | 0.41 | 0.17 | -0.65 | -3.09 | -2.92 |
| ours_s0_raw - ours_s0_avg | -0.29 | 0.39 | -0.11 | 0.36 | 0.33 | -0.49 | 0.58 | 0.37 |
| ours_s2_avg - ours_s0_avg | 0.94 | -1.80 | 0.68 | 0.68 | 0.44 | -0.14 | 0.59 | 0.41 |
| ours_s2_raw - ours_s0_avg | 0.22 | -0.94 | 0.75 | 0.05 | -0.09 | 0.20 | 0.49 | -0.32 |
| rankA_s0_avg - ours_s0_avg | -14.43 | -5.22 | -15.06 | -10.51 | -30.66 | -18.03 | -18.99 | -29.48 |
| rankA_s0_raw - ours_s0_avg | -14.89 | -6.05 | -17.62 | -10.18 | -31.99 | -18.73 | -21.57 | -31.79 |
| rankB1_s0_avg - ours_s0_avg | -1.24 | -1.40 | 2.12 | 0.32 | -1.08 | -0.08 | 2.42 | -0.36 |
| rankB1_s0_raw - ours_s0_avg | -1.36 | -1.66 | 1.68 | -0.57 | -0.78 | 0.94 | 1.46 | -0.48 |
| rankB2_s1_avg - ours_s0_avg | -1.23 | 0.12 | 0.60 | 0.57 | 0.34 | 0.09 | 1.84 | 0.72 |
| rankB2_s1_raw - ours_s0_avg | -0.31 | 0.74 | 0.60 | 0.25 | 1.20 | -1.67 | 2.54 | 0.09 |
| rankB2_s2_avg - ours_s0_avg | 0.72 | -1.18 | -0.07 | -0.64 | -0.09 | 0.00 | 0.53 | 0.35 |
| rankB2_s2_raw - ours_s0_avg | 1.05 | -1.62 | 0.47 | -0.41 | -1.14 | 1.24 | 1.67 | 0.89 |
| naive_s0_raw - naive_s0_avg | -0.75 | -0.42 | 0.62 | -1.34 | -0.89 | 0.51 | 0.14 | 1.61 |
| naive_s2_avg - naive_s0_avg | 1.31 | -0.14 | 1.13 | -2.90 | -1.75 | 1.38 | -1.40 | 0.61 |
| naive_s2_raw - naive_s0_avg | 1.13 | 0.11 | 1.53 | -2.80 | -0.66 | 0.91 | -0.59 | 1.31 |
| ours_s0_avg - naive_s0_avg | 0.72 | 1.74 | 1.40 | -3.21 | -0.83 | 1.56 | 2.51 | 4.23 |
| ours_s0_raw - naive_s0_avg | 0.44 | 2.13 | 1.29 | -2.85 | -0.50 | 1.08 | 3.09 | 4.60 |
| ours_s2_avg - naive_s0_avg | 1.66 | -0.06 | 2.08 | -2.53 | -0.39 | 1.42 | 3.09 | 4.64 |
| ours_s2_raw - naive_s0_avg | 0.94 | 0.79 | 2.15 | -3.15 | -0.92 | 1.77 | 3.00 | 3.91 |
| rankA_s0_avg - naive_s0_avg | -13.71 | -3.48 | -13.67 | -13.72 | -31.49 | -16.46 | -16.49 | -25.25 |
| rankA_s0_raw - naive_s0_avg | -14.17 | -4.32 | -16.22 | -13.39 | -32.82 | -17.17 | -19.06 | -27.56 |
| rankB1_s0_avg - naive_s0_avg | -0.52 | 0.34 | 3.52 | -2.89 | -1.91 | 1.48 | 4.93 | 3.86 |
| rankB1_s0_raw - naive_s0_avg | -0.64 | 0.07 | 3.08 | -3.78 | -1.61 | 2.50 | 3.96 | 3.75 |
| rankB2_s1_avg - naive_s0_avg | -0.51 | 1.85 | 1.99 | -2.63 | -0.48 | 1.65 | 4.35 | 4.95 |
| rankB2_s1_raw - naive_s0_avg | 0.42 | 2.48 | 2.00 | -2.96 | 0.38 | -0.10 | 5.05 | 4.31 |
| rankB2_s2_avg - naive_s0_avg | 1.44 | 0.56 | 1.33 | -3.85 | -0.92 | 1.57 | 3.03 | 4.58 |
| rankB2_s2_raw - naive_s0_avg | 1.77 | 0.12 | 1.87 | -3.62 | -1.96 | 2.80 | 4.18 | 5.12 |
| NOISE:naive_s0_raw - naive_s0_avg | -0.75 | -0.42 | 0.62 | -1.34 | -0.89 | 0.51 | 0.14 | 1.61 |
| NOISE:ours_s0_raw - ours_s0_avg | -0.29 | 0.39 | -0.11 | 0.36 | 0.33 | -0.49 | 0.58 | 0.37 |
| NOISE:naive_s2_raw - naive_s2_avg | -0.17 | 0.25 | 0.40 | 0.10 | 1.08 | -0.47 | 0.82 | 0.70 |
| NOISE:ours_s2_raw - ours_s2_avg | -0.72 | 0.85 | 0.07 | -0.62 | -0.53 | 0.34 | -0.10 | -0.73 |
| NOISE:naive_s2_raw - naive_s0_raw | 1.89 | 0.53 | 0.91 | -1.47 | 0.22 | 0.40 | -0.73 | -0.30 |
| NOISE:naive_s2_avg - naive_s0_avg | 1.31 | -0.14 | 1.13 | -2.90 | -1.75 | 1.38 | -1.40 | 0.61 |
| NOISE:ours_s2_raw - ours_s0_raw | 0.51 | -1.34 | 0.86 | -0.30 | -0.42 | 0.69 | -0.09 | -0.69 |
| NOISE:ours_s2_avg - ours_s0_avg | 0.94 | -1.80 | 0.68 | 0.68 | 0.44 | -0.14 | 0.59 | 0.41 |
| NOISE:rankB2_s2_raw - rankB2_s1_raw | 1.35 | -2.36 | -0.13 | -0.66 | -2.34 | 2.90 | -0.87 | 0.81 |
| NOISE:rankB2_s2_avg - rankB2_s1_avg | 1.95 | -1.30 | -0.66 | -1.22 | -0.44 | -0.08 | -1.31 | -0.37 |
| NOISE:rankB1_s0_raw - rankB1_s0_avg | -0.12 | -0.26 | -0.44 | -0.89 | 0.30 | 1.02 | -0.96 | -0.11 |
| NOISE:rankA_s0_raw - rankA_s0_avg | -0.46 | -0.84 | -2.56 | 0.33 | -1.33 | -0.71 | -2.57 | -2.31 |

## 5. Verdict

**Short answer: rankA is catastrophically different, the reward family as a whole is weakly but consistently different from naive, and ours / rankB1 / rankB2 are not distinguishable from each other by anything measured here. Where the reward effect does show up is `complex` and `color`, not the six categories the ranking negatives target.**

1. **rankA is a broken model, unambiguously.** -0.0846 CompBench and -6.82 GenEval2 points against naive, sign p ~ 1e-59, 0.413 of prompts moving by more than 0.1. It is 20-50x the noise floor and it is a *collapse*, concentrated exactly in numeracy (-0.171), 3D-spatial (-0.154), 2D-spatial (-0.115), texture (-0.081) and color (-0.078) with non-spatial (-0.004) untouched -- i.e. the compositional categories, not the caption-similarity ones. On GenEval2 the object atom loses 21.7 points. This is the one arm where the answer is not subtle.

2. **The reward family as a whole beats naive at the prompt level, but rankA aside no arm is distinguishable from any other arm.** Every one of the 10 reward-trained runs (ours, rankB1, rankB2, both seeds, raw and averaged) wins more CompBench prompts against `naive_s0_avg` than it loses (excess +104 to +177, win rate 0.526-0.544, sign p 8.6e-05 to 0.022), while the three naive control runs give -66 to +53 (all ns). Across the 5 *independent* reward training runs the sign is 5/5 positive (binomial p = 0.0625 two-sided, 0.031 one-sided). Pooled over seeds the effect is +0.0024 CompBench (Wilcoxon p 0.0081, sign p 0.0016) against a matched crossed-pool control of +0.0007 (p 0.9647). The paired *t*-test on the same data is p ~ 0.3-0.6, so this is a small consistent nudge riding on very large symmetric per-prompt noise, not a mean shift.

3. **rankB1 and rankB2 are inside `ours`.** Pooled over seeds, rankB1 - ours = -0.0013 (sign p 0.6412) and rankB2 - ours = -0.0005 (sign p 0.6890); every per-category cell is inside the control scatter except color for rankB1 (-0.0094, p 0.0044, |z| ~ 1.7 against the per-category noise sd and not surviving any correction over 8 categories x 5 comparisons). Their per-prompt disagreement with `ours` (0.114-0.134) is the same size as `ours` seed-to-seed. The ranking negatives changed nothing measurable.

4. **The changes do NOT concentrate in the six targeted categories.** For every arm except rankA, the cosine between the 8-category difference vector and the targeted-category mask is between -0.3 and +0.2, the same range the control pairs produce, and `target - other` is +-0.004 or smaller. The one apparently structured signature -- the +color/+texture/+numeracy/-spatial pattern that every arm shows against `naive_s0_avg` -- is reproduced by the *naive seed-2 control* against the same reference (color +0.0072, texture +0.0056, numeracy +0.0138, spatial -0.0248), so it is a property of `naive_s0_avg` being a lucky draw on 2D-spatial (0.2411 vs 0.2163 for naive seed 2), not of any recipe. Once the reference is pooled over seeds the pattern shrinks to +0.003 color / +0.006 texture and the spatial cell loses significance. rankA is the only vector that points anywhere: cos = -0.60, i.e. it destroys the targeted categories specifically.

5. **Where the reward effect actually lives: `complex` and `color`, not the six targeted categories.** Pooled over seeds and against the pooled naive reference, `complex` is the only category with a mean shift the paired *t*-test resolves, and all three reward recipes reproduce it (ours +0.0075 t p 0.034, rankB2 +0.0081 t p 0.013, rankB1 +0.0065 t p 0.090) while the crossed-pool control gives -0.0002 (t p 0.90). `color` shows the opposite profile for ours -- +0.0030 with Wilcoxon p 0.0045 / sign p 0.0022 but t p 0.66, a broad many-prompt nudge rather than a mean shift. Of the six targeted categories, only color moves at all, and `complex` -- the one CompBench category the ranking negatives do *not* target -- carries the mean. Neither cell survives a Bonferroni correction over 8 categories x 5 comparisons; treat both as leads, not results.

### Minimal detectable effect

* **Mean-based, one run vs one run.** n = 2398 CompBench prompts, paired-diff sd 0.114 -> alpha 0.05 / power 0.80 detects **0.0065** CompBench points. GenEval2 (n = 800, sd 0.175) -> **1.73** points x100. Every arm difference in this campaign is 0.001-0.003, i.e. 2-5x *below* this; the mean can never settle these arms at one seed.
* **Sign-based, one run vs one run.** With ~2006 non-tied prompts the detectable win-rate excess is 0.031 (~125 excess prompts). The ours-vs-naive effect (+132 to +177) sits just above this, which is why the sign test fires and the t-test does not. The sign test is roughly 2x more sensitive here and is the statistic to use on this data.
* **Run-level scatter.** Training seed moves the CompBench mean by 0.0001-0.0018; checkpoint averaging by 0.0001-0.0014 (rankA excluded). Two seeds per arm is enough to resolve ~0.004 and no better; resolving the ~0.002 differences between ours / rankB1 / rankB2 by the mean would need roughly 11x the prompt count or the equivalent in images per prompt.
* **What would settle it.** The reward-vs-naive contrast is real but small; the arm-vs-arm contrasts are below every detector used here. The dominant noise term is single-image resampling, not model quality: the paired-diff sd of 0.114 implies a per-image score noise of ~0.081, a quarter of the 0.318 between-prompt score sd, and it is independent across the two runs being compared. Ten images per prompt (the official CompBench protocol, cf. the `eval10_*` dirs) would shrink it by up to ~3x and bring the mean-based MDE to ~0.002, which is the scale these arms actually differ at.

See the JSON for every number; the per-prompt matrices are in `cb_matrix.npy` / `ge2_matrix.npy` with `matrix_keys.json` giving the row (run) and column (prompt) order. Two caveats on the data: the `spatial` category carries 298 prompts, not 300 (the UniDet evaluator drops two, consistently in every run), and `rankA_s0_avg` had no GenEval2 at the time of writing.
