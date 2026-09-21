# Benchmark-prompt selection: evaluator-argmax vs random pick (raw final checkpoints)

Seeds [0, 1, 2]. Both arms: the same 5,559 T2I-CompBench++ TRAIN prompts, 16 teacher candidates each, K=10 grid, converged schedule, 2,780 updates. The ONLY difference is which of the 16 trajectories the student distils: the one the category's own official evaluator scores highest, or a random one. Evaluated on the 2,398 held-out val prompts, 1 image per prompt.

| | seed 0 | seed 1 | seed 2 | mean | CTCal delta | ours - theirs |
|---|---|---|---|---|---|---|
| **CompBench mean** | +0.0134 | +0.0068 | +0.0103 | **+0.0102** | +0.0150 | -0.0048 |
| GenEval2 | +0.0079 | +0.0074 | +0.0186 | +0.0113 | – | – |
| color | +0.0061 | +0.0113 | +0.0092 | +0.0089 | +0.0311 | -0.0222 |
| shape | +0.0173 | +0.0123 | +0.0296 | +0.0197 | +0.0083 | +0.0114 |
| texture | +0.0228 | +0.0184 | +0.0025 | +0.0146 | +0.0247 | -0.0101 |
| spatial | +0.0532 | +0.0157 | +0.0241 | +0.0310 | +0.0276 | +0.0034 |
| 3d_spatial | +0.0039 | -0.0079 | +0.0074 | +0.0012 | +0.0033 | -0.0021 |
| numeracy | -0.0057 | -0.0018 | +0.0049 | -0.0009 | +0.0118 | -0.0127 |
| non_spatial | -0.0002 | -0.0001 | +0.0002 | -0.0000 | +0.0085 | -0.0085 |
| complex | +0.0101 | +0.0068 | +0.0046 | +0.0071 | +0.0043 | +0.0028 |

Pooled per-prompt paired test over 3 seed(s): win rate 0.534, sign p 1.03e-07, Wilcoxon p 8.7e-11, n = 7194.

Absolutes for reference:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| evaluator-argmax | 0.5015 | 0.4967 | 0.4993 |
| random pick | 0.4881 | 0.4898 | 0.4890 |

Our 28-step guided teacher scores 0.5053 on the same pool; our best COCO-trained student 0.4951.
