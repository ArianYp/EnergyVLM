# Benchmark-prompt selection: evaluator-argmax vs random pick (averaged checkpoints)

Seeds [0, 1, 2]. Both arms: the same 5,559 T2I-CompBench++ TRAIN prompts, 16 teacher candidates each, K=10 grid, converged schedule, 2,780 updates. The ONLY difference is which of the 16 trajectories the student distils: the one the category's own official evaluator scores highest, or a random one. Evaluated on the 2,398 held-out val prompts, 1 image per prompt.

| | seed 0 | seed 1 | seed 2 | mean | CTCal delta | ours - theirs |
|---|---|---|---|---|---|---|
| **CompBench mean** | +0.0121 | +0.0106 | +0.0119 | **+0.0115** | +0.0150 | -0.0034 |
| GenEval2 | +0.0107 | +0.0145 | +0.0105 | +0.0119 | – | – |
| color | +0.0139 | +0.0144 | +0.0073 | +0.0119 | +0.0311 | -0.0192 |
| shape | +0.0121 | +0.0139 | +0.0357 | +0.0206 | +0.0083 | +0.0123 |
| texture | +0.0239 | +0.0164 | +0.0109 | +0.0171 | +0.0247 | -0.0076 |
| spatial | +0.0384 | +0.0204 | +0.0193 | +0.0261 | +0.0276 | -0.0015 |
| 3d_spatial | +0.0039 | +0.0043 | +0.0109 | +0.0064 | +0.0033 | +0.0031 |
| numeracy | -0.0080 | +0.0067 | +0.0078 | +0.0021 | +0.0118 | -0.0097 |
| non_spatial | -0.0000 | +0.0003 | +0.0003 | +0.0002 | +0.0085 | -0.0083 |
| complex | +0.0123 | +0.0084 | +0.0027 | +0.0078 | +0.0043 | +0.0035 |

Pooled per-prompt paired test over 3 seed(s): win rate 0.539, sign p 1.89e-09, Wilcoxon p 1.47e-14, n = 7194.

Absolutes for reference:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| evaluator-argmax | 0.5012 | 0.4986 | 0.5011 |
| random pick | 0.4892 | 0.4880 | 0.4893 |

Our 28-step guided teacher scores 0.5053 on the same pool; our best COCO-trained student 0.4951.
