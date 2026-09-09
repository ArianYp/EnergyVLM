# Inference step sweep, averaged seed-0 models, CompBench / GenEval2 (x100), one image per prompt, w = 1

| steps | argmax CompBench / GenEval2 | argmax + exact reward CompBench / GenEval2 |
| 2 | 0.1798 / 2.25 | 0.1794 / 1.90 |
| 4 | 0.4849 / 24.17 | 0.4906 / 24.83 |
| 8 | 0.4996 / 22.76 | 0.4954 / 22.66 |
| 16 | 0.4932 / 20.99 | 0.4891 / 20.42 |
| 28 | 0.4908 / 19.80 | 0.4841 / 19.30 |

Jobs: phaseW/step_sweep_jobs.txt (single-GPU evaluations). Reference: base model 28 steps w=7 scores 0.5053 / 17.05.
