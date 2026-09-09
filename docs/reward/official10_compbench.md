# Official ten-images-per-prompt T2I-CompBench, 3k pool, averaged models (2k/4k/6k), three seeds

| arm | s0 / s1 / s2 | mean +- sd |
|---|---|---|
| random pick | 0.4787 / 0.4778 / 0.4758 | 0.4774 +- 0.0015 |
| argmax | 0.4848 / 0.4926 / 0.4940 | 0.4904 +- 0.0050 |
| argmax + exact reward (first-round runs) | 0.4973 / 0.5023 / 0.4964 | 0.4987 +- 0.0032 |

| contrast | per seed | mean +- sd (p) |
|---|---|---|
| argmax - random pick | +0.0061 / +0.0148 / +0.0182 | +0.0130 +- 0.0062 (p=0.069) |
| argmax + exact reward (first-round runs) - argmax | +0.0125 / +0.0097 / +0.0024 | +0.0082 +- 0.0052 (p=0.112) |
| argmax + exact reward (first-round runs) - random pick | +0.0186 / +0.0245 / +0.0206 | +0.0212 +- 0.0030 (p=0.007) |

Note: the exact-reward models here are the first-round (S4h) runs, whose caption order differed from the argmax arm's at the same seed; the argmax seed-1 run's GenEval2 stage was not completed (16 h limit) and its CompBench mean was assembled from the eight category score files.
