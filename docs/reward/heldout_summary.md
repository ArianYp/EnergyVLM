# Held-out DINO monitor (16 validation captions, offline PIL scorer)

Teacher argmax candidate (8 steps, w=7), same captions: DINO 0.5278

| arm | checkpoint | seeds | x0-hat DINO (mean +- sd over seeds) | 4-step sample DINO (mean +- sd) |
|---|---|---|---|---|
| random pick | step2000 | 3 | 0.5267 +- 0.0019 | 0.4765 +- 0.0158 |
| random pick | step4000 | 3 | 0.5311 +- 0.0008 | 0.4821 +- 0.0111 |
| random pick | final | 3 | 0.5293 +- 0.0011 | 0.4836 +- 0.0144 |
| random pick | avg_last5 | 3 | 0.5314 +- 0.0003 | 0.4900 +- 0.0086 |
| argmax | step2000 | 3 | 0.5287 +- 0.0006 | 0.4836 +- 0.0168 |
| argmax | step4000 | 3 | 0.5288 +- 0.0003 | 0.4949 +- 0.0167 |
| argmax | final | 3 | 0.5292 +- 0.0013 | 0.4918 +- 0.0180 |
| argmax | avg_last5 | 3 | 0.5298 +- 0.0007 | 0.5052 +- 0.0097 |
| argmax + exact DINO reward | step2000 | 3 | 0.5350 +- 0.0009 | 0.4939 +- 0.0038 |
| argmax + exact DINO reward | step4000 | 3 | 0.5383 +- 0.0012 | 0.5131 +- 0.0113 |
| argmax + exact DINO reward | final | 3 | 0.5380 +- 0.0019 | 0.5048 +- 0.0038 |
| argmax + exact DINO reward | avg_last5 | 3 | 0.5379 +- 0.0009 | 0.5117 +- 0.0114 |

## Seed-paired contrasts (difference of per-seed means; t-test over seeds, n = common seeds)

| contrast | checkpoint | n | x0-hat: diff +- sd, p | 4-step sample: diff +- sd, p |
|---|---|---|---|---|
| argmax - random pick | step2000 | 3 | +0.0020 +- 0.0019, p=0.200 | +0.0071 +- 0.0064, p=0.194 |
| argmax - random pick | step4000 | 3 | -0.0023 +- 0.0009, p=0.043 | +0.0128 +- 0.0197, p=0.377 |
| argmax - random pick | final | 3 | -0.0001 +- 0.0023, p=0.924 | +0.0082 +- 0.0309, p=0.690 |
| argmax - random pick | avg_last5 | 3 | -0.0015 +- 0.0010, p=0.108 | +0.0152 +- 0.0183, p=0.287 |
| argmax + exact DINO reward - argmax | step2000 | 3 | +0.0063 +- 0.0014, p=0.016 | +0.0103 +- 0.0155, p=0.368 |
| argmax + exact DINO reward - argmax | step4000 | 3 | +0.0095 +- 0.0011, p=0.005 | +0.0181 +- 0.0265, p=0.357 |
| argmax + exact DINO reward - argmax | final | 3 | +0.0088 +- 0.0030, p=0.038 | +0.0130 +- 0.0213, p=0.403 |
| argmax + exact DINO reward - argmax | avg_last5 | 3 | +0.0080 +- 0.0011, p=0.007 | +0.0065 +- 0.0083, p=0.311 |
| argmax + exact DINO reward - random pick | step2000 | 3 | +0.0083 +- 0.0026, p=0.030 | +0.0174 +- 0.0133, p=0.150 |
| argmax + exact DINO reward - random pick | step4000 | 3 | +0.0072 +- 0.0007, p=0.003 | +0.0310 +- 0.0204, p=0.119 |
| argmax + exact DINO reward - random pick | final | 3 | +0.0087 +- 0.0009, p=0.003 | +0.0212 +- 0.0107, p=0.075 |
| argmax + exact DINO reward - random pick | avg_last5 | 3 | +0.0065 +- 0.0009, p=0.006 | +0.0217 +- 0.0187, p=0.182 |

## Caption-level paired contrast on the averaged model (seeds x captions pooled)

- argmax - random pick: x0-hat -0.0015 (paired t p=0.167, n=48); 4-step sample +0.0152 (p=0.012, n=48)
- argmax + exact DINO reward - argmax: x0-hat +0.0080 (paired t p=0.000, n=48); 4-step sample +0.0065 (p=0.222, n=48)
- argmax + exact DINO reward - random pick: x0-hat +0.0065 (paired t p=0.000, n=48); 4-step sample +0.0217 (p=0.001, n=48)
