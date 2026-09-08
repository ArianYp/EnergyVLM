# Held-out DINO monitor (validation captions, offline scorer)

Teacher argmax candidate, same captions: DINO 0.6039

| arm | checkpoint | seeds | x0-hat DINO (mean +- sd over seeds) | 4-step sample DINO (mean +- sd) |
|---|---|---|---|---|
| random pick | avg_last5 | 3 | 0.5970 +- 0.0005 | 0.5831 +- 0.0029 |
| argmax | avg_last5 | 3 | 0.5973 +- 0.0006 | 0.5888 +- 0.0060 |
| argmax + exact DINO reward | avg_last5 | 3 | 0.6094 +- 0.0002 | 0.6080 +- 0.0012 |

## Seed-paired contrasts (difference of per-seed means; t-test over seeds)

| contrast | checkpoint | n | x0-hat: diff +- sd, p | 4-step sample: diff +- sd, p |
|---|---|---|---|---|
| argmax - random pick | avg_last5 | 3 | +0.0003 +- 0.0011, p=0.723 | +0.0056 +- 0.0051, p=0.193 |
| argmax + exact DINO reward - argmax | avg_last5 | 3 | +0.0121 +- 0.0005, p=0.001 | +0.0192 +- 0.0065, p=0.036 |
| argmax + exact DINO reward - random pick | avg_last5 | 3 | +0.0124 +- 0.0006, p=0.001 | +0.0248 +- 0.0023, p=0.003 |

## Caption-level paired contrast on the averaged model (seeds x captions pooled)

- argmax - random pick: x0-hat +0.0003 (paired t p=0.335, n=900); 4-step sample +0.0056 (p=0.000, n=900)
- argmax + exact DINO reward - argmax: x0-hat +0.0121 (paired t p=0.000, n=900); 4-step sample +0.0192 (p=0.000, n=900)
- argmax + exact DINO reward - random pick: x0-hat +0.0124 (paired t p=0.000, n=900); 4-step sample +0.0248 (p=0.000, n=900)
