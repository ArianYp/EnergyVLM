# GenEval2 breakdown (Soft-TIFA gmean, Qwen3-VL judge), x100, averaged models

| measure | naive CD (5 seeds) | argmax + exact reward, factorial (5 seeds) | argmax + exact reward, first round (3 seeds) | factorial - naive (seed-paired) |
|---|---|---|---|---|
| overall (official, gmean of atoms) | 22.5 ± 0.7 | 24.4 ± 0.8 | 25.2 ± 0.3 | +1.9 ± 0.7 (p 0.003, n 5) |
| atom mean (arithmetic) | 61.9 ± 0.8 | 64.0 ± 0.6 | 64.7 ± 0.3 | +2.2 ± 1.1 (p 0.010, n 5) |
| collapsed prompts % (gmean < 0.05) | 36.5 ± 3.6 | 31.4 ± 1.5 | 31.8 ± 0.6 | -5.2 ± 4.2 (p 0.053, n 5) |
| skill: object | 83.7 ± 1.5 | 85.9 ± 0.7 | 85.9 ± 0.7 | +2.3 ± 2.0 (p 0.066, n 5) |
| skill: count | 35.9 ± 1.1 | 37.5 ± 1.2 | 38.6 ± 0.1 | +1.6 ± 1.9 (p 0.125, n 5) |
| skill: attribute | 70.8 ± 1.2 | 73.5 ± 1.0 | 73.6 ± 1.1 | +2.7 ± 1.9 (p 0.034, n 5) |
| skill: position | 48.3 ± 2.9 | 51.8 ± 0.8 | 51.7 ± 0.4 | +3.5 ± 3.0 (p 0.062, n 5) |
| skill: verb | 23.5 ± 4.6 | 26.5 ± 2.2 | 29.8 ± 1.4 | +3.0 ± 4.8 (p 0.231, n 5) |
| prompts with atoms = 3 | 44.2 ± 1.8 | 46.2 ± 1.5 | 50.6 ± 1.9 | +2.0 ± 1.7 (p 0.064, n 5) |
| prompts with atoms = 4 | 34.7 ± 2.7 | 37.0 ± 2.9 | 37.6 ± 2.3 | +2.3 ± 4.9 (p 0.365, n 5) |
| prompts with atoms = 5 | 24.9 ± 0.4 | 27.2 ± 3.2 | 29.7 ± 3.2 | +2.3 ± 3.5 (p 0.212, n 5) |
| prompts with atoms = 6 | 22.0 ± 1.1 | 22.6 ± 1.0 | 24.1 ± 1.7 | +0.6 ± 1.6 (p 0.435, n 5) |
| prompts with atoms = 7 | 17.8 ± 1.7 | 18.5 ± 2.2 | 17.4 ± 1.9 | +0.7 ± 3.8 (p 0.717, n 5) |
| prompts with atoms = 8 | 13.6 ± 1.1 | 16.1 ± 1.4 | 16.3 ± 2.8 | +2.6 ± 2.0 (p 0.046, n 5) |
| prompts with atoms = 9 | 13.1 ± 1.0 | 16.8 ± 1.0 | 15.3 ± 1.0 | +3.8 ± 1.4 (p 0.004, n 5) |
| prompts with atoms = 10 | 9.5 ± 1.3 | 10.8 ± 1.1 | 11.0 ± 0.7 | +1.4 ± 2.2 (p 0.239, n 5) |
