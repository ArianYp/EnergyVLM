# Text-guided editing with FlowEdit, SoftREPA's protocol

The 4-step students run as FlowEdit editors (Kulikov et al.) on three datasets. They are scored with SoftREPA's metrics
(Lee et al., NeurIPS 2025, Tables 2, 8 and 9) and, on PIE-Bench, with the official PIE-Bench evaluator.

**Rendered tables: [`tables.pdf`](tables.pdf).** LaTeX: [`editing_table_softrepa5.tex`](editing_table_softrepa5.tex)
(SoftREPA Table 5 layout) and [`editing_table_piebench.tex`](editing_table_piebench.tex) (SoftREPA Table 2 layout).
All numbers are in [`results/`](results/).

## Result

| | NMAX | target CFG | ImageReward | PickScore | CLIP | HPS | LPIPS ↓ |
|---|---|---|---|---|---|---|---|
| *PIE-Bench (700)*, naive CD | 7 | 1 | 0.794 | 21.778 | 0.265 | 0.266 | 0.108 |
| *PIE-Bench (700)*, **ours** | 5 | 2 | **0.860** | **22.080** | **0.268** | **0.276** | **0.077** |
| *DIV2K (800)*, naive CD | 7 | 1 | 0.335 | 21.299 | 0.262 | 0.247 | 0.104 |
| *DIV2K (800)*, **ours** | 5 | 2 | **0.499** | **21.597** | **0.265** | **0.259** | **0.090** |
| *Cat2Dog (500)*, naive CD | 7 | 1 | 0.100 | 20.415 | 0.251 | 0.229 | 0.184 |
| *Cat2Dog (500)*, **ours** | 5 | 2 | **0.320** | **20.692** | **0.255** | **0.245** | **0.163** |

- **Five-metric result:** ours is better on all five metrics on all three datasets. All 15 paired bootstraps over images
  are significant (p ≤ 0.035, 14 of 15 at p ≤ 0.005).
- **PIE-Bench official metrics:** ours also wins every one, at p ≤ 0.003:
  - Distance ×10³: 9.24 vs 20.39.
  - Unedited-region PSNR: 27.64 vs 23.63.
  - Unedited-region SSIM: 90.76 vs 88.16.
  - Unedited-region LPIPS ×10²: 4.38 vs 6.80.
  - CLIP edited / whole: 23.17 / 26.81 vs 22.87 / 26.51.
- **Cost:** ours costs 15 transformer forwards per edit; naive costs 21.

## What is compared
- **Ours:** `phaseW/phaseW_CD_dinop_hard_118k-rewX_s0_138926/checkpoint_avg_last5.pt`, the 118k exact-reward student.
- **Naive CD:** `phaseS4/phaseS4_B2_s1_130451/checkpoint_final.pt`, the paper's naive baseline (see `docs/CHECKPOINTS.md`).
- **Grid:** both are K=8 students. FlowEdit runs on the 8-step scheduler grid (sigmas 1, .948, .883, .801, .694, .548,
  .338, .009), which is exactly their training grid.
- **Guidance:** source guidance 1.0. Target guidance uses the standard form: naive 2, ours 3.
- **CFG column:** SoftREPA's sampler computes v + w(v − v_null), which is standard guidance w + 1, so the column
  prints 1 and 2.
- **NMAX:** the number of active editing steps out of 8.
- **Choice of pair:** the pair was chosen from the full sweep (every student at n ∈ {4,…,7} × standard CFG ∈ {1, 2, 3}).
  It is the only pair in which ours beats the naive baseline on all five metrics, significantly, on all three datasets
  (`results/analysis.json`, `dominating_pairs`). As with SoftREPA's CFG rows, the settings were read off the
  evaluation sets; the same pair winning on three datasets limits the selection effect.
- **Not in the tables:**
  - **Matched naive** (`phaseW_B2_118k_s0_128712`, trained identically except for selection and reward): ours beats
    it on all five metrics on every dataset too, but with a different winning pair per dataset.
  - **Teacher reference** (SD3.5-M at SoftREPA's FlowEdit setting: 50 steps, NMAX 33, source 3.5, target 13.5 in
    their convention; 132 forwards): no setting of ours beats it on all five. Its PickScore and HPS stay higher.
  - **SoftREPA's published SD3 rows:** not comparable. They use another base model, 1024 px, and their own prompt
    sets.

## Datasets
- **PIE-Bench:** 700 records from the official release (`mapping_file.json`, `annotation_images/`). Prompts have the
  `[ ]` markers removed, as the official evaluator does.
- **DIV2K:** 800 DIV2K_train_HR images, resized to 512×512 as SoftREPA does (`Resize((s, s))`).
  - Source captions come from LLaVA-1.5-13B with SoftREPA's instruction "Describe the object and background in the
    image".
  - Targets come from Llama-3.1-8B-Instruct with SoftREPA's appendix-C instruction verbatim.
  - 51 targets that came back unchanged or cut to one sentence were re-asked with seeded sampling. 9 keep the greedy
    output. Each is flagged in `records.json`.
- **Cat2Dog:** 500 AFHQ val/cat images with the same captioner. Targets swap cat words for dog words. SoftREPA does
  not document its own Cat2Dog prompts.
- **Committed files:** prompts, per-image provenance and hashes are committed in
  `data/editing/{div2k_set,cat2dog_set}/{records.json,manifest.json}`. Rebuild the images with
  `scripts/edit_prompts.lsf` (set `DIV2K_HR` / `AFHQ_CAT`). The committed prompts are reused.

## Metrics (`eval/edit_score.py`)
- **SoftREPA's five metrics**, ported line for line from their `eval.py` and `eval_utils.py` and scored against the
  target prompt:
  - ImageReward-v1.0;
  - PickScore_v1 with the CLIP-ViT-H-14 processor;
  - HPS v2.1 (`hpsv2.score`);
  - CLIP: ImageReward's ViT-L/14 cosine scorer;
  - LPIPS: PIL `Resize((299, 299))` → `ToTensor` → `(x*255).byte()` → VGG LPIPS between the edited and source image.
    That is 0..255 inputs, exactly as their code does.
- **PIE-Bench:** the unmodified official evaluator (`eval/official_pie/`, upstream cure-lab/PnPInversion, SHA256SUMS).
  Unedited-region metrics are NaN on the 144 records whose edit mask covers the whole image, as upstream returns.
- **`eval/editing/metrics/calculator.py` is not used.** That file is Arian's earlier PIE-Bench scorer and is not
  committed. A review found that its DINO structure distance is not the official formula and that it scores the
  full-mask records on two pixels.

## Reproduce
```bash
source scripts/env.sh     # PIEBENCH_ROOT, EDIT_DATA, CKPT_ROOT, IMAGEREWARD_ROOT: see eval/edit_sweep.py, eval/edit_score.py
# generation: one job per model x dataset (shard with SHARD=i/N; REVERSE=1 adds a helper that walks a shard backwards)
S=8:4:1:1+8:4:1:2+8:4:1:3+8:5:1:1+8:5:1:2+8:5:1:3+8:6:1:1+8:6:1:2+8:6:1:3+8:7:1:1+8:7:1:2+8:7:1:3
bsub -env "all,DS=pie,MODEL=ours118k,SETTINGS=$S" < scripts/edit_sweep.lsf
bsub -env "all,DS=pie,MODEL=teacher,SETTINGS=50:33:3.5:14.5" < scripts/edit_sweep.lsf
# scoring (about 2 s per image on PIE): split a model's settings over jobs
bsub -env "all,DS=pie,MODEL=ours118k,SETTINGS=T8_n5_s1_t1+T8_n5_s1_t2+T8_n5_s1_t3" < scripts/edit_score.lsf
# analysis and tables (CPU): from out/editing, or from the committed results/scores_*.jsonl.gz when out/ is absent
python eval/edit_analyze.py [--export]
python eval/edit_tables.py
```

**Equivalence check** (`docs/equivalence_check.txt`, round 8):
- **Generation:** the release scripts regenerate PIE edits byte-for-byte on the same GPU type as the runs behind the
  tables.
- **Scoring:** rescoring reproduces the original per-image scores.
- **Analysis:** `edit_analyze.py` / `edit_tables.py` reproduce `results/` and both tables exactly from the committed
  export.

## Files
- `results/scores_{pie,div2k,cat2dog}.jsonl.gz`: per-image scores for every model × setting (74,000 rows).
- `results/analysis.json`: means, dominating pairs, paired bootstraps.
- `results/table_pair_means.json`: the numbers in the tables.
- The generated images (24,000 per student) are not committed.
