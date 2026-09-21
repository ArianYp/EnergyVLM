# CTCal (Guo et al., CVPR 2026) vs our T2I evaluation/training settings — alignment audit

Date: 2026-09-18.
Reference: *CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration*,
Xiefan Guo, Xinzhu Ma, Haiyu Zhang, Di Huang. CVPR 2026 (proceedings pp. 43558–43567),
arXiv:2603.20741. Rendered pages read: `scratchpad/ctcal/page-01..10.png`.

## 0. Source availability (important caveat)

`git clone https://github.com/xiefan-guo/ctcal third_party/ctcal` **succeeded**, but the repository
contains **no code**: HEAD `3c605dc` ("Update README.md", Xiefan Guo, 2026-03-24) and the working
tree is exactly

```
third_party/ctcal/README.md      (2 lines: title + arXiv link)
third_party/ctcal/LICENSE
third_party/ctcal/.gitignore
```

`third_party/ctcal/README.md:1-2` is the entire content — no configs, no scripts, no branches
(`origin/main` only). So **no file/line citations from their repo are possible**; every CTCal
setting below is cited to the paper by page/section.

The paper repeatedly defers hyperparameters to a **supplementary** ("More parameter setting,
training and evaluation details are provided in the supplementary material", p. 6 §4.1; also for
λ1..λ4, τ, the autoencoder architecture, the attention-map workflow, and the SD3 t_tea rule).
That supplementary is **not public**: the arXiv listing (`arxiv.org/abs/2603.20741`, comments
"Accepted by CVPR 2026") offers only PDF/HTML/TeX of the 10-page main paper, and the arXiv HTML has
no appendix. Everything marked **NOT DISCLOSED** below is genuinely unavailable, not unread.

The one thing that *is* fully recoverable is the evaluation and data-construction protocol, because
CTCal states it **adopts reference [17] = T2I-CompBench++** verbatim ("we adopt the dataset
construction method proposed by [17]", p. 6 §4.1) and **copies [17]'s baseline rows into its
Table 1**. I verified this numerically — see §1.1. So T2I-CompBench++ (which we vendor at
`/lustre/.../EnergyVLM/T2I-CompBench`) is the authoritative source for their protocol, and *that*
repo I can cite by file:line.

---

## 1. What CTCal does

### 1.1 Table 1 is T2I-CompBench++ TABLE XIII (verified)

Our vendored copy of the ++ paper (`T2I-CompBench/paper/T2I-CompBench++.pdf`, TABLE XIII,
"Benchmarking on all categories with proposed metrics") has these column headers:

```
          Color  Shape  Texture  2D-Spatial  3D-Spatial  Numeracy | Non-Spatial            | Complex
          B-VQA  B-VQA  B-VQA    UniDet      UniDet      UniDet   | CLIP GPT-4V Share-CoT  | 3-in-1 GPT-4V Share-CoT
Stable v1-4  0.3765 0.3576 0.4156 0.1246 0.3030 0.4456 | 0.3079 0.7717 0.7487 | 0.3080 0.6453 0.7727
Stable v2    0.5065 0.4221 0.4922 0.1342 0.3230 0.4582 | 0.3127 0.8153 0.7567 | 0.3386 0.6483 0.7783
Stable v3    0.8132 0.5885 0.7334 0.3200 0.4084 0.6174 | 0.3140 0.9093 0.7782 | 0.3771 0.8717 0.7919
```

CTCal Table 1 (p. 6) reports for SD 1.4: `0.3765 0.3576 0.4156 0.1246 0.3030 0.4456 0.7487 0.3080`,
for SD 2.1: `0.5065 0.4221 0.4922 0.1342 0.3230 0.4582 0.7567 0.3386`, and for SD 3 (2B):
`0.8132 0.5885 0.7334 0.3200 0.4084 0.6174 0.7782 0.3771`. Every value matches, and the Non-Spatial
values match the **Share-CoT** column (not the CLIP column). Same for SD XL, PixArt-α-ft, DALL·E 3
and FLUX-schnell. Conclusion, with certainty:

| CompBench++ category | CTCal's evaluator | our evaluator |
|---|---|---|
| color / shape / texture | **BLIP-VQA (B-VQA)** | BLIP-VQA ✅ |
| 2D-spatial / 3D-spatial / numeracy | **UniDet** | UniDet ✅ |
| **non-spatial** | **Share-CoT** = ShareGPT4V-7B with chain-of-thought, `MLLM_eval/ShareGPT4V-CoT_eval/Share_eval.py --cot` | **CLIPScore (ViT-B/32)** ❌ |
| **complex** | **3-in-1** | 3-in-1 ✅ |

Note also that CTCal's row *labelled* "SD 2.1" is ++'s **"Stable v2"** row, and ++'s GORS code
instantiates `stabilityai/stable-diffusion-2-base`
(`T2I-CompBench/GORS_finetune/inference_eval.py:67`, `train_text_to_image.py:89`), i.e. the **512px
SD2-base**, not SD2.1-768. CTCal cites [32] (Rombach et al.) for it and never gives an HF id.

Our vendored `T2I-CompBench` is at upstream HEAD `1b70949` authored by the upstream owner
(Karine-Huang), working tree clean apart from `__pycache__`, and **does contain**
`MLLM_eval/ShareGPT4V-CoT_eval/Share_eval.py` and `MLLM_eval/gpt4v_eval.py`. The ++ Readme
(`T2I-CompBench/Readme.md:169-233`) documents ShareGPT4V-CoT explicitly as "evaluation metrics as in
TABLE XIII". So the missing piece on our side is an *unimplemented* option in code we already have,
not a missing dependency on the benchmark side (weights and env are still needed — see §3).

### 1.2 Base models, inference, benchmarks

| item | CTCal | evidence |
|---|---|---|
| base models | Stable Diffusion 2.1 [32] and Stable Diffusion 3 (2B) [9]; "model-agnostic", implemented in Diffusers | p. 6 §4.1 |
| HF ids | **NOT DISCLOSED** (inherited protocol implies `stabilityai/stable-diffusion-2-base` for the SD2 rows and SD3-medium/2B for the SD3 rows) | p. 6; `GORS_finetune/inference_eval.py:67` |
| eval resolution | **NOT DISCLOSED** in the paper. Inherited: GORS/++ generates with the stock `StableDiffusionPipeline` defaults on SD2-base ⇒ **512×512**; the SD3 (2B) rows come from ++'s own benchmarking of SD3 at its native default ⇒ **1024×1024** | `GORS_finetune/inference_eval.py:67-71,105` |
| sampler / steps / guidance | **NOT DISCLOSED**. Inherited: **DDIMScheduler, 30 steps** in the eval generator, guidance left at the diffusers default (7.5) | `GORS_finetune/inference_eval.py:70,105`; (`inference.py:46` uses 50 steps for single-image demos) |
| seeds | fixed seed shared across all compared models; the eval generator uses `manual_seed(42)` then `42+n+1` per image | `GORS_finetune/inference_eval.py:102,106`; ++ p.: "the images are generated using the fixed seed across all models" |
| images / prompt (CompBench++) | **10** | `GORS_finetune/inference_eval.py:23` (`--n_iter` default 10); `T2I-CompBench/Readme.md:310` ("10 images are generated per prompt for metric calculation, and we use the fixed seed across all methods"); ++ §VI.B |
| eval prompt split | the **300-prompt val split per category** (`*_val.txt`); the 700-prompt `*_train.txt` per category is the *training* source | ++ p. (1,000 prompts per category: 700 train / 300 test); `examples/dataset/*_val.txt` |
| GenEval | Table 2, SD3 (2B) only, all-category **joint** LoRA (not per-category); protocol otherwise GenEval's own | p. 7 §4.3 |
| images / prompt (GenEval) | **NOT DISCLOSED** (GenEval's own default is 4) | p. 7 |
| official scripts? | yes — they report the ++ metrics and copy ++'s baseline numbers, so by construction the official BLIP-VQA / UniDet / 3-in-1 / ShareGPT4V-CoT scripts | §1.1 above |

Evaluator implementation details they inherit (for reproduction): BLIP w/ ViT-B + CapFilt-L VQA,
`--np_num` default **8** (`BLIPvqa_eval/BLIP_vqa.py:65-70`); UniDet trained on COCO/Objects365/
OpenImages/Mapillary; CLIPScore = **ViT-B/32** (`CLIPScore_eval/CLIP_similarity.py:16`);
ShareGPT4V temperature **0.2**, GPT-4V temperature 1.

### 1.3 Training-data construction (GORS-style)

CTCal, p. 6 §4.1 "Datasets", verbatim content:

* prompt source: **the T2I-CompBench++ text prompts, 700 per category** (the `*_train.txt` split).
* for each prompt, generate **k** images with the *target* model (SD 2.1 or SD 3) ⇒ candidate
  text–image pairs.
* score each candidate with **the scoring metric introduced in [17]** (i.e. the ++ metric of that
  category — B-VQA / UniDet / CLIP / 3-in-1 — this is the *biased* GORS variant, not GORS-unbiased).
* keep the **top-n** highest-scoring pairs per category.
* **k = 100, n = 10,000 for SD 2.1**; **k = 30, n = 10,000 for SD 3**; per category.
* Table 1 fine-tunes a **per-category** LoRA (as GORS does); Table 2 (GenEval) instead aggregates
  all categories → **80,000 pairs** for one joint fine-tune (p. 7 §4.3).
* training-image resolution: **NOT DISCLOSED**; inherited GORS dataloader resizes/center-crops to
  **512** (`GORS_finetune/train_dataset.py:24,29,83-90`).

Deviation to note: GORS as published selects by **threshold**, not top-n
(`train_dataset.py:59-68`); CTCal changed this to a fixed top-n = 10,000.

### 1.4 CTCal method + training hyperparameters

| item | CTCal | evidence |
|---|---|---|
| framework | Diffusers + **LoRA** | p. 6 §4.1 |
| LoRA targets | **self-attention layers of the text encoder** + **attention layers of the denoising network** | p. 6 §4.1 (identical wording to ++ §VI.A; GORS code: `inject_trainable_lora(unet, target_replace_module=["CrossAttention","Attention"])`, `train_text_to_image.py:517-519`, and `["CLIPAttention"]` for the text encoder, `:532-535`) |
| LoRA rank / alpha | **NOT DISCLOSED** (GORS default `--lora_rank 4`, `train_text_to_image.py:244-248`) | — |
| learning rate | **NOT DISCLOSED** (GORS: unet 1e-4, text encoder 5e-6, `:250-260`) | — |
| batch size | **NOT DISCLOSED** (GORS: per-device 2, `:220-224`; ++ paper says effective batch 5) | — |
| steps / epochs | **NOT DISCLOSED** (GORS: `--max_train_steps 15000`, `--num_train_epochs 100`, `:225-230`) | — |
| optimizer | **NOT DISCLOSED** (GORS/++: AdamW, β=(0.9,0.999), ε=1e-8, wd 0.01, `:325-329`) | — |
| LR schedule / warm-up | **NOT DISCLOSED** (GORS: `constant`, 500 warm-up steps, `:268-278`) | — |
| mixed precision | **NOT DISCLOSED** (GORS launcher: `--mixed_precision=fp16`, `GORS_finetune/train.sh:7`) | — |
| GPUs / time | **NOT DISCLOSED** (++/GORS: 8× 32GB V100) | — |
| EMA | not mentioned anywhere | — |
| gradient clipping | **NOT DISCLOSED** (GORS: `--max_grad_norm 1.0`, `:329`) | — |
| data augmentation | **NOT DISCLOSED** (GORS dataset: center-crop True, random h-flip True, colour jitter off, `train_dataset.py:81-98` + `train_text_to_image.py:201-213`) | — |
| **t_stu sampling** | "we strictly adhere to the inherent timestep sampling protocol established by the text-to-image diffusion model during training" — i.e. **uniform for SD2.1**, **logit-normal for SD3** | p. 4 §3.3 |
| **t_tea** | **t_tea = 0** for classical diffusion models (SD 2.1); for SD3 a "reevaluation of timestep priority based on the sampling distribution" is needed because naive t_tea = 0 "may degrade performance" — the actual SD3 rule is **NOT DISCLOSED** (supplementary) | p. 4 §3.3, p. 6 |
| t_tea ablation | randomly sampling 0 ≤ t_tea < t_stu still helps but is worse than t_tea = 0 (Color 0.7028 vs 0.7233; 2D-spatial 0.2029 vs 0.2142) | Table 5, p. 8 |
| gradient flow | optimisation restricted to the network at t_stu; **gradient through A_tea is truncated**; A_stu and A_tea come from the *same* (currently fine-tuned) model, not a separate frozen extractor | p. 3 §3.1 |
| loss | `L = L_diffusion + λ_t · L_CTCAL`, with `L_CTCAL = (1/N_noun) Σ [ λ1 D(A_stu,A_tea) + λ2 D(f_enc(A_stu), f_enc(A_tea)) + λ3 D(f_dec(f_enc(A_tea)), A_tea) ] + λ4 R_subject` | Eq. (7), p. 4 |
| λ1..λ4, τ | **NOT DISCLOSED** (supplementary) | p. 4 |
| adaptive weight | **λ_t = t_stu / T_train** (linear in the current timestep) | Eq. (8), p. 6 |
| D(·) | mean squared error | p. 6 §4.1 |
| noun selection | **Stanza** part-of-speech tagging; only **noun** tokens' attention maps are used (adjective tokens help further, Table 6, but are not in the main method) | p. 6 §4.1; p. 4 §3.2; p. 8 |
| attention-map source | for SD2.1: the cross-attention map A^{y_i}; for SD3/MM-DiT: the image→text block A^{IT} of the joint self-attention. **Which layers/blocks and at which spatial resolution the maps are aggregated is NOT DISCLOSED** ("More details on the workflow for processing cross-attention maps are provided in the supplementary material", p. 3) | p. 2 §2, p. 3 §3.1 |
| semantic autoencoder | "a lightweight autoencoder, composed of an encoder f_enc and a decoder f_dec" over attention maps, with a reconstruction proxy task (λ3 term) to stop f_enc collapsing to a constant encoding. **Architecture NOT DISCLOSED** (supplementary) | p. 4 §3.2 |
| R_subject | `mean_i ReLU( max_i max(A_stu,y_i) − max(A_stu,y_i) − τ )` — pull low-response subjects up to the highest-response subject, with dead-band τ | Eq. (6), p. 4 |
| other reported extras | user study (12 participants, 10 questions each); diversity via Mean LPIPS; quality via Aesthetic score | Tables 3, 7 |

---

## 2. Side-by-side: theirs vs ours

Our citations: `exp0/phaseA_generate.py` (PA), `exp0/generate_candidates.py` (GC),
`phaseC/compbench_eval.py` (CE), `phaseC/geneval2_eval.py` (GE),
`ablations/phaseN_eval_alignment.lsf` (E1), `ablations/phaseN_eval_official10.lsf` (E10),
`phaseFP/eval_pool_101203/manifest.json` (POOL), `phaseW/_equiv/clean-tree/README.md` (RM),
`phaseW/_equiv/clean-tree/train/distill.py` (DS), `.../data/build_candidates.py` (BC).

### 2.1 Model and inference

| setting | CTCal | ours | match? |
|---|---|---|---|
| base model | SD 2.1 (=SD2-base) and SD 3 **2B** | **SD3.5-Medium** (`stabilityai/stable-diffusion-3.5-medium`), PA:40, DS:226, BC:44 | ✗ different backbone |
| what is trained | LoRA on text-encoder self-attn + denoiser attn | **full-parameter** AdamW on the whole transformer, no LoRA, text encoders frozen (DS:345-346,361) | ✗ |
| teacher/source of supervision | real images (GORS-selected generations) + diffusion loss | **teacher trajectories**: frozen SD3.5-M, K=8 Euler steps, CFG 7, 512px (BC:45-48, RM "Method") | ✗ different paradigm (consistency distillation) |
| eval sampler | DDIM, 30 steps (inherited) | **Euler (FlowMatch), 4 steps** for students, 28 for the base teacher (PA:39,150; GC:69-112; E1:42) | ✗ |
| eval guidance | ~7.5 (diffusers default, inherited) | **cfg 1.0** for guidance-distilled students (PA:38; RM "Evaluate students with CFG=1.0"), **7.0** for the base teacher; note E1:43 defaults `CFG=7.0` because Phase-N students were trained with `--student_cfg 7` | ✗ / context-dependent |
| resolution | 512 for the SD2 rows, **1024** for the SD3 rows | **512×512** everywhere (PA:52, GC:144, DS:299, E1:48) | ✗ for SD3 comparison |
| seeding | fixed seed 42 shared across models | `manual_seed(seed + idx + j·10^6)` — paired across models/steps (PA:146); candidates `idx*1000+j` (GC:209) | ✓ in spirit (paired, shared across arms) |
| dtype | fp16 (inherited) | bf16 throughout (PA:101-106) | ≈ |

### 2.2 Benchmark protocol

| setting | CTCal | ours | match? |
|---|---|---|---|
| benchmark | T2I-CompBench++ + GenEval | T2I-CompBench++ + **GenEval2** (Soft-TIFA gmean, Qwen3-VL judge; GE:1-33) | partial |
| split | **val**, 300/category | **val**, 300/category (`split: "val"`, POOL) | ✓ |
| n prompts | 2,400 (8×300) | **2,398** — 300 in 7 categories, **298** in spatial (2 val prompts dropped because they also appear in `spatial_train.txt`; GC:34-55 `exclude`) | ≈ (−2) |
| images / prompt | **10** | **1** by default (E1 stages `cand0.png` only, CE:404); **10** only in `phaseN_eval_official10.lsf` (E10:35,49) | ✗ unless E10 is used |
| color/shape/texture | B-VQA, np_num 8 | B-VQA, np_num 8 (CE:75-79, `run --np_num` default 8, CE:415) | ✓ |
| 2D-spatial / 3D-spatial / numeracy | UniDet | UniDet, official scripts (CE:81-101) | ✓ |
| **non-spatial** | **Share-CoT (ShareGPT4V-7B + CoT)**, values ≈0.75–0.79 | **CLIPScore ViT-B/32** (CE:66,111-115), values ≈0.31 | ✗ **different metric and different scale** |
| complex | 3-in-1 | 3-in-1, routing reproduced from `3_in_1_eval/3_in_1.py` and proven identical over all 300 complex_val prompts by `phaseC/test_3in1_identity.py` (CE:249-285) | ✓ |
| evaluator code | official, unmodified | official, unmodified, run from their own cwd; only a `ruamel_yaml` compat shim on PYTHONPATH (CE:230-240); repo pinned at upstream `1b70949`, clean tree | ✓ |
| aggregate reported | per-category only | per-category **plus** an unweighted mean over all 8 categories, `compbench_mean` (E1:136, E10:102) | ✗ non-standard (see §4) |

### 2.3 Training-data construction

| setting | CTCal (GORS-style) | ours |
|---|---|---|
| prompt source | CompBench++ `*_train.txt`, 700/category (5,600 prompts) | **COCO captions** (3,000 or 113,948), RM "build_pool.py" |
| candidates per prompt | **k = 100** (SD2.1) / **k = 30** (SD3) | **N = 4** (BC:45) |
| generator of candidates | the target model itself | frozen SD3.5-M teacher, 8 Euler steps, cfg 7 (BC:45-48) |
| scorer | ++ category metric (B-VQA / UniDet / CLIP / 3-in-1) | **DINOv2 mean-patch cosine** to the caption's real COCO photograph (RM "Method") |
| selection | **top-n = 10,000 per category** (GORS itself uses a threshold) | **argmax over the 4 candidates**, per caption |
| reference image needed? | no | **yes** — a real photo per caption; this is why captions must come from an image–caption corpus |
| training-image resolution | 512 (inherited) | 512 latents, never materialised as images (trajectories only) |
| total pairs | 10,000/category; 80,000 for the joint GenEval model | 3,000 or 113,948 captions × 1 selected trajectory |

### 2.4 Training hyperparameters

| setting | CTCal | ours (paper's converged schedule, RM §"Converged schedule") |
|---|---|---|
| objective | diffusion loss + λ_t·L_CTCAL (attention calibration) | Huber consistency-distillation loss on teacher clean-latent estimates, + projector reward λ=80 for "ours" |
| optimizer | AdamW (inherited) | AdamW (DS:361) ✓ |
| lr | not disclosed (GORS 1e-4 / 5e-6) | **1e-5** (DS:232) |
| schedule | constant + 500 warm-up (inherited) | **cosine to 0**, warm-up 150 (3k) / 700 (118k) (DS:229-231, RM step 10) |
| batch | 5 (++) | **16** = 4 GPUs × accum 4 (RM step 10) |
| updates | 15,000 (inherited default) | **3,000** (3k pool, 16 passes) / 14,244 (118k, 2 passes) |
| grad clip | 1.0 (inherited) | 1.0 (DS:234) ✓ — but note it fires on ~100% of updates under the constant-LR schedule (RM) |
| EMA | none | none; **uniform average of the last 5 checkpoints** instead (RM step 6) |
| precision | fp16 | bf16 |
| GPUs | 8× V100-32GB (++) | 1 GPU (3k) / 4 GPUs (118k) |

---

## 3. Ranked misalignments (why our numbers are not comparable to their Table 1)

Ranked by how much each one invalidates a direct comparison.

**1. Non-spatial evaluator: we use CLIPScore, their Table 1 column is Share-CoT.**
Different metric, different scale (ours ≈0.31, theirs ≈0.75–0.79), different ranking behaviour (++
TABLE XIII shows CLIP is nearly constant across models — 0.2980–0.3197 over 12 systems — i.e. it
cannot separate models at all, which is exactly why ++ replaced it). Any non-spatial number of ours
placed beside their column is meaningless, and it also contaminates our `compbench_mean`.
*Fix*: implement a `sharegpt4v_cot` evaluator in `phaseC/compbench_eval.py`'s `EVALUATORS` map
calling `T2I-CompBench/MLLM_eval/ShareGPT4V-CoT_eval/Share_eval.py --category non_spatial --cot`
(the script is already vendored). *Cost*: ShareGPT4V-7B weights (~14 GB) + `ShareGPT4V-7B_Pretrained
_vit-large336-l12`, a separate venv (InternLM-XComposer/LLaVA stack, conflicts with our
transformers pin — same pattern as the existing `cache/venv_geneval2`), then ~3,000 CoT generations
per model (300 prompts × 10 images) ≈ 1–2 GPU-hours per model. Half a day of setup, then cheap.
Sampling temperature 0.2 makes it non-deterministic — fix a seed and report it.

**2. Resolution 512 vs 1024 for the SD3 comparison.**
The SD3 (2B) row in their Table 1 comes from ++'s benchmarking at SD3's native 1024. Our whole
pipeline generates at 512 (PA:52, DS:299). CompBench's evaluators (BLIP-VQA at 480px, UniDet at
detectron2 scales) are resolution-sensitive: small objects in numeracy/spatial are systematically
under-detected at 512. Our absolute numbers therefore sit below a 1024 baseline for reasons that
have nothing to do with method.
*Fix*: for a comparison table only, regenerate the base/teacher reference at 1024 (`--height 1024`
is already plumbed through PA:52 and E1:48). *Cost for the teacher reference*: 2,398 prompts × 10
images at 1024, ~4× the 512 cost, i.e. a few hundred GPU-hours on 4 GPUs — feasible.
*Cost for our students*: **not feasible cheaply** — the students are distilled at 512 and the
candidate cache, projector and reference embeddings are all 512; a 1024 student means rebuilding
the cache and retraining. Recommended stance: keep 512 and say so explicitly, comparing only
within our own 512 family, and never print our numbers in the same table as their Table 1.

**3. Images per prompt: 1 (default) vs 10 (official).**
`ablations/phaseN_eval_alignment.lsf` scores one image per prompt (E1:94-96 stages the default
`cand0.png`). The official protocol is 10 with a per-prompt mean (`Readme.md:310`,
`inference_eval.py:23`). One image per prompt inflates per-prompt variance ~3.2× and is not the
benchmark.
*Fix*: already implemented — use `ablations/phaseN_eval_official10.lsf` (E10:35 `NIMG=10`, E10:49
builds `cand0..cand9`), which is a strict superset (candidate 0 is the same image/seed).
*Cost*: 10× generation + 10× scoring, ~16 h wall-clock on 4 GPUs per model as budgeted in E10.
This is the cheapest high-value fix.

**4. Base model and step count: SD3.5-Medium 4-step guidance-free student vs SD3-2B / SD2.1 at ~30
steps with CFG ≈7.5.**
Both the backbone (2.5B SD3.5-M vs 2B SD3) and the inference budget differ. A 4-step distilled
student is expected to lose several points of B-VQA versus a 30-step guided teacher; that gap is
the cost of distillation, not a property of our method.
*Fix*: none that preserves the research question — instead always report our own SD3.5-M teacher at
28 steps / cfg 7 in the same table as the anchor, so the reader sees our teacher's absolute level
next to their SD3 row. *Cost*: zero, we already generate it (`CKPT=base, CFG=7.0, STEPS=28`).

**5. Guidance: cfg 1.0 (ours, students) vs ~7.5 (theirs).**
Correct for our guidance-distilled students (sampling them with guidance roughly halves their
scores, RM) but it is a different operating point from every literature row. Also a live foot-gun:
`phaseN_eval_alignment.lsf:43` defaults `CFG=7.0` while `exp0/phaseA_generate.py:38` defaults
`--cfg 1.0`, and only the header comment (E1:19-25) records which student needs which.
*Fix*: make `EVAL_CFG` mandatory in E1 as it already is in E10 (`E10:32`). *Cost*: one line.

**6. Prompt-split provenance: 2,398 vs 2,400 prompts, and spatial at 298.**
We drop 2 spatial val prompts that also appear in `spatial_train.txt` (GC:50-51 + `exclude`). That
is methodologically *better* for us (our training set is COCO anyway, so it costs nothing) but it
means our spatial denominator differs from theirs by 2/300 = 0.67%.
*Fix*: report n per category alongside every score (the `scores.json` already carries it, CE:365).
*Cost*: zero.

**7. Sampler family: DDIM ε-prediction (SD2) vs FlowMatch-Euler (ours).**
Unavoidable consequence of the backbone choice; mention it once and move on. *Cost*: zero.

**8. GenEval vs GenEval2.**
Their Table 2 is GenEval (Ghosh et al. 2023, 6 skills, object-detector based). We run **GenEval2**
with a Qwen3-VL Soft-TIFA judge (GE:1-33). Different benchmark, different numeric range; our 20–23
figures are not comparable to their 0.50–0.69.
*Fix*: if a GenEval row is wanted, run the original GenEval — *cost*: a third evaluator env
(mmdet + the GenEval object detector), ~1 day of setup, then cheap. Otherwise label the column
"GenEval2" everywhere and never abbreviate it to "GenEval".

---

## 4. Things in our pipeline that look unusual relative to standard practice

* **`compbench_mean`, the unweighted mean over 8 categories, is not a literature metric** (E1:136,
  E10:102). Nobody in the T2I-CompBench line reports a single scalar; and ours averages a ~0.31-scale
  CLIPScore column with ~0.5–0.8-scale B-VQA columns, so a fixed shift in non-spatial moves the
  headline by 1/8 of itself. It is fine as an internal, pre-registered decision statistic — it must
  not appear in a paper table beside per-category literature numbers. If a summary is needed, use
  the per-category table plus paired per-prompt tests (which `eval/compare_arms.py` already does).
* **Training prompts are COCO captions, not CompBench train prompts.** This is *stricter* than
  CTCal/GORS: their evaluation prompts are drawn from the same 1,000-prompt-per-category generator
  as their training prompts (700 train / 300 val, same templates, same object vocabulary), so part
  of their CompBench gain is in-domain template adaptation; ours is fully out-of-domain. Say this
  explicitly — it is a point in our favour that a reader will otherwise assume the other way.
* **Full-parameter fine-tuning vs LoRA.** Every comparator in this literature (GORS, CTCal, ++)
  fine-tunes rank-4 LoRA on attention only. We train all transformer parameters (DS:361). Nothing
  wrong with it, but it means our "same budget" claims need an explicit parameter-count footnote.
* **No EMA; post-hoc average of the last 5 checkpoints instead** (RM step 6). Non-standard but
  documented and justified by the ±0.035 single-checkpoint spread at 118k. Keep the justification
  in the paper — reviewers will ask.
* **Gradient clipping fires on ~100% of updates** under the constant-LR schedule (RM "Converged
  schedule"), which makes every update a fixed-size normalised step. That is an unusual regime;
  the cosine schedule fixes it, and the paper's main tables should use the cosine runs.
* **Symlinked staging + positional `question_id`.** `stage` symlinks images into `samples/` and
  relies on BLIP/CLIPScore emitting `question_id = position in the id-sorted listing`
  (CE:31-35, 290-302). The guard is correct and asserted, but it is a fragile contract worth a
  sentence in any artifact release.
* **Prompts containing `_` or `/` are skipped** (CE:135-136, 193-195). Verified currently 0/2,398
  affected, so no silent loss today — but a future pool change could shrink the benchmark quietly.
* **`--np_num 8`** matches the official BLIP-VQA default (`BLIP_vqa.py:65-70`) ✓ — no deviation.
* **Our 3-in-1 is a reimplementation**, not upstream `3_in_1.py`, because upstream hardcodes 10
  images per prompt and indexes by line position (CE:270-285). It is proven equal over all 300
  complex_val prompts by `phaseC/test_3in1_identity.py`. Cite that test if challenged.
* **The vendored T2I-CompBench HEAD `1b70949` includes a merged community PR** ("Add MiniMax M2.7 as
  alternative vision evaluator in gpt4v_eval.py", `ab5bec0`). It is upstream (merged by the repo
  owner) and touches only `MLLM_eval/gpt4v_eval.py`, which we do not run — harmless, but record the
  commit in the paper's reproducibility appendix (the eval driver already pins it under `pins`,
  E1:125).

---

## 5. The exact GORS data recipe, if we replicate it

Enough detail to reimplement without the CTCal code. Sources: CTCal p. 6 §4.1 for k/n, and
`T2I-CompBench/GORS_finetune/*` for everything the paper defers.

**Step 1 — prompts.** `T2I-CompBench/examples/dataset/{color,shape,texture,spatial,3d_spatial,
non_spatial,numeracy,complex}_train.txt`, 700 prompts each (files hold 700 lines; `wc -l` reports
699 for most because there is no trailing newline). Note GORS's own dataloader treats the **first
560** prompts of a category as "fixed template" and the rest as "natural"
(`train_dataset.py:59,64`).

**Step 2 — candidate generation.** For each prompt generate k images with the model being
fine-tuned: **k = 100 for SD 2.1, k = 30 for SD 3** (CTCal p. 6). Filenames must be
`"{prompt}_{index:06d}.png"` in one flat directory — the dataloader parses prompt and index out of
the filename (`train_dataset.py:42,106`; `Readme.md` "Generate images … examples/samples/").
Inherited generation config: `StableDiffusionPipeline` + `DDIMScheduler`, 512×512, 30 steps,
guidance 7.5, `manual_seed(42)`, then `42+n+1` per additional image
(`inference_eval.py:70,102,105,106`).

**Step 3 — scoring.** Run the ++ evaluator for that category over the candidate directory and emit
`{reward_root}/vqa_result.json` as a list of `{"question_id": int, "answer": "0.6900"}` in the
directory's id-sorted order (`Readme.md`, `train_dataset.py:43-45`). Category → scorer:
B-VQA for color/shape/texture, UniDet-2D for spatial, UniDet-3D for 3d_spatial, UniDet-numeracy for
numeracy, CLIPScore for non_spatial, 3-in-1 for complex. (GORS-*unbiased* instead uses
Grounded-SAM for attributes and GLIP for spatial/numeracy to avoid reward/metric circularity —
CTCal does **not** do this, it uses the metric itself as reward, so a CTCal-style replication
carries that circularity; flag it if we ever build on this.)

**Step 4 — selection.** CTCal: keep the **top n = 10,000** scored pairs per category (so SD2.1 keeps
10k of 70k, SD3 10k of 21k). Published GORS instead thresholds
(`train_dataset.py:59-68`): first-560 prompts need reward > **0.92** color / 0.85 shape / 0.9
texture / 0.8 spatial / 0.75 non-spatial / 0.4 complex; remaining prompts need > **0.7** color /
0.6 shape / 0.65 texture / 0.8 spatial / 0.75 non-spatial / 0.4 complex.

**Step 5 — fine-tune.** Reward-weighted diffusion loss
`L = E[ s · ||ε − ε_θ(z_t,t,y)||² ]` (`train_text_to_image.py:820,833`), LoRA rank 4 injected into
the UNet `["CrossAttention","Attention"]` modules and the text encoder `["CLIPAttention"]`
(`:517-519,532-535`), AdamW β=(0.9,0.999) ε=1e-8 wd=0.01 (`:325-329`), lr 1e-4 unet / 5e-6 text
(`:250-260`), constant schedule with 500 warm-up (`:268-278`), grad-norm clip 1.0 (`:329`),
per-device batch 2 × 8 GPUs (`:220-224`, `train.sh:8`; ++ paper states effective batch 5),
`--max_train_steps 15000` (`:226-230`), resolution 512 with center-crop and random h-flip
(`:192-213`, `train_dataset.py:81-98`), fp16 via accelerate (`train.sh:7`), `DDPMScheduler`
(`:504`), seed 42 (`:191`). One LoRA **per category** for the CompBench table; one joint LoRA over
all 80,000 pairs for the GenEval table (CTCal p. 7).

**Step 6 — evaluate.** 10 images per prompt on the 300-prompt `*_val.txt` split, fixed seed shared
across all compared models (`inference_eval.py:23`, `Readme.md:310`), then the official scripts,
with **Share-CoT** for non-spatial and **3-in-1** for complex if the target is CTCal's Table 1.
