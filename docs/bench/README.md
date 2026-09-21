# Benchmark-prompt distillation (2026-09-18 to 09-20): the GORS / CTCal data protocol on our recipe

The question: how much does scored selection move a student when the training prompts are the
benchmark's own and the scorer is the benchmark's own evaluator, and how does that compare with CTCal
(Guo et al., CVPR 2026), whose Table 1 is built on the same data protocol (GORS, T2I-CompBench++)?

## Protocol

| | |
|---|---|
| prompts | the 5,600 T2I-CompBench++ **train** prompts, 41 removed for appearing in our held-out val pool (colour 21, shape 15, texture 5) = **5,559**, ~700 per category (`pool_train_prompts.json`, `pool_manifest.json`) |
| candidates | **16** per prompt from the frozen SD3.5-M teacher, **K=10** Euler steps, cfg 7, 512 px, seeds `idx*1000 + j` (the trainer re-rolls the same trajectory) |
| scorer | the **official T2I-CompBench++ evaluator of the prompt's own category** on every candidate: BLIP-VQA (colour / shape / texture), UniDet (2D-spatial, 3D-spatial, numeracy), CLIPScore (non-spatial), 3-in-1 (complex); `--routing_split train` for the complex prompts |
| arms | `--selector bench` (the evaluator's argmax trajectory) vs `--selector random` (a random one of the 16); nothing else differs |
| schedule | the converged one: cosine to 0, peak lr 1e-5, batch 16 (4 GPUs x accum 4), warm-up 150, window states 6-9 of the K=10 grid, **8 passes = 2,780 updates**, average of the 2000 / 2500 / final checkpoints |
| reward | none by default: these prompts have no photograph, so the DINO-photo reward has no target (a substitute target was built and tested, below) |
| evaluation | the 2,398 held-out val prompts, 4 steps, guidance 1, scheduler grid, 1 image per prompt (`scripts/eval_alignment.lsf`) |

Selection headroom in the cache (`cache_meta.json`; official score of the argmax candidate vs the
mean candidate): 2D-spatial +0.275, numeracy +0.260, 3D-spatial +0.242, shape +0.230, colour +0.184,
texture +0.150, complex +0.107, non-spatial +0.023 (CLIPScore cannot discriminate; see Share-CoT below).

## Result: selection is worth +0.0115 CompBench, replicated over three seeds

`DELTAS.md` (from `eval/bench_deltas.py`, averaged checkpoints), the random control on the same cache
being the only difference:

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| evaluator-argmax | 0.5012 | 0.4986 | 0.5011 | **0.5003** |
| random pick | 0.4892 | 0.4880 | 0.4893 | 0.4888 |
| **delta** | +0.0121 | +0.0106 | +0.0119 | **+0.0115** |
| GenEval2 delta | +0.0107 | +0.0145 | +0.0105 | +0.0119 |

Pooled per-prompt paired test over the three seeds: win rate 0.539, sign p 2e-9, Wilcoxon p 1e-14,
n = 7,194. Per category (mean over seeds): 2D-spatial +0.026, shape +0.021, texture +0.017, colour
+0.012, complex +0.008, 3D-spatial +0.006, numeracy +0.002, non-spatial +0.000.

Three single-seed extensions (seed 0, averaged checkpoints):

| | argmax | random | delta |
|---|---|---|---|
| + projector reward on the best-of-8 teacher reference (below) | 0.5011 | 0.4883 | +0.0128 |
| sampled at **8 steps** (scheduler grid) | **0.5107** | 0.5004 | +0.0103 |
| sampled on **grid A** (`docs/nested_grid/`) | 0.5052 | 0.4933 | +0.0119 |

- **In-domain prompts alone are worth nothing.** The random-pick student on benchmark prompts
  (0.4892) matches the random-pick student on COCO captions on the same K=10 grid (0.4877, seed 0,
  `docs/k10/RESULTS.md`). The whole gain is the selection.
- **0.5107 at 8 steps** is the highest CompBench number the project has produced, above the 28-step
  guided teacher (0.5053), but 8 steps lifts the random control by the same amount: a sampler
  effect, not a training one (the step-count trade is measured in `docs/CHECKPOINTS.md`).
- The checkpoint `phaseW_CD_bench_hard_bench-k10-hp1-acc4_s0_155631` is tuned to the benchmark's
  prompt templates; the COCO-trained students of `docs/CHECKPOINTS.md` are the ones to use for
  anything else.

## Is the official argmax real? An independent judge says yes

The same 16 candidates of 200 prompts per category re-scored with VQAScore (clip-flant5-xxl, the
oracle of the COCO caches, no relation to BLIP-VQA / UniDet; `eval/bench_independent_check.py`,
`independent_check.json`):

| category | official argmax = VQAScore argmax | chance | VQAScore of the official pick | mean | best-of-16 | headroom recovered |
|---|---|---|---|---|---|---|
| colour | 0.115 | 0.063 | 0.948 | 0.902 | 0.975 | **62%** |
| texture | 0.135 | 0.063 | 0.943 | 0.892 | 0.976 | **61%** |
| numeracy | 0.140 | 0.063 | 0.742 | 0.674 | 0.870 | **35%** |
| 2D-spatial | 0.150 | 0.063 | 0.848 | 0.790 | 0.957 | **34%** |

Agreement is about twice chance in every category and the official pick recovers a third to
two thirds of the independent judge's own headroom, so the selection is genuine, not detector noise.

**What limits the transfer is the student's remaining headroom, not the selection.** Colour recovers
the most selection quality but moves the student least (already 0.80, its teacher at 0.798); spatial
recovers the least but moves it most (0.23 to 0.26, teacher 0.27). A better scorer will not fix colour
or numeracy; only a better generator will.

## The projector reward is a null here (predicted, then confirmed)

The paper's arm adds a reward through the latent projector towards the DINO embedding of the
caption's photograph. CompBench prompts have none, so a target was built: 8 reference candidates per
prompt at the teacher's own documented setting (**40 steps, cfg 4.5**, the SD3.5-M model card, not
the pipeline defaults of 28 / 7.0; `data/build_bench_candidates.py --K 40 --cfg 4.5 --N 8`), scored
by the official evaluator, the DINO patch-mean embedding of the best one as the reference
(`data/build_bench_ref_emb.py`; `ref_sheet.jpg` shows best vs worst of the 8 per category).
Reference-selection headroom (best-of-8 minus mean-of-8): 2D-spatial +0.214, numeracy +0.205,
3D-spatial +0.190, shape +0.171, colour +0.121, texture +0.114, complex +0.061, non-spatial +0.016.

Trained on the same cache with `REWARD_MODE=proj REWARD_LAMBDA=80 REWARD_REFRESH=100
REWARD_REFRESH_STEPS=16 REWARD_REF=cache/reward/ref_emb_bench.pt`: **0.5011 with the reward vs 0.5012
without** (random: 0.4883 vs 0.4892). The training monitors predicted it: the reward moved only
0.391 to 0.440 over 2,780 updates, the true DINO score of the predictions stayed at 0.72, and the
correlation between the projector's score and the true DINO score ended at 0.03; the consistency
loss was unaffected (28.10 vs 27.57), so convergence was fine. The reward had nothing to pull toward:
the target is a teacher image, and a student distilling teacher trajectories already resembles it.
On COCO, where the target is a real photograph, the same reward is worth about +0.002.

Open item: Share-CoT scores of the non-spatial reference candidates were computed (the CLIPScore
headroom there is +0.016, i.e. the best-of-8 is nearly a coin flip) but never merged into a second
reference set or retrained; given the null above, not worth the two 7-hour runs.

## Versus CTCal (CVPR 2026)

Their repository (`xiefan-guo/ctcal`) holds no code; settings were recovered from the paper and from
T2I-CompBench++ / GORS, and their Table 1 baselines are copied verbatim from ++ Table XIII. Audit:
`docs/lit/ctcal_alignment_audit.md`. Their absolutes are SD3 (2B) at 1024 px with ~30 guided steps,
ours a 4-step guidance-free student at 512 px, so the fair comparison is each method's **delta over
its own baseline**:

| category | ours (3 seeds) | CTCal | |
|---|---|---|---|
| 2D-spatial | +0.026 | +0.028 | tie |
| texture | +0.017 | +0.025 | |
| **shape** | **+0.021** | +0.008 | win |
| **3D-spatial** | **+0.006** | +0.003 | win |
| **complex** | **+0.008** | +0.004 | win |
| colour | +0.012 | +0.031 | |
| numeracy | +0.002 | +0.012 | |
| non-spatial | +0.000 | +0.009 | (CLIPScore vs Share-CoT: not the same column, see below) |
| **mean** | **+0.0115** | **+0.0150** | |

We do not beat them on the mean (0.003 behind; +0.0128 on the seed-0 pair with the reward on both
arms). We win shape, 3D-spatial and complex, tie 2D-spatial, lose colour, numeracy and non-spatial.
The colour gap is the whole story: our student is already at 0.80, its teacher's level, so selection
has nothing to add; their SD3 base starts lower with room above. What we have that they do not:
4 to 8 forward passes at inference instead of ~60, and (for the COCO-trained students) training that
never sees a benchmark prompt.

Absolute numbers, for reference (categories with a CTCal column; non-spatial below):

| model | steps | px | colour | shape | texture | 2D-sp | 3D-sp | numeracy | complex |
|---|---|---|---|---|---|---|---|---|---|
| SD3 (2B) base, their baseline | ~30 | 1024 | 0.813 | 0.589 | 0.733 | 0.320 | 0.408 | 0.617 | 0.377 |
| SD3 (2B) + CTCal | ~30 | 1024 | 0.844 | 0.597 | 0.758 | 0.348 | 0.412 | 0.629 | 0.381 |
| our teacher, SD3.5-M | 28 | 1024 | 0.784 | 0.586 | 0.744 | 0.295 | 0.388 | 0.632 | 0.368 |
| our teacher, SD3.5-M | 28 | 512 | 0.798 | 0.573 | 0.743 | 0.272 | 0.366 | 0.592 | 0.384 |
| our best student (bench, argmax) | 8 | 512 | 0.813 | 0.578 | 0.756 | 0.284 | 0.374 | 0.592 | 0.380 |
| our best student (bench, argmax) | 4 | 512 | 0.804 | 0.570 | 0.757 | 0.271 | 0.343 | 0.564 | 0.386 |
| SD3.5-M base, undistilled | 4 | 512 | 0.417 | 0.292 | 0.405 | 0.058 | 0.141 | 0.257 | 0.205 |

Non-spatial on **Share-CoT**, the evaluator of their column (`docs/sharecot.md`): our student 0.773,
our teacher 0.780, their base 0.778, their + CTCal 0.787. Our CLIPScore non-spatial column (~0.31) is
not comparable to theirs.

A 512-trained student does not transfer to 1024 (student 0.5013 at 512 -> 0.4817 at 1024, while the
teacher gains, 0.5053 -> 0.5142), so an absolute comparison at their resolution would need a 1024
cache and a retrain; not recommended, since our teacher at 1024 already trails their SD3 base on the
categories where we are behind.

## Qualitative

- `win_sheet.jpg`: the largest per-prompt gains of the argmax arm over the random control in the
  categories where our delta beats CTCal's (same prompt, same noise, same recipe).
- `ctcal_sheet_seed0.jpg`: the exact prompts of CTCal's figures (`ctcal_prompts.json`), our teacher-28
  and the three 4-step students side by side.
- `ref_sheet.jpg`: the reward references, best vs worst of 8 per category.

## Reproduce

```
python data/build_bench_pool.py                                   # pools/bench_train/ (5,559 prompts)
for S in 0 1; do bsub -env "all,SHARD=$S,NSHARD=2" < scripts/build_bench_candidates.lsf; done   # 16 x K=10 candidates
for C in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  bsub -env "all,CAT=$C" < scripts/score_bench_candidates.lsf; done                            # official evaluators
python data/build_bench_selection.py                              # cache/bench_k10_n16 (prints the headroom table)
for SEL in bench random; do
  bsub -env "all,SELECTOR=$SEL,CACHE=cache/bench_k10_n16,K=10,WINDOW=0.6:0.9,EPOCHS=8,ACCUM=4,LR=1e-5,WARMUP=150,LR_SCHEDULE=cosine,SAVE_EVERY=500,TAG=-bench" < scripts/train.lsf
done                                                              # then scripts/average_checkpoints.lsf STEPS=2000:2500:final, scripts/eval_alignment.lsf
python eval/bench_deltas.py --artifacts <tree with the phaseN/ records>          # DELTAS.md
bsub < scripts/bench_independent_check.lsf                        # VQAScore judge, independent_check.json
# the reward arm: reference candidates at the teacher's setting, scored, embedded, then REWARD_* on scripts/train.lsf
for S in 0 1; do bsub -env "all,SHARD=$S,NSHARD=2,N=8,K=40,CFG=4.5,LABEL=bench_ref_k40cfg45" < scripts/build_bench_candidates.lsf; done
for C in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  bsub -env "all,CAT=$C,N=8,K=40,LABEL=bench_ref_k40cfg45" < scripts/score_bench_candidates.lsf; done
bsub < scripts/build_bench_ref_emb.lsf                            # cache/reward/ref_emb_bench.pt
```

Records behind the tables: `phaseN/eval_W_{CD_bench_hard,B2}-k10-hp1_bench-hp1{,-avg}_s{0,1,2}_*`
(the `-rew-` variants for the reward arms; `eval_W_bench-{argmax,random}_steps8_s0_*` and
`eval_W_bench-{argmax,random}-gridA_s0_*` for the sampler rows) on the experimental tree.
