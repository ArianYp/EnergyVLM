# Scored consistency distillation

Few-step distillation of SD3.5-Medium where the student imitates a **selected** teacher
trajectory instead of a random one. For every training caption the frozen teacher samples four
candidates; a scorer picks one; the student is trained by consistency distillation on that
trajectory only. Two arms, identical in every other respect:

| arm | selector |
|---|---|
| `random` (naive distillation) | a fixed uniform draw among the four candidates |
| `dino_patch` (scored distillation) | the candidate whose mean-pooled DINOv2 patch embedding is closest to the caption's real photograph |
| `dino_patch` + projector reward, refreshed (**the paper's arm, "ours"**) | scored selection plus a reward on the student's own clean-latent estimates through a latent-to-DINO projector (`--reward_mode proj --reward_lambda 80`), the projector refreshed every 100 updates for 16 steps on decoded recent predictions (`--reward_refresh_every 100 --reward_refresh_steps 16`); see `paper/` |

The scorer needs no text model. The caption enters only through the photograph it was written for,
so training captions must come from an image-caption corpus (COCO here).

Three more selectors exist for the selection-rule ablation of the report (`--temp` sets T):

| selector | what it trains on | verdict |
|---|---|---|
| `boltzmann` | all four candidates, each loss weighted by softmax(S/T); 4 rollouts + 4 student passes per caption | no detected improvement over `dino_patch` at T=0.04 or 0.08, 2.3x the compute |
| `boltzmann_sample` | one candidate drawn from softmax(S/T) on every visit | 0.004 below `dino_patch` after checkpoint averaging (0.014 on raw finals: final-iterate noise) |
| `boltzmann_frozen` | one softmax(S/T) draw per caption (`--map_seed`), fixed for the run | 0.002 below `dino_patch` after averaging; 0.003 above `boltzmann_sample` |
| `boltzmann_mc` | `--mc_draws` iid draws per visit, losses weighted by count / draws | same expected gradient as `boltzmann_sample`, variance / draws; implemented, not run |
| `uniform_visit` | one uniform draw on every visit (vs `random`, which draws once per caption) | level with `random` after averaging |
| `latent` | argmax of `latent_cos`: a 10.7M-parameter projector from the terminal latent into DINO space, scored against the photograph, no decode (`data/build_latents.py`, `train/latent_scorer.py`) | recovers ~3/4 of the `dino_patch` gain (offline headroom 34% vs 44%; trained +0.010 vs +0.014 over random, 3 seeds); `boltzmann --score_field latent_cos` behaves the same |

## Layout

```
common/      sampling.py (teacher rollout, decode)  distributed.py (torchrun setup)
data/        build_pool.py       training captions paired with their photographs
             build_eval_pool.py  T2I-CompBench, GenEval2 and COCO-val prompt pools
             build_candidates.py the candidate cache: 4 trajectories per caption, scored
             build_latents.py    terminal latents + DINO embeddings of held-out captions (latent scorer data)
             build_latent_manifest.py  the train / val / test split of those captions
             build_reward_refs.py  reference-photo embeddings and a replay set for the reward arms (3k pool)
             build_ref_emb.py    reference-photo embeddings for every caption of any cache (118k pool)
train/       distill.py          the trainer (--selector random | dino_patch | latent | boltzmann |
                                 boltzmann_sample | boltzmann_frozen | boltzmann_mc |
                                 uniform_visit; --accum for larger batches; --reward_mode proj | rgb
                                 adds a reward on the student's clean estimates, --reward_refresh_every
                                 refreshes the projector (any number of GPUs); --lr_schedule cosine
                                 is the paper's converged schedule)
             latent_scorer.py    the decode-free latent projector (trained once, offline)
             average_checkpoints.py  uniform average of the last checkpoints of a run
eval/        generate.py         sample a model on a prompt pool (paired noise per prompt)
             compbench.py        T2I-CompBench with the official evaluators
             geneval2.py         GenEval2 with the official evaluator
             fidelity.py         FID, CMMD, precision, recall vs COCO val2017
             compare_arms.py     paired per-prompt comparison of evaluated models
             grad_diagnostic.py  per-candidate gradient geometry of the selection rules
             latent_scorer_regret.py  the latent scorer on students' own samples
             heldout_dino.py / heldout_compare.py  held-out DINO monitor of the reward arms
             reward_monitor.py   the in-training reward vs. true-score monitor, from wandb
scripts/     LSF launchers for each stage; env.sh holds cluster paths
paper/       the paper (iclr2027_conference.tex / .pdf): ours vs naive distillation, with the
             scripts that recompute its numbers, tables and figures from the raw evaluation records
docs/        the full technical report (report.tex / .pdf, every arm and ablation) and its records
third_party/ (not included) T2I-CompBench, GenEval2, t2v_metrics clones, see below
```

## Method

SD3.5 is a rectified flow: `z_sigma = (1 - sigma) x0 + sigma eps`, velocity `v = eps - x0`, Euler
sampling `z_{k+1} = z_k + (sigma_{k+1} - sigma_k) v`. The teacher runs K = 8 steps with
classifier-free guidance w = 7. Candidate j of caption `idx` starts from `manual_seed(idx*1000 + j)`,
so the cache stores only scores and the trainer re-rolls the winner from its seed.

Score: `S_j = cos( mean_patches DINOv2(candidate_j), mean_patches DINOv2(reference photo) )`,
CLS token dropped. Selection: `argmax_j S_j`, frozen offline; no gradient reaches the scorer. (The
reward arms below are the exception: `--reward_mode rgb` back-propagates through the frozen scorer
into the student.)

Distillation, per update, over teacher states k in the window 0.4K..0.9K:
```
v_k    = (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k)
x0_k   = z_k - sigma_k v_k                       teacher clean-latent estimate (stop-gradient)
x0_hat = z_{k-d} - sigma_{k-d} v_theta(z_{k-d}, c),   d ~ U{1,2,3}
loss   = mean_k  sqrt(||x0_hat - x0_k||^2 + c^2) - c,   c = 0.00054 sqrt(D)
```
The student makes one conditional forward, so guidance is absorbed; sample it with cfg 1.

The paper's arm adds a reward on the student's own clean estimates at the two least-noisy supervised
states, through the latent projector `P` (`train/latent_scorer.py`: 10.7M parameters, terminal latent
-> DINO patch space, pretrained once on 22k held-out captions' candidates):
```
loss  +=  -lambda * mean_{k in the 2 least-noisy states}  < P(x0_hat_k), u(photo) >,     lambda = 80
```
lambda = 80 puts the reward gradient at about 20% of the consistency gradient. Every 100 updates
the projector is refit for 16 AdamW steps to the decoded RGB DINOv2 score of the last 32 predictions
(plus a replay batch of teacher candidates with a ranking KL, so it keeps ranking those too); the
refresh runs on rank 0 and the projector is broadcast to the other ranks. The reward costs about 2%
of the update at batch 16 (the exact decode-based reward, `--reward_mode rgb`, costs 14-16%).

## Running

```
# 0. external code (not vendored here)
git clone https://github.com/Karine-Huang/T2I-CompBench third_party/T2I-CompBench
git clone https://github.com/facebookresearch/GenEval2     third_party/GenEval2
git clone https://github.com/linzhiqiu/t2v_metrics          third_party/t2v_metrics   # only for --vqa
pip install -r requirements.txt

# 1. prompt pools
python data/build_pool.py --coco_root $COCO_ROOT --split train2017 \
    --out_prompts pools/train/prompts.json --out_manifest pools/train/pool_manifest.json
python data/build_eval_pool.py --out_root pools/eval \
    --compbench_dir third_party/T2I-CompBench/examples/dataset \
    --geneval2_data third_party/GenEval2/geneval2_data.jsonl \
    --coco_captions $COCO_ROOT/annotations/captions_val2017.json

# 2. candidate cache (4 x 8 GPUs shown; ranks are independent)
for S in 0 1 2 3; do bsub -env "all,SHARD=$S,NSHARD=4" < scripts/build_candidates.lsf; done

# 3. the two arms
bsub -env "all,SELECTOR=random,SEED=0"     < scripts/train.lsf
bsub -env "all,SELECTOR=dino_patch,SEED=0" < scripts/train.lsf

# 4. alignment (CompBench + GenEval2), students at cfg 1, base at cfg 7
bsub -env "all,LABEL=random_s0,CKPT=checkpoints/random_s0/checkpoint_final.pt,CFG=1.0"         < scripts/eval_alignment.lsf
bsub -env "all,LABEL=dino_patch_s0,CKPT=checkpoints/dino_patch_s0/checkpoint_final.pt,CFG=1.0" < scripts/eval_alignment.lsf
python eval/compare_arms.py --root out/eval --baseline random --arms dino_patch

# 5. fidelity
bsub -env "all,LABEL=random_s0,CKPT=checkpoints/random_s0/checkpoint_final.pt,CFG=1.0,STEPS=4"         < scripts/fidelity_generate.lsf
bsub -env "all,LABEL=dino_patch_s0,CKPT=checkpoints/dino_patch_s0/checkpoint_final.pt,CFG=1.0,STEPS=4" < scripts/fidelity_generate.lsf
bsub -env "all,LABEL=base,CKPT=base,CFG=7.0,STEPS=28"                                                   < scripts/fidelity_generate.lsf
bsub < scripts/fidelity_score.lsf

# 6. at the 118k scale: average the last five checkpoints and evaluate the average
A=$(bsub -env "all,RUN=checkpoints/dino_patch_s0" < scripts/average_checkpoints.lsf | grep -oE "[0-9]+")
bsub -w "done($A)" -env "all,LABEL=dino_patch_avg_s0,CKPT=checkpoints/dino_patch_s0/checkpoint_avg_last5.pt,CFG=1.0" \
     < scripts/eval_alignment.lsf

# 7. absolute per-category tables (markdown + LaTeX)
python eval/absolute_tables.py --model "naive=out/eval/eval_random_s[0-9]" \
    --model "DINO patches=out/eval/eval_dino_patch_s[0-9]" --tex out/absolute_tables.tex

# 8. the reward arms (3k configuration): reference embeddings + replay from the latent shards of
#    train/latent_scorer.py, then a reward on the student's own clean estimates (report 11.6-11.7)
python data/build_reward_refs.py --shards cache/latents --out_dir cache/reward
bsub -env "all,SELECTOR=dino_patch,SEED=0,REWARD_MODE=proj,REWARD_LAMBDA=80"                    < scripts/train_3k.lsf   # frozen projector
bsub -env "all,SELECTOR=dino_patch,SEED=0,REWARD_MODE=proj,REWARD_LAMBDA=80,REWARD_REFRESH=100" < scripts/train_3k.lsf   # refreshed projector, 4 steps
bsub -env "all,SELECTOR=dino_patch,SEED=0,REWARD_MODE=proj,REWARD_LAMBDA=80,REWARD_REFRESH=100,REWARD_REFRESH_STEPS=16" \
     < scripts/train_3k.lsf                                                                                              # THE PAPER'S ARM (ours, 3k)
bsub -env "all,SELECTOR=dino_patch,SEED=0,REWARD_MODE=rgb,REWARD_LAMBDA=15.5"                   < scripts/train_3k.lsf   # exact decode + DINOv2

# 9. the paper's arm at 118k: reference embeddings for every caption of the 118k cache (the replay set
#    and the projector are the 3k ones), then 4 GPUs (the refreshed projector is broadcast from rank 0),
#    then step 6
python data/build_ref_emb.py --cache cache/train --out cache/reward/ref_emb_118k.pt --check cache/reward/ref_emb.pt
bsub -env "all,SELECTOR=dino_patch,SEED=0,REWARD_MODE=proj,REWARD_LAMBDA=80,REWARD_REFRESH=100,REWARD_REFRESH_STEPS=16,REWARD_REF=cache/reward/ref_emb_118k.pt" \
     < scripts/train.lsf

# 10. the paper's converged schedule (Section 3.6): cosine decay, batch 16 = 4 GPUs x ACCUM 4, lr 1e-5,
#     window K={4..7}; 3k = 16 passes (3,000 updates), 118k = 2 passes (14,244 updates); one seed
bsub -env "all,SELECTOR=random,CACHE=cache/train_3k,EPOCHS=16,ACCUM=4,LR=1e-5,WARMUP=150,LR_SCHEDULE=cosine,WINDOW=0.5:0.9,SAVE_EVERY=500,TAG=-3k" < scripts/train.lsf
bsub -env "all,SELECTOR=random,ACCUM=4,LR=1e-5,WARMUP=700,LR_SCHEDULE=cosine,WINDOW=0.5:0.9,SAVE_EVERY=1250" < scripts/train.lsf
#     (add the REWARD_* settings of step 9 for "ours"; average STEPS=2000:2500:final resp. 10000:11250:12500:13750:final)
python eval/heldout_dino.py --ckpt checkpoints/dino_patch-rewX_3k_s0/checkpoint_avg_last5.pt \
    --manifest cache/latents/manifest.jsonl --out out/heldout/dino_patch-rewX_s0@avg_last5.json   # per checkpoint, every arm
python eval/heldout_compare.py --dir out/heldout
python eval/reward_monitor.py --project $WANDB_PROJECT
```

Two training configurations are used in the report. `scripts/train_3k.lsf` is the small one
(3,000 captions, one GPU, 6,000 updates, lr 1e-5, 300 warm-up steps; every three-seed result and
every ablation). `scripts/train.lsf` is the scale one (113,948 captions, four GPUs, 56,974 updates,
lr 2e-5, 1,000 warm-up steps). At the scale configuration single checkpoints of one run differ by
up to 0.035 CompBench, so the reported model is the uniform average of its last five checkpoints
(step 6 above); `train/average_checkpoints.py` writes it in the same layout as a checkpoint.

Evaluate students with `CFG=1.0`. Sampling a student with guidance applies it twice and roughly
halves its scores.

`IMAGES_PER_PROMPT=10` on `scripts/eval_alignment.lsf` runs T2I-CompBench's official protocol
(10 images per prompt, per-prompt mean); the default 1 is the cheap sweep setting and candidate 0
is the same image in both. Every `alignment.json` records the evaluator repo commits, package
versions, prompt-pool hashes and the checkpoint sha256 under `pins`, and a `pip_freeze.txt` is
written beside it. Missing generations make staging fail rather than silently shrink the benchmark.

In-domain alignment (VQAScore on the held-out COCO captions of the fidelity pool):
`python eval/coco_vqascore.py --images out/fidelity/images/dino_patch_s0_s4 --steps 4
--prompts pools/eval/fidelity_prompts.json --out out/coco_vqa/dino_patch_s0.json`.

`eval/fidelity.py` reports CMMD per the reference implementation (CLIP ViT-L/14@336, center crop
then bicubic resize, unit-norm embeddings, RBF sigma 10, biased estimator, x1000); it agrees with
the public PyTorch port to three decimals on a frozen image set.

GenEval2's judge (Qwen3-VL) needs `transformers >= 4.57`; if the training environment pins an
older version, point `GENEVAL2_PYTHON` at a second interpreter that has it.

## Results

### The paper (`paper/`)

`paper/iclr2027_conference.tex` (compiled: `paper/iclr2027_conference.pdf`) reports one arm against
naive distillation only. **Ours** = `dino_patch` selection + the projector reward, lambda 80, projector
refreshed every 100 updates for 16 steps (`-rewR-s16` in the launchers; steps 8-9 above). Every number
in it is recomputed from the raw evaluation records by `paper/verify_numbers.py` (-> `paper/numbers.json`),
the tables by `paper/make_tables.py`, the figures by `paper/make_paper_figures.py`; like
`docs/figs/make_figures.py` these read the experimental branch's `phaseN/` and `phaseW/` records.

| setting | naive CD | ours | ours - naive, seed-paired (p) |
|---|---|---|---|
| 3k captions, 6k updates, 3 seeds, average of 2k/4k/6k: CompBench | 0.4738 +- 0.0017 | 0.4887 +- 0.0025 | +0.0149 +- 0.0017 (0.004) |
| same, GenEval2 (x100) | 22.53 +- 0.95 | 23.54 +- 1.56 | +1.00 +- 0.91 (0.20) |
| same, CMMD / precision / recall / FID | 0.837 / 0.479 / 0.054 / 30.42 | 0.817 / 0.474 / 0.067 / 30.89 | -0.020 (0.32) / -0.005 (0.60) / +0.013 (0.02) / +0.47 (0.03) |
| 118k captions, 57k updates, 3 seeds, last-5 average: CompBench | 0.4668 +- 0.0036 | 0.4810 +- 0.0050 | +0.0142 +- 0.0085 (0.10; prompt-level p < 0.001) |
| same, GenEval2 (x100) | 20.29 +- 0.48 | 21.66 +- 1.13 | +1.37 +- 1.28 (0.21; prompt-level p 0.02) |
| same, CMMD / precision / recall / FID | 0.837 / 0.492 / 0.062 / 31.16 | 0.873 / 0.479 / 0.051 / 32.09 | +0.037 (0.008) / -0.014 (0.04) / -0.011 (0.02) / +0.92 (0.007) |

At 3k the gain is in colour (+0.021, p 0.02), shape (+0.019, p 7e-5) and complex prompts (+0.012,
p 0.01); at 118k the direction holds in every category but non-spatial, and the arm's fidelity is worse
than naive's on every metric and seed (the paper states this as a limitation). Reward variants at 3k,
all against naive (three seeds, averaged checkpoints): projector frozen +0.0109 (p 0.13); refreshed every
100 updates for 4 steps +0.0138 (0.12); every 25 updates +0.0128 (0.03); every 100 updates for 16 steps
+0.0149 (0.004) = ours; ours with max-pooled DINOv2 patches +0.0077 (0.03) -- that row needs max-pooled
cache / projector / reference artifacts, which only the experimental branch builds, and its comparison is
confounded: under max pooling the within-caption score spread is 10x smaller, so the same lambda gives a
10x smaller reward gradient. Against argmax selection alone (0.4868 over five seeds, factorial below) the
16-step-refreshed projector reward adds +0.002, unresolved; the paper compares to naive CD only.

**Converged schedule** (paper Section 3.6; one seed, last-checkpoint averages; step 10 above). Under
the constant-LR schedule of every other run the pre-clip gradient norm (130-900) exceeds the clip of 1
on every update, so each update is a fixed-size normalised step, the loss falls by the same 7% at both
scales and single checkpoints oscillate (`train/clip_coef` and `train/dist_from_teacher` in the wandb
log show it). With cosine decay to 0, batch 16, lr 1e-5 and the window narrowed to K={4..7}: 3k pool,
3,000 updates = 16 passes, naive 0.4860 / ours 0.4877 CompBench (raw finals 0.4846 / 0.4878); 118k pool,
14,244 updates = 2 passes, naive 0.4747 / ours 0.4818 (raw 0.4744 / 0.4780; GenEval2 21.9 / 22.3). Raw
and averaged checkpoints now agree, naive CD alone rises above every constant-LR 118k student, the arm
contrast shrinks to +0.002 at 3k and +0.007 at 118k (single seed), and both 118k students stay below
their 3k counterparts trained for sixteen passes.

### Prompt-aware ranking (`docs/rank/`, 2026-09-16)

A follow-up campaign tested whether a fixed text ranking over each caption and structured negatives
(`data/build_negatives.py`) can make the projector's DINO space prompt-aware and transfer that into
the student (`train/distill.py --rank_*`, `train/rank_utils.py`, `eval/rank_probe.py`). The
representation half holds modestly (a trained head orders positive over negative on unseen captions,
+5-9 points in projector space, photo-dependent); no transfer path improved the paper's arm on
CompBench or GenEval2, two hurt, and the shaped scorer used for candidate selection was worse than
raw DINO against the VQAScore oracle. Tables, mechanism and the review in `docs/rank/README.md`.

### The technical report (`docs/`)

Paired against the random-selection student on identical prompts. The 3k rows are three training
seeds with 95% hierarchical bootstrap intervals over seeds and prompts; the 118k rows are three
training seeds with weight-averaged checkpoints and per-prompt paired tests pooled over seeds. Full
tables, ablations, figures and the evaluation protocol are in `docs/report.pdf` (source
`docs/report.tex`, figures from `docs/figs/make_figures.py`).

| setting | selector | T2I-CompBench | GenEval2 (x100) | CMMD |
|---|---|---|---|---|
| 3k captions, 6k updates | random | 0.4584 | 20.73 | 0.963 |
| 3k captions, 6k updates | dino_patch | +0.0140 [+0.0051, +0.0237] | +2.17 [+0.24, +4.08] | 0.893 |
| 118k captions, 57k updates, averaged, 3 seeds | random | 0.4668 | 20.29 | 0.84 |
| 118k captions, 57k updates, averaged, 3 seeds | dino_patch | +0.0175 (p 1e-14) | +1.29 (p 0.02) | 0.78 |
| 118k, official 10-images-per-prompt CompBench | random / dino_patch | 0.4719 / +0.0131 (p 2e-41) | | |
| teacher, 28 steps, cfg 7 | | 0.5053 | 17.05 | 0.64 |

The scorer reads no text. Selection costs nothing at inference and does not cost fidelity: the
scored student has lower CMMD and higher precision and recall than the random-selection student on
every seed in both settings. What the ablations established:

- **Selection rule** (3k pool, three seeds, every arm evaluated on its raw final checkpoint and on
  the average of its 2k/4k/6k checkpoints, uncertainty at the level of training runs). Argmax over
  random is the one contrast established at seed level (+0.0203 +- 0.0008 raw, +0.0135 +- 0.0042
  averaged). Exact Boltzmann weighting of all four candidates shows no detected improvement over
  argmax at T=0.04 or 0.08 and costs 2.3x; uniform weighting is level with random, so the weighting
  toward the best candidate is what matters. Sampling one candidate per visit from the same weights
  is 0.014 below argmax on raw finals but 0.004 after averaging (the resampled arm's final iterate
  is four times noisier across seeds), and a uniform redraw per visit is level with a fixed draw
  after averaging; a frozen per-caption draw is 0.003 above the resampled arm. A gradient
  diagnostic on held-out captions shows why the estimators can differ under clipping and AdamW
  (candidate gradients nearly orthogonal, one-sample relative variance ~1, clipped one-sample
  update 25% shorter and parallel). Argmax is the recipe; "target churn" is not established as a
  separate mechanism.
- **Batch size.** Accumulating to 16 captions per update at the same data budget leaves the gap
  unchanged (+0.0221 vs +0.0223, seed 0) and lifts both arms by +0.005.
- **Candidates.** Eight candidates instead of four: +0.001 (null), although the offline headroom
  grows by 16%.
- **Teacher.** With a 28-step or a 16-step cfg-4.5 teacher the within-caption score spread halves
  and the trained gain vanishes: this is a fixed-teacher gain, not a substitute for teacher quality.
- **Latent scorer as a reward.** Adding `-lambda * cos(P(x0_hat), DINO(photo))` on the student's
  least-noisy clean estimates (lambda set so the reward gradient is 20% of the consistency
  gradient), with the projector frozen or refreshed every 100 updates on decoded predictions: the
  reward rises by 0.03-0.04 in every run, the true RGB score of the same predictions does not
  (-0.024 to 0.000), and CompBench is unchanged (-0.003 / +0.000 vs argmax after averaging). A
  proxy-optimisation signature relative to argmax. The projector ranks teacher candidates, not student
  predictions (top-1 agreement with the RGB scorer 0.37 on students' own samples vs 0.51). Refreshing
  for 16 steps instead of 4 (the paper's arm) gives the same picture against argmax (+0.002) and the
  cleanest contrast against naive distillation (+0.0149, p 0.004; the paper's Table 1).
- **The exact reward works.** Replacing the projector by the RGB scorer itself (`--reward_mode rgb`:
  VAE decode, differentiable resize/crop/normalise, DINOv2 in fp32, gradients through both; verified
  against the offline scorer to 0.011, lambda 15.5 for the same 20% gradient ratio, 1.17x wall-clock)
  raises the true DINO score of the predictions in training (0.583 -> 0.608), and after averaging
  gives CompBench +0.011 +- 0.006 over argmax (every seed; +0.025 over random, p 0.01), GenEval2 +1.5,
  CMMD 0.80 -> 0.69 with precision +0.05 and recall +0.02, and held-out DINO of the clean estimates
  +0.008 (p 0.01). Those first three runs saw a different caption order from the argmax arm (the
  scorer load advanced the global RNG before the first data draw; the trainer now restores it).
- **Selection x reward factorial** (five seeds per cell, caption order matched, averaged models):
  CompBench random 0.4751, random + reward 0.4819, argmax 0.4868, argmax + reward 0.4923. The two
  effects are additive (interaction -0.001 +- 0.006); selection stays worth +0.010 with the reward
  present (p 0.02), the reward is worth +0.0055 over argmax (p 0.004) and +0.007 over random; on
  fidelity the reward is the larger lever (CMMD -0.07 to -0.10, p 0.001) and selection's effect is
  not resolved. Argmax + exact reward vs naive distillation: +0.017 CompBench (p 0.002), +1.9 GenEval2
  (p 0.003), CMMD 0.73 vs 0.85. Half of the first-reported +0.011 was the caption order (`docs/reward/`).
- **Not adopted.** Regressing onto the reference photograph (lambda 0.2) costs 0.02-0.03 CompBench;
  an EMA-of-student teacher collapses at decay 0.999 (its online selection entropy rising to 0.93
  is the early warning) and is inert at 0.9999.
- **The reward is under-weighted at 15.5.** A one-factor ablation (three seeds each) shows CompBench,
  CMMD, precision and recall all improving monotonically from lambda 7.75 to 62, and rewarding all
  five supervised states instead of the two least-noisy adds +0.0093 CompBench (p 0.02). Combining
  lambda 31 with all five states (three seeds) gives CompBench 0.5082, +0.0153 over the recipe
  (p 0.012) and +0.0093/+0.0060 over either factor alone, CMMD 0.64 matching lambda 62's fidelity
  without its GenEval2 cost -- but GenEval2 (23.59) and FID (31.5) both move the wrong way past what
  either factor gives alone. The same reward re-tried on two stronger teachers (28-step w=7, 16-step
  w=4.5) moves CompBench in the same direction as on the 8-step teacher but is not resolved at three
  seeds (+0.005, +0.006, both n.s.).
- **The exact reward holds at the full 118k scale.** Unmodified (lambda 15.504), three seeds,
  56,974 updates: CompBench 0.4930 vs argmax-only 0.4843 (+0.0087, every seed positive, p 0.08) and
  vs random 0.4668 (+0.0262, p 0.003); GenEval2 23.03 vs 21.58 (+1.45). Fidelity improves further on
  top of selection's own gain: CMMD 0.683 vs argmax's 0.783 vs random's 0.837, FID 30.79 vs 31.09,
  precision 0.563 vs 0.518, recall 0.087 vs 0.067. Confirms the 3k-scale factorial result at 40x the
  caption count.

## Logging

With `WANDB_PROJECT` set, each run logs per step: loss, the loss at every supervised trajectory
state (`train/loss_k*`), gradient norm, learning rate, epoch, samples seen, steps/s, peak GPU
memory, `train/clip_coef` (1 when the clip is inactive; it is 0.001-0.01 on every update of the
constant-LR runs) and `train/dist_from_teacher` (the L2 distance of the student's weights from the
teacher's, which a run on a plateau keeps growing), and `sel/gain_<score>`: the running mean of (score of the selected candidate − mean over
the four candidates) under every scorer stored in the cache, i.e. what the arm's selection buys
under each scorer. Every `SAMPLE_EVERY` steps the student samples a fixed prompt list
(`SAMPLE_PROMPTS`, any json list of `{idx, prompt}`; noise seeded by `idx` like the evaluation
generator) and logs an image grid and a table; the guided 28-step teacher is logged once on the
same prompts as a reference. Sampling uses private generators only, so it does not change the
training trajectory.

Finished evaluations can be pushed to the same project with `eval/log_to_wandb.py` (summary
metrics per category, a per-prompt score table, fidelity numbers, and image grids), one run per
evaluated model:

```
python eval/log_to_wandb.py --project $WANDB_PROJECT --eval_dir out/eval/eval_dino_patch_s0 \
    --fidelity out/fidelity_report.json
```

Neither arm needs VQAScore. `data/build_candidates.py --vqa` additionally records
`endpoint_vqa` / `oracle_idx` per caption (for a VQAScore-selected arm) and requires
`third_party/t2v_metrics` to be importable in the environment.
