# Scored consistency distillation

Few-step distillation of SD3.5-Medium where the student imitates a **selected** teacher
trajectory instead of a random one. For every training caption the frozen teacher samples four
candidates; a scorer picks one; the student is trained by consistency distillation on that
trajectory only. Two arms, identical in every other respect:

| arm | selector |
|---|---|
| `random` (naive distillation) | a fixed uniform draw among the four candidates |
| `dino_patch` (scored distillation) | the candidate whose mean-pooled DINOv2 patch embedding is closest to the caption's real photograph |

The scorer needs no text model. The caption enters only through the photograph it was written for,
so training captions must come from an image-caption corpus (COCO here).

Three more selectors exist for the selection-rule ablation of the report (`--temp` sets T):

| selector | what it trains on | verdict |
|---|---|---|
| `boltzmann` | all four candidates, each loss weighted by softmax(S/T); 4 rollouts + 4 student passes per caption | ties `dino_patch` at T=0.04 and 0.08 for 2.3x the compute |
| `boltzmann_sample` | one candidate drawn from softmax(S/T) on every visit | loses 0.014 to `dino_patch` through target churn |
| `uniform_visit` | one uniform draw on every visit (vs `random`, which draws once per caption) | 0.008 below `random`: a changing target costs more than a worse one |

## Layout

```
common/      sampling.py (teacher rollout, decode)  distributed.py (torchrun setup)
data/        build_pool.py       training captions paired with their photographs
             build_eval_pool.py  T2I-CompBench, GenEval2 and COCO-val prompt pools
             build_candidates.py the candidate cache: 4 trajectories per caption, scored
train/       distill.py          the trainer (--selector random | dino_patch | boltzmann |
                                 boltzmann_sample | uniform_visit; --accum for larger batches)
             average_checkpoints.py  uniform average of the last checkpoints of a run
eval/        generate.py         sample a model on a prompt pool (paired noise per prompt)
             compbench.py        T2I-CompBench with the official evaluators
             geneval2.py         GenEval2 with the official evaluator
             fidelity.py         FID, CMMD, precision, recall vs COCO val2017
             compare_arms.py     paired per-prompt comparison of evaluated models
scripts/     LSF launchers for each stage; env.sh holds cluster paths
third_party/ (not included) T2I-CompBench, GenEval2, t2v_metrics clones, see below
```

## Method

SD3.5 is a rectified flow: `z_sigma = (1 - sigma) x0 + sigma eps`, velocity `v = eps - x0`, Euler
sampling `z_{k+1} = z_k + (sigma_{k+1} - sigma_k) v`. The teacher runs K = 8 steps with
classifier-free guidance w = 7. Candidate j of caption `idx` starts from `manual_seed(idx*1000 + j)`,
so the cache stores only scores and the trainer re-rolls the winner from its seed.

Score: `S_j = cos( mean_patches DINOv2(candidate_j), mean_patches DINOv2(reference photo) )`,
CLS token dropped. Selection: `argmax_j S_j`, frozen offline; no gradient reaches the scorer.

Distillation, per update, over teacher states k in the window 0.4K..0.9K:
```
v_k    = (z_{k+1} - z_k) / (sigma_{k+1} - sigma_k)
x0_k   = z_k - sigma_k v_k                       teacher clean-latent estimate (stop-gradient)
x0_hat = z_{k-d} - sigma_{k-d} v_theta(z_{k-d}, c),   d ~ U{1,2,3}
loss   = mean_k  sqrt(||x0_hat - x0_k||^2 + c^2) - c,   c = 0.00054 sqrt(D)
```
The student makes one conditional forward, so guidance is absorbed; sample it with cfg 1.

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

- **Selection rule.** Exact Boltzmann weighting of all four candidates ties argmax at T=0.04 and
  T=0.08 (+0.0204 / +0.0190 vs +0.0203 over random, 3k pool, three seeds) at 2.3x the compute;
  uniform weighting of all four is worth only +0.006, so the gain is the weighting, not the extra
  trajectories. Sampling one candidate per visit from the same weights loses 0.014 to the exact
  objective, and a uniform redraw per visit is 0.008 *below* a fixed random draw: target churn,
  not the softer tilt, is what soft selection pays for. Argmax is the recipe.
- **Batch size.** Accumulating to 16 captions per update at the same data budget leaves the gap
  unchanged (+0.0221 vs +0.0223, seed 0) and lifts both arms by +0.005.
- **Candidates.** Eight candidates instead of four: +0.001 (null), although the offline headroom
  grows by 16%.
- **Teacher.** With a 28-step or a 16-step cfg-4.5 teacher the within-caption score spread halves
  and the trained gain vanishes: this is a fixed-teacher gain, not a substitute for teacher quality.
- **Not adopted.** Regressing onto the reference photograph (lambda 0.2) costs 0.02-0.03 CompBench;
  an EMA-of-student teacher collapses at decay 0.999 (its online selection entropy rising to 0.93
  is the early warning) and is inert at 0.9999.

## Logging

With `WANDB_PROJECT` set, each run logs per step: loss, the loss at every supervised trajectory
state (`train/loss_k*`), gradient norm, learning rate, epoch, samples seen, steps/s, peak GPU
memory, and `sel/gain_<score>`: the running mean of (score of the selected candidate − mean over
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
