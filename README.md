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

## Layout

```
common/      sampling.py (teacher rollout, decode)  distributed.py (torchrun setup)
data/        build_pool.py       training captions paired with their photographs
             build_eval_pool.py  T2I-CompBench, GenEval2 and COCO-val prompt pools
             build_candidates.py the candidate cache: 4 trajectories per caption, scored
train/       distill.py          the trainer (--selector random | dino_patch)
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
```

Evaluate students with `CFG=1.0`. Sampling a student with guidance applies it twice and roughly
halves its scores.

GenEval2's judge (Qwen3-VL) needs `transformers >= 4.57`; if the training environment pins an
older version, point `GENEVAL2_PYTHON` at a second interpreter that has it.

Neither arm needs VQAScore. `data/build_candidates.py --vqa` additionally records
`endpoint_vqa` / `oracle_idx` per caption (for a VQAScore-selected arm) and requires
`third_party/t2v_metrics` to be importable in the environment.
