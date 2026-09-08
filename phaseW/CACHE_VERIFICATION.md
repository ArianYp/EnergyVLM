# Verification of the 118k cache and the `CD_dinop_hard` training path

Done 2026-09-02 while the cache was building (jobs 128542/128543 at 8 GPUs, 128548-128551 at
4 GPUs, 32 independent ranks). Everything below was checked against the running artifacts, not
against how the code reads.

## 1. Cache record invariants (15,520 records at the time, all 32 rank files present)

| check | result |
|---|---|
| duplicate `idx` across rank files | 0 |
| `seed_base == idx * 1000` (what `train_pilot` re-rolls from) | 15,520 / 15,520 |
| required keys (`dino_patch_cos`, `endpoint_vqa`, `oracle_idx`, `random_idx`, `seed_base`, `reference`, `N`) | all present |
| `dino_patch_cos` length 4, no NaN | all |
| `dino_patch_cos_argmax_idx == argmax(dino_patch_cos)` | all |

Note on sharding: the builder shards by **position in the pool list**, not by `idx`
(`enumerate(pool)`, `i % WORLD_SIZE == RANK`). The pool is a sorted, unique but
**non-contiguous** idx set (113,948 rows, max idx 118,286: images filtered out leave gaps), so
`idx % 32 == rank` does NOT hold and is not supposed to. Partition-by-position is still a
partition, and the resume logic keys on each rank's own output, so this is fine. Recorded because
the first version of the check assumed the wrong invariant and reported 14,606 "violations".

Re-cutting the two pending 8-GPU shards as four 4-GPU jobs is safe for the same reason: with
`NSHARD=8, GPUS=4` the WORLD is still 32 and shards 4-7 cover ranks 16-31 exactly, and the ranks
run no collectives.

## 2. Sampler equivalence, static

`phaseC/train_pilot.py:rollout_states` and `exp0/generate_candidates.py:euler_cfg_sample` are the
same computation line for line: `scheduler.set_timesteps(K)`, the same sigma grid, the same CFG
composition `v_u + w (v_c - v_u)`, the same per-step bf16 cast under autocast, the same scheduler
(shift 3.0, asserted by the trainer). The one difference is batch shape: the builder rolls the
four candidates in one batch, the trainer rolls one. That changes bf16 kernel tiling and is the
source of the ~1e-3 drift measured below.

The decode paths agree: `train_self_distill.vae_decode` and `generate_candidates.decode_and_save`
apply the same `latents / scaling_factor + shift_factor`, and the PNG quantisation in
`torchvision.save_image` (`*255 + 0.5`, clamp, uint8) is reproduced in the test.

## 3. Pairing, end to end (job 128552, `phaseW/verify_cache_pairing.py`)

For 12 records drawn from ranks 0, 9, 18, 27 (both job shapes), the test re-rolls each of the 4
candidates with the trainer's own `rollout_states` from `manual_seed(seed_base + j)`, decodes with
the trainer's `vae_decode`, and re-scores DINOv2-base mean-patch and CLS cosine against the
reference photo, then compares with what the cache stored.

| channel | max abs diff | mean abs diff | argmax agree (all) | agree among clear margins | near-ties | shifted control |
|---|---|---|---|---|---|---|
| mean-patch | 2.2e-03 | 3.6e-04 | 11/12 | **10/10 = 100%** | 2 | 8.3% (chance 25%) |
| CLS | 3.2e-03 | 4.1e-04 | 12/12 | **9/9 = 100%** | 3 | 8.3% |

The one all-records disagreement is idx 224: cached top-2 margin **0.0013** (0.5109 vs 0.5096)
against a recompute drift of 1.8e-3. That is a hard argmax over two candidates that are equal to
within the noise, not a pairing fault: a pairing fault scrambles all four values (the shifted
control shows what that looks like), it does not swap two that are 1e-3 apart. The pass criterion
in the script was corrected to "100% agreement among records whose cached top-2 margin exceeds
3x the measured drift"; the original all-records criterion was wrong for near-ties.

**Verdict: the trainer re-rolls the candidates the cache scored. PAIRING_RESULT PASS.**

Practical consequence worth knowing: a few percent of captions have a top-2 margin below ~0.01,
where the `_hard` label is decided by rounding. Harmless for training (the two candidates are
near-equal under the scorer), but it caps how "hard" a hard selection really is.

## 4. Trainer path for `CD_dinop_hard`

* `selection_weights`: `variant.split("_")[1] == "dinop"` -> `dino_patch_cos`; `_hard` -> one-hot
  argmax; `pick_index` returns the argmax deterministically. Raises `KeyError` if the field is
  missing, so a cache without the score cannot silently train.
* `is_consistency = variant in ("B2","B4") or variant.startswith("CD_")` -> the x0 pseudo-Huber
  consistency branch, same as B2/B4.
* Multi-GPU: `train_pilot.py` has run under `torchrun --nproc-per-node=4` before (phaseC B2/B4/B5
  launchers), so the `DistributedSampler` / rank-0 checkpoint / wandb path is exercised. 8 GPUs is
  the same path with a different world size.
* `ablations/phaseW_train118k.lsf`: `num_steps = round(epochs * n_records / world)`; LR 2.8e-5
  (sqrt(8) scaling for the 8x effective batch), identical across the three arms so the contrast
  stays matched. Host RAM: the 1-GPU run peaked at 9.8 GB, so 450 GB for 8 ranks is comfortable.
  Disk: a run with intermediate checkpoints is ~28 GB at 6k steps; 3 arms with `--save_every 5000`
  over 28.5k steps is roughly 3 x 60 GB against 3.0 PB free.

## 5. Not verified here

* The VQAScore channel (`endpoint_vqa` / `oracle_idx`) was not re-run in the pairing test; the
  candidates are shown to be the same images, and VQAScore is a deterministic function of the
  image and prompt, so it inherits the pairing, but its numerical drift was not measured.
* The 8-GPU DDP configuration itself has not been run yet in this session; the first minutes of
  the training logs should be checked for the `[schedule]` / `[window]` / `[pairing]` diagnostics
  and a sane loss before leaving it unattended.
