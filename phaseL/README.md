# Phase L — publication-scale preference + REPA

Phase L promotes the strongest completed Phase-I recipe: explicit top-versus-bottom
preference training with the audited REPA branch. The large run is gated by the held-out
correct-prompt versus counterfactual-label control in job `100593`; training must not be
submitted until that control supports text-specific transfer.

The locked promotion rule is: the paired CompBench interval must lie above zero, the
GenEval2 effect must point in the same direction, and matched fidelity plus both diversity
checks must be non-inferior. The gate is logged as its own immutable W&B run. Submitted
training jobs exit cleanly before model loading when the gate is negative.

## Locked scale-up

- Training prompts: the disjoint T2I-CompBench training split already audited in Phase C.
- Preference data: four independent frozen-teacher seed pools per prompt, four candidates
  per pool, eight teacher steps, CFG 7, and VQAScore endpoint orientation.
- Size: 5,559 source prompts × 4 seed pools = 22,236 preference records. Repeat zero is
  copied bit-for-bit from the existing cache; three fresh pools are generated and scored.
- Student: SD3.5-M M1 four-step checkpoint, explicit relative-error preference loss,
  M1 anchor weight 1.0, REPA weight 0.5, 12,000 updates, learning rate 1e-6.
- Replication: three training seeds. Since the data is four times larger than the pilot,
  12,000 updates preserve approximately the pilot's number of dataset passes.
- Logging: immutable scheduler-ID paths and W&B IDs, full loss and gradient telemetry,
  fixed-seed generations every 500 updates, scored teacher-pair audit images, code/data
  hashes, and atomic checkpoints every 3,000 updates.

## Publication evaluation

Every seed is evaluated at four steps with paired generation seeds on the full held-out
T2I-CompBench validation split and GenEval2. Fidelity uses 5,000 COCO captions against
the corresponding COCO real-image distribution. Diversity uses eight samples for each
of 400 stratified held-out prompts. Report per-seed effects, prompt-bootstrap intervals,
and the across-training-seed mean and range. VQAScore remains an in-objective diagnostic,
not a primary evaluation metric.

The Phase-I pilot checkpoint and M1 are frozen baselines. No test result selects a
checkpoint, training duration, or hyperparameter. A blinded human pairwise set will be
exported from the held-out samples, but no external annotation is initiated automatically.

The automated publication verdict requires every one of the three training seeds to
improve over the frozen Phase-I preference+REPA pilot on both alignment benchmarks, while
each seed stays within the locked CMMD and diversity tolerances. Prompt-bootstrap intervals
and across-training-seed variation are reported separately rather than conflated.
