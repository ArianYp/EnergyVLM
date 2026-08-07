# Architecture slide assets

This directory contains the small, curated image set required to render
`reports/architecture.html` from a clean checkout. It intentionally excludes full
generation pools, checkpoints, metric caches, and W&B runtime files.

## Provenance

- `compbench_*` and `geneval2_*`: fixed-index held-out examples from Phase-I evaluation
  job `99955`. They illustrate the placebo-versus-preference comparison; the aggregate
  benchmark results, not these selected images, are the evidence.
- `preference_prompt*`: fixed-seed qualitative samples from preference-training job
  `99953` at updates 0, 1,500, and 3,000.
- `counterfactual_control_*`: one logged training-pair audit from jobs `100591` and
  `100592` at update 500. The complete intervention gate uses 5,042 prompts.

The provenance and numerical interpretations remain documented in the slide captions,
the Phase-I evaluation reports, and `phaseL/campaign_20260807T044116Z.json`.
