# PSO and D3PO reference-implementation audit

**Audit date:** 2026-08-07

**Local clones:**

- `third_party/Pairwise_Sample_Optimization`, commit `85e09df883a93782b64379097489abd9ba85d50a`
- `third_party/D3PO`, commit `40d31758048f62ff49814eaa0403d820493ad374`

## Bottom line

**PSO is required as the closest baseline and intellectual parent. D3PO is required in the genealogy, but not
for the first corrective experiment.** Neither repository is a drop-in implementation for our SD\(3.5\)-M
rectified-flow student. The PSO clone should remain a read-only reference because the repository has no license
file. D3PO is MIT licensed, but its SD\(1.x\)/DDIM-style stochastic trajectory machinery would still require a
substantial and scientifically meaningful port.

## What the repositories actually implement

| Dimension | D3PO | PSO repository | Our Phase-I implementation |
|---|---|---|---|
| Role | Direct preference optimization of diffusion trajectories from pairwise feedback | Pairwise Sample Optimization; repository explicitly builds on D3PO | Offline preference tuning of an already distilled rectified-flow student |
| Data | Two on-policy trajectories, compared by a reward or human label | Two on-policy trajectories; bundled SDXL-DMD2 and SDXL-Turbo scripts | Cached frozen-teacher top/bottom endpoints |
| Model family | Conventional stochastic diffusion | SDXL and few-step SDXL-DMD2/Turbo | SD\(3.5\)-Medium MMDiT with rectified flow |
| Update statistic | Preferred-versus-rejected log transition-probability ratio relative to a reference | Clipped current/reference transition-probability ratios in a logistic pairwise loss | Preferred-versus-rejected flow-error difference relative to frozen \(M1\) |
| Pair noise | Separate sampled trajectories | Separate sampled trajectories | Shared noising randomness for the pair |
| Preservation | Reference policy ratio | Reference ratio and clipped importance ratio | Explicit \(M1\) function-space anchor; optional REPA |
| Typical strength | Configuration-dependent | Bundled DMD2 configuration uses \(\beta=50\) and ratio clipping \(\epsilon=0.1\) | Pilot uses \(\beta=100\), not yet calibrated |

The common family resemblance is the reference-relative Bradley–Terry/DPO structure:

\[
\mathcal L
=-\log\sigma\!\left(
\beta\left[
\log\frac{p_\theta(\tau^+\mid c)}{p_0(\tau^+\mid c)}
-\log\frac{p_\theta(\tau^-\mid c)}{p_0(\tau^-\mid c)}
\right]
\right).
\]

Our flow-error statistic is a compatibility surrogate for the trajectory log-ratio, not a new preference-loss
family:

\[
\log\frac{p_\theta(u\mid z_t,c)}{p_0(u\mid z_t,c)}
\propto
-\left(e_\theta(u)-e_0(u)\right).
\]

That proportionality is exact only under an explicitly stated Gaussian transition/constant-variance model; in
the paper it must be presented as the modelling bridge and tested, not silently assumed.

## Code paths consulted

- PSO online DMD2 sampling and pair formation: `human_preference_tuning/train_online_pso_sdxl_dmd2.py`,
  especially the two sampled trajectories, reward comparison, current/reference ratios, ratio clipping, and
  logistic loss.
- PSO DMD2 hyperparameters: `human_preference_tuning/config/config_sdxl_dmd_dpo.py`.
- D3PO reward-model training path: `scripts/rm/train_d3po.py`, especially the two-trajectory sampler and
  reference-ratio pairwise objective.

## What we should use from each repository

### PSO

- Use the paper and repository to define the closest baseline, nomenclature, and expected on-policy mechanics.
- Independently implement the published equation for our model family; do not copy repository code without a
  clear license.
- Match its key experimental question: offline versus online preferences, at equal model, reward, prompts,
  update budget, and evaluation.
- Treat its \(\beta=50\) as one comparison point, not as justification for our \(\beta=100\).

### D3PO

- Cite it as the direct-reward/human-feedback trajectory-preference ancestor on which PSO builds.
- Use its implementation as a sanity reference for preference sign, reference ratios, and paired trajectory
  bookkeeping.
- Do not port it before the fixed-pair causal correction. An online D3PO/PSO-style control is a later baseline,
  not the blocker for the next decisive run.

## Required comparison in the eventual paper

| Comparison | Scientific question |
|---|---|
| Existing Phase-I anchored error preference | Does the current rectified-flow-compatible surrogate work? |
| Fixed-pair randomized and inverted labels | Is the effect caused by preference orientation? |
| Offline PSO/Diffusion-DPO equation | Is any gain specific to our surrogate or standard preference optimization? |
| Online PSO-style control | Does on-policy sampling remove cached-teacher covariate shift? |
| Shared versus independent pair noise | Is shared noise a real variance-reduction contribution? |

The first three are mandatory. The online control is highly valuable but should be attempted only after the
fixed-pair pilot establishes a prompt-specific effect worth scaling.
