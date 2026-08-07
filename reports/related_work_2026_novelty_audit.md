# Related-work audit (July 2026) — where our results still have a novelty margin

> **Major correction, 2026-08-07 — PSO was the missing closest work.** Pairwise Sample Optimization
> ([OpenReview](https://openreview.net/forum?id=fXnE4gB64o)) directly derives offline and online pairwise
> preference objectives for diffusion models and includes few-step SDXL-DMD2/Turbo implementations. D3PO
> ([arXiv 2311.13231](https://arxiv.org/abs/2311.13231), CVPR 2024) is its direct trajectory-preference ancestor.
> Consequently, Phase I is **not a new preference objective**. It is an offline, cached-teacher,
> shared-noise, anchored rectified-flow compatibility instance of the PSO/Diffusion-DPO family. The surviving
> contribution is the controlled measurement: input-channel selection versus explicit objective-channel
> preference, fixed-pair prompt counterfactual identification, and the alignment–fidelity–diversity trade-off.

> **Second correction, 2026-08-07 — mechanism and theory.** The controlled **measurement** survives; the
> previously asserted conditional-mean, optimum-invariance, reward-curvature, target-variance, and
> dose/learnability mechanisms do not. Shared-noise high-\(\sigma\) input coalescence is now a hypothesis with
> an explicit shared-versus-independent and \(\sigma\)-band falsification test. BOND does not prove a
> forward-KL impossibility. ArXiv:2605.11361 separates a KL exponential tilt from a Wasserstein proximal
> transport, so its deterministic proximal map is not an exact KL-tilt representation theorem. Wherever the
> historical chronology below disagrees, use `reports/project_handoff_2026-08-07.md`.

## 0. Closest preference-optimization lineage

| work | mechanism | relevance to this project |
|---|---|---|
| **Diffusion-DPO** (CVPR 2024) | Recasts denoising as preference optimization relative to a frozen reference | Establishes the basic reference-relative pairwise family |
| **D3PO** (CVPR 2024) | Optimizes paired stochastic diffusion trajectories directly from reward or human preference | Direct ancestor for online trajectory preference; PSO code builds on it |
| **Pairwise Sample Optimization** (ICLR 2025) | Offline and online pairwise sample optimization, including few-step distilled models | **Closest work and mandatory baseline**; removes method-level novelty from our loss |
| **Our Phase I** | Cached teacher pairs, shared noising, flow-error log-ratio surrogate, explicit \(M1\) anchor, optional REPA | Compatibility engineering and experimental instrument; novelty must be mechanistic |

The repository audit is recorded in
[`pso_d3po_reference_audit.md`](pso_d3po_reference_audit.md). The PSO repository has no license file in the
audited commit, so it is used as a read-only reference; any rectified-flow implementation must be written
independently from the published equations.

**Purpose.** Before committing GPU-months to Phase E, check the 2025–26 literature on reward + few-step
distillation. **Headline: the field moved fast and Phase E as specified is largely pre-empted; the C1 negative
result is not.** This document records what exists, what it costs us, and what is left.

---

## 1. Direct competitors to Phase E (reward-gradient on a few-step model)

| work | what it does | overlap with Phase E |
|---|---|---|
| **RTDMD** — *Reinforcing Few-step Generators via Reward-Tilted Distribution Matching* ([2605.26108](https://arxiv.org/abs/2605.26108), code: [Harahan/RTDMD](https://github.com/Harahan/RTDMD)) | Two-stage: AC-DMD distillation, then reward RL. **Hybrid gradient: GRPO on stochastic steps + *direct reward backprop through the deterministic final step*.** KL to a *reward-tilted teacher* = distribution-matching + reward terms. **SD3-M / SD3.5-M / FLUX.2-4B at 4 NFE.** Evaluated on CLIPScore, PickScore, HPSv2, ImageReward, **GenEval, GenEval2**. SOTA; distilled FLUX.2-4B beats the 9B teacher at 50 NFE on most rewards. | **Near-total.** Same base model, same NFE, same final-step-backprop scheme, same benchmarks. Phase E ≈ RTDMD stage 2 minus GRPO. |
| **LaSRO** — *Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward* ([2411.15247](https://arxiv.org/abs/2411.15247)) | Reward-fine-tunes an **already-distilled ≤2-step** model; learns **latent-space surrogate rewards** to make arbitrary (non-differentiable) rewards differentiable; off-policy; beats DDPO / Diffusion-DPO. | High — and it **also subsumes our planned B3 "latent verifier"**, which was to be the deployable contribution. |
| **FAV** — *Aligning Few-Step Generative Models by Amortizing Sample-based Variational Inference* ([2605.26552](https://arxiv.org/abs/2605.26552)) | Alignment as sampling from a **reward-tilted distribution anchored to a reference**; SVGD particles **amortized into generator parameters** by fixed-point regression. Needs only sample access. Scales to 1024² T2I. | High on **framing** — "amortizing" alignment into a few-step generator is their title claim. |
| Others in the same lane | **TDM-R1** ([2603.07700](https://arxiv.org/pdf/2603.07700)), **ReDiF** ([2512.22802](https://arxiv.org/pdf/2512.22802)), **AdvDMD** ([2604.28126](https://arxiv.org/pdf/2604.28126)), **DMDR/DMD-meets-RL** ([2511.13649](https://arxiv.org/pdf/2511.13649)), **Rewarded Moment Matching Distillation** ([2606.30414](https://arxiv.org/pdf/2606.30414)), **Drifting Preference Optimization for One-Step** ([2606.02521](https://arxiv.org/pdf/2606.02521)), **DOLLAR** (video, [2412.15689](https://arxiv.org/html/2412.15689v2)), **Dense Reward Difference Learning** ([2411.11727](https://arxiv.org/pdf/2411.11727)), **RG-LCD**, **RLCM** ([2404.03673](https://arxiv.org/html/2404.03673v1)), **ShortFT** ([2507.22604](https://arxiv.org/pdf/2507.22604)) | The "reward + few-step distillation" niche is **densely occupied**. |
| **Design space of reward backprop for flow matching** ([2606.11075](https://arxiv.org/pdf/2606.11075)) | An entire paper systematically ablating *our* design choices (truncation depth, LoRA, regularization). | Our guard/hyperparameter study is pre-empted as a contribution. |

**Consequence.** Phase E cannot be sold as a new method. Our exact recipe (init from a distilled few-step
SD3.5-M, LoRA, backprop through the last step, anchor to the reference) is **RTDMD's stage 2**, published in
May 2026 with stronger machinery and better results.

## 2. Our D1 "tension" finding is real — and already documented

D1 showed reward-target flow-matching from base destroys few-step capability (VQA rising with steps
0.66→0.75→0.77). Independent confirmation:

> **D-OPSD** — *On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models*
> ([2605.05204](https://arxiv.org/html/2605.05204v1)): "Standard supervised fine-tuning often compromises the
> model's original distilled few-step ability to generate" — because flow-matching supervises **externally
> induced states of an offline target** rather than the states the few-step sampler actually visits.

Good news: our diagnosis was **correct** and independently corroborated. Bad news: it is **known**, so the
tension is not a novel finding. (It does validate our decision to abandon the D2/D1p line.)

## 3. What is still ours — the C1 negative result

Searches for a published test of *"does best-of-N / reward-selected trajectory data amortize into the student
through consistency distillation?"* returned **nothing**. The nearest neighbours:

- **Reward-Guided Trajectory Distillation for video** ([OpenReview N5RV691l3H](https://openreview.net/forum?id=N5RV691l3H))
  uses a reward model to "mitigate redundant data points" in trajectory distillation and reports a *positive*
  effect — for **video**, with a different objective, and without an isolated selection-vs-random control.
- Everyone else injects reward by **gradient or policy-gradient**, never by pure data selection.

The later audit found positive reward-selection work in GORS, GRAFT/P-GRAFT, Diffusion-Sharpening, and related
target-channel settings. Our narrower asset is the controlled selection-versus-random measurement inside the
audited consistency/trajectory-distillation objective:

\[
\Delta_{\mathrm{CompBench}}=+0.0004\;[-0.0090,+0.0097],
\qquad
T\approx0.03.
\]

This is a useful objective- and setting-specific negative result. It does not by itself explain why the field
needs reward gradients and it is not licensed as a universal structural theorem.

## 4. Honest assessment of the options

**A. Analysis / negative-results paper (recommended).** "When does best-of-N alignment amortize into a few-step
generator?" — a pre-registered empirical map: **selected-trajectory consistency transfer is unresolved at
approximately \(\pm0.009\)** (C1, ours, no established mechanism) →
**target replacement breaks few-step capability** (D1, ours, corroborating D-OPSD) → **gradient works**
(E, reproducing RTDMD/LaSRO on our harness as the positive control). Strengths: genuinely unclaimed central
result; unusually rigorous methodology (locked pre-registration, deviations, validated official evaluators,
power analysis, fidelity + diversity gates). Weakness: not a new method — realistic venue is an empirical-study
/ negative-results track, or a solid workshop, rather than a headline ICLR/NeurIPS method slot.

**B. Find a differentiating angle for Phase E.** The remaining gaps are narrow: competitors optimize
aesthetic/preference rewards (HPSv2, PickScore, ImageReward), while our stack is aimed at **compositional**
alignment (VQAScore selector, T2I-CompBench numeracy/spatial/binding, GenEval2). A "reward-tuning for
compositionality in few-step models, and its ceiling" study — using our unique **oracle-headroom (Exp-0)** and
**trajectory-coherence (Exp-0.5)** assets plus the **amortization-efficiency η** framing — is not obviously
occupied. But it is a thinner margin and needs RTDMD as a baseline we must reproduce.

**C. Stop.** The infrastructure and the C1 result exist; write them up and move on.

**Recommendation: A, with the Phase-E run kept small** — one E1 + one λ_kl=0 ablation, framed as the positive
control that completes the map, *not* as the contribution. Do **not** build a full method stack to compete with
RTDMD on its own turf; we would be behind on machinery (GRPO hybrid, AC-DMD) and compute.

---
*Audit date 2026-07-26. Sources are the arXiv/OpenReview links inline. Note several competitors are 2026
preprints; they still count as prior art for novelty. Related pre-registrations:
[phaseC1_results_report.md](phaseC1_results_report.md), [phaseD_reward_target_preregistration.md](phaseD_reward_target_preregistration.md),
[phaseE_reward_gradient_preregistration.md](phaseE_reward_gradient_preregistration.md).*

---

## 5. CORRECTION (2026-07-26) — four significant misses, and the reframe they force

An external expert review found four papers this audit missed. All four were verified. One of them changes the
interpretation of our central result, for the better.

### 5.1 GRAFT / P-GRAFT — the important miss
**"Fine-Tuning Diffusion Models via Intermediate Distribution Shaping"**, Anil, Haque, Kannen, Nagaraj,
Shakkottai, Shanmugam (Google DeepMind), [arXiv 2510.02692](https://arxiv.org/abs/2510.02692). Verified from
the paper:

- reward = **VQAScore** (ours)
- train on **T2I-CompBench++ train**, evaluate on **val + GenEval** (ours)
- selection = **top-1 of 4** on CompBench (literally our N=4 argmax)
- training = standard SFT **on the accepted clean sample**
- T2I-Val VQAScore: base SDv2 **69.20** → GRAFT **75.69** → P-GRAFT(0.25N) **76.12**

Same reward, same prompts, same benchmark, same selection rule — and a large **positive** where we measured
+0.0004. This is **not** a refutation of Phase C1; it is the missing positive control, and the difference is
exactly the mechanism in §6.4 of the master story:

| | what is selected | what the loss regresses onto | result |
|---|---|---|---|
| **GRAFT** | a clean sample | **the accepted sample itself** → selection enters the **target** | large positive |
| **our C1** | a trajectory | a deterministic function **h(student input)** → selection enters the **input** | null |

P-GRAFT sharpens it further: it deliberately relocates the tilt to intermediate noise levels where the
conditional variance of the reward given the latent is favourable, and its Appendix D.3.1 works the limiting
cases — the case where the reward is independent of the latent being selected on yields **no tilt at all**.
Our C1 sits at the opposite extreme (deterministic ODE ⇒ the reward is a *function* of z_T), and the tilt
cancels for a different reason: the accepted object is the *input* and the target is downstream of it. That
last step is ours and is not in their paper.

### 5.2 Noise Hypernetworks — our framing, already published
**Eyring, Karthik, Dosovitskiy, Ruiz, Akata**, [arXiv 2508.09968](https://arxiv.org/abs/2508.09968),
NeurIPS 2025. Amortizes reward-guided *test-time noise optimization* into a hypernetwork that modulates the
initial noise, keeping the base generator frozen, explicitly motivated by the KL term being intractable for
**step-distilled** models. This pre-empts "amortizing test-time compute into a few-step generator" as a framing
more directly than FAV, and is the **constructive complement** to our negative: for a distilled generator the
reward-tilted distribution is put in the **noise**, not in the map.

### 5.3 Diffusion-Sharpening — reward-selected trajectories, positive
**Tian, Yang, Zhang, Tong, Wang, Cui**, [arXiv 2502.12146](https://arxiv.org/abs/2502.12146). Path-integral
selection of training trajectories by reward, on T2I-CompBench, explicitly framed as amortizing inference cost.
Their ablations are useful to us: candidate count n = 1→8 moves CLIP Score only 0.334→0.338, while trajectory
length m = 1→3 moves it 0.322→0.338 — i.e. **almost all their gain comes from trajectory-level supervision, not
from selection breadth**, which is independent evidence that the pure selection channel is weak, from a paper
reporting selection as a positive. Their target is a real dataset image with known ε, so again the target is
not a deterministic function of the input.

### 5.4 DRM — a direct problem for the Phase F counting claim
[arXiv 2605.25661](https://arxiv.org/abs/2605.25661), CVPR 2026, built on an **SD3.5-M** DiT backbone with
Step-wise GRPO. A reward model trained on human preference data reportedly produces large gains on objective
compositional axes including counting. Different setting (multi-step, step-wise dense reward), but it
contradicts *"preference-reward tuning does not buy counting"* as a **general** claim. Phase F's headline is
narrowed accordingly (see `11_phaseF_rtdmd_reward_effect.md`).

### 5.5 Also added
DMSampler (ICML 2026, reward-aware distillation reusing high-reward trajectories), Reward Score Matching
([2604.17415](https://arxiv.org/abs/2604.17415), unifies reward fine-tuning as score matching against a
value-guided target — general-form scaffolding for our "reward must enter the target" corollary),
LOOP ([2503.00897](https://arxiv.org/abs/2503.00897)), and **BOND: Best-of-N Distillation**
([2407.14622](https://arxiv.org/abs/2407.14622)) as the LLM analogue.

### 5.6 The reframe this forces — reward channels

The audit's original framing ("who pre-empted us") was the wrong axis. The right one:

| channel | does reward move the student? | representative work |
|---|---|---|
| **noise** | yes | Noise Hypernetworks (and it is where the field puts the tilt for *distilled* models) |
| **target** | yes | GRAFT / P-GRAFT, Diffusion-Sharpening |
| **input** (selection under a deterministic teacher) | no measurable independent transfer under the audited consistency objective | **this work (Phase C1)** |
| **gradient** | yes, but **compositionally lopsided** | RTDMD, LaSRO; lopsidedness measured by **this work (Phase F)** |

"Nobody tested selection" is now **false** — GRAFT and Diffusion-Sharpening tested it and it worked. What
remains unclaimed is a controlled selection-vs-random ablation **inside a consistency/trajectory-distillation
objective**, where the target is a deterministic function of the input. Our claim becomes conditional, which is
a better and more falsifiable claim than the one we started with.


---

## 6. ROUND-5 CORRECTION (2026-07-26) — three more misses, and the positioning demotion they force

### 6.1 GORS — in the benchmark paper we build on
**T2I-CompBench** ([2307.06350](https://arxiv.org/abs/2307.06350), NeurIPS 2023) introduces **GORS**
(*Generative mOdel fine-tuning with Reward-driven Sample selection*): fine-tune SD on self-generated images that
score highly against compositional prompts, loss weighted by the alignment score, samples above a threshold
selected. It includes a **selection-strength ablation** (lowering the threshold, and removing selection
entirely, both degrade performance).

That is a monotone selection dose-response **in the target channel, published in 2023, inside the benchmark
paper this project uses throughout.** Missing it was a genuine failure of the audit. With GRAFT and
Diffusion-Sharpening this makes **three independent positives for selection when it enters through the target**.
Our nulls are specifically about the *distillation* objective, and that contrast is the contribution — but this
cannot be written without GORS in it.

### 6.2 The obstruction is already asserted in the literature
- **ZeNO** ([2605.11347](https://arxiv.org/abs/2605.11347)) opens by noting that existing reward-alignment
  methods rely on multi-step **stochastic** trajectories and are therefore hard to extend to **deterministic**
  generators, naming consistency models and distilled diffusion.
- **Didr** ([2605.24001](https://arxiv.org/abs/2605.24001)) names **terminal reward domination**: endpoint-only
  objectives let the optimiser exploit stochastic degrees of freedom, improving reward at the cost of fidelity;
  its fix propagates the reward-tilted clean-image density across **all** noise levels. This is a **competing
  published explanation** for Phase G's underperformance — and it is *not* our variance identity.

**Consequence for positioning:** "reward cannot enter through supervised selection" is closer to received
wisdom in this literature than to a thesis. What remains genuinely unclaimed is the **controlled measurement**
— matched arms, measured and order-statistic-verified doses, a negative control, and a ceiling — plus the
**dose/learnability identity** (§6.4). Position it that way, or a referee will.

### 6.3 The conditional-averaging ceiling is documented qualitatively
Blur under a fitted map is a known failure with known workarounds: Self-Corrected Flow Distillation
([2412.16906](https://arxiv.org/abs/2412.16906)) on blurry one-step generation, Continuous-Time Distribution
Matching ([2605.06376](https://arxiv.org/abs/2605.06376)) on oversmoothing from missing dense supervision and
mode-seeking reverse KL, Distilled Decoding 2 ([2510.21003](https://arxiv.org/abs/2510.21003)) on fixed
noise-to-data mappings being inherently hard to learn. Our ceiling number is the **quantitative** version —
frame it as measurement, not discovery. Correlated branches likewise have a name in the search literature
(TReASURe / UnMaskFork [2602.04344](https://arxiv.org/abs/2602.04344); DTS
[2506.20701](https://arxiv.org/abs/2506.20701)), which is our σ-collapse.

### 6.4 The theory: KL tilting and Wasserstein proximal transport must remain separate
**"The tractability landscape of diffusion alignment"** ([2605.11361](https://arxiv.org/abs/2605.11361)).
The paper presents a KL exponential tilt and a separate Wasserstein proximal transport. The deterministic
proximal-argmax pushforward belongs to the Wasserstein construction. It cannot be used as an exact
representation theorem for the KL-tilted distribution, and it supplies no theorem about learning either law
by consistency regression. We therefore make neither a representational-impossibility claim nor the older
proximal-argmax-versus-central-tendency claim.

### 6.5 Phase E is dead
**TDM-R1** ([2603.07700](https://arxiv.org/abs/2603.07700)) covers non-differentiable rewards for few-step;
**LaSRO** already derives a BLIP-VQA **attribute-binding** reward from T2I-CompBench for a 2-step distilled
model; **MFM** ([2601.14430](https://arxiv.org/abs/2601.14430)) amortises inference-time steering into a
few-step model positively; **MIRA** ([2510.01549](https://arxiv.org/abs/2510.01549)) argues from inside the
noise-channel literature that best-of-N selection is *dominated* by reward-gradient noise search. Retired
(Amendment 2 in the Phase-E pre-registration).

### 6.6 What remains after all corrections
1. **A controlled oracle-to-amortization negative measurement:** substantial teacher best-of-four headroom,
   but no resolved independent transfer through the audited selected-trajectory consistency recipe.
2. **A positive comparative-objective result:** a PSO-family offline baseline changes fresh samples relative to
   randomized orientation.
3. **A causal decomposition in progress:** the completed text-conditioned pair-construction policy effect plus
   the fixed-pair orientation-only experiment.
4. **An alignment–diversity question:** the DINO failure, calibrated over \(\beta\) and referenced to the
   oracle's own empirical diversity price.
5. **Phase F's reward-composition dissociation** on released RTDMD checkpoints.

The paper should be written around these measurements. Shared-noise coalescence becomes a mechanism only if
its preregistered falsification tests pass.
