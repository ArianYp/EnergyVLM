# Phase C1 — Results: the pre-registered cheap kill fired (H1 not supported)

**Status: FINAL RESULT; MECHANISM CORRECTED (2026-08-07).** All pre-registered metrics have landed. This report is written *against* the
locked pre-registration ([phaseC_eval_preregistration.md](phaseC_eval_preregistration.md), + Deviations 1–3)
and does not introduce any metric, gate, or test chosen after seeing a B2/B4 number.

> **Retrospective theory correction.** The numerical result and locked decision remain valid. The original
> conditional-mean/invariance explanation in Section 4 does not. A four-step student is misspecified relative
> to the teacher, so reweighting can move its optimum; the available theory predicts neither an invariant
> optimum nor the observed sign. The later reward-curvature and target-variance explanations were also
> retracted after controls. Use the corrected interpretation below and
> `reports/project_handoff_2026-08-07.md`.

> **TL;DR.** B4 (oracle-VQA-selected trajectory distillation) does **not** beat B2 (random-trajectory
> distillation) on either pre-registered independent primary, at any of {4, 8, 28} steps. The student *did*
> move on the selector's own metric (VQA +0.014, tight CI) but that gain **did not transfer** to independent
> evaluators (transfer ratio **T ≈ 0.03**). Per §5 this is reported as **VQA-overfit, not a real alignment
> gain → H1 not supported.** Both robustness gates (fidelity, diversity) pass, so the null is clean, not
> confounded. **Decision: do not spend the pre-specified 2× retry on this recipe — pivot** (Phase D). The null
> is an empirical limitation of this audited recipe, not a universal impossibility theorem.

---

## 1. Headline result

Independent primaries, paired over held-out CompBench-`val` prompts (disjoint from training), 10k bootstrap:

| metric | 4-step | 8-step | 28-step | gate |
|---|---|---|---|---|
| **PRIMARY 1** — CompBench 7-cat equal-weighted B4−B2 | +0.0004 [−0.0090,+0.0097] | +0.0026 [−0.0058,+0.0110] | +0.0025 [−0.0062,+0.0111] | **NULL** ×3 |
| **PRIMARY 2** — GenEval2 soft-TIFA-gm B4−B2 (n=800) | +0.0065 [−0.0109,+0.0239] | −0.0006 [−0.0189,+0.0179] | +0.0039 [−0.0136,+0.0216] | **NULL** ×3 |

Both primaries, all three step counts: the 95% CI includes 0. No per-category direction is coherent —
`shape` is even significantly *negative* at 4 steps (−0.028). B4 ≈ B2.

## 2. The student moved — but only on the selector's own metric (VQA-overfit)

The one axis on which B4 beats B2 is the **in-objective** metric, VQAScore (clip-flant5-xxl) — the very
selector B4 used to choose its training candidates:

| step | in-objective VQA B4−B2 | independent (Primary 1) | **transfer T = indep / VQA** |
|---|---|---|---|
| 4 | **+0.0142** [+0.0075,+0.0207] ✓ | +0.0004 (null) | **0.03** |
| 8 | +0.0070 [+0.0010,+0.0130] ✓ | +0.0026 (null) | 0.37* |
| 28 | +0.0093 [+0.0030,+0.0156] ✓ | +0.0025 (null) | 0.27* |

*(the independent numerator's CI spans 0, so T is statistically indistinguishable from zero at every step;
the 4-step value, where both terms are well estimated, is the honest headline: **T ≈ 0.03**.)*

This is exactly the case pre-registration **§5** named in advance:

> *"If VQAScore (secondary) shows a large B4−B2 gain but the independent primary metric does not (T ≈ 0), we
> report this as VQA-overfit, not a real gain — H1 not supported."*

The pilot absorbed ~11% of the +0.124 supervised endpoint gap **on VQA**, and ~none of it generalized. That
this is a *significant* in-objective move (not zero) is important: it **rules out** the trivial explanations
of a null — the pipeline is not broken, the pilot is not undertrained, the measurement is not underpowered.
The student demonstrably shifted toward the selector's preferences; those preferences simply do not correspond
to gains any independent evaluator can see.

## 3. This is a clean, well-powered, strong-form null

Three facts make the null credible rather than a failure-to-measure:

1. **The measurement was powered.** Primary-1 CI half-widths are ≈ ±0.009 — tight enough to have excluded 0
   had the true effect reached the low end of the pre-registered 0.01–0.04 prediction. The point estimate
   sits at ~0.00, not "positive but n.s."

2. **The training signal was delivered at full strength.** From the selection cache (5,559 prompts, N=4),
   the oracle endpoints averaged **+0.124 VQA** over random (0.874 vs 0.750), concentrated exactly where
   Exp-0 measured the headroom (oracle−random): numeracy +0.184, shape +0.147, spatial +0.143, 3d_spatial +0.116,
   complex +0.113. B4 genuinely trained on materially better trajectory endpoints. The headroom existed and
   was fed in.

3. **Both robustness gates pass** (pre-registration §5 + Deviation 2 tolerances), so no gain was hidden by a
   quality/diversity trade and no loss confounds the null:
   - **Fidelity (5k COCO-val2017, FID + CMMD):** CMMD(B4)−CMMD(B2) = **+0.06** (tolerance +2) → PASS; FID Δ −0.12.
   - **Diversity (8 seeds × 400 prompts, DINOv2 + LPIPS):** ΔDINO −0.0014 [−0.007,+0.005], ΔLPIPS +0.0010
     [−0.004,+0.006] → PASS (non-inferior on both).

A +0.124 supervised endpoint gap, delivered into the training data in the right categories, produced ~0.00
transferable student gain with no confound. That is the falsification the pre-registration was built to catch.

## 3b. Negative control (added 2026-07-26) — the gate that calibrates the null

A second training seed of the **control** arm (B2′: identical selection rule, identical data and cache; only
data order, the per-step Δ draws, and kernel nondeterminism differ — initialisation is deterministic from base
weights) was trained and evaluated on the same harness.

| comparison | Δ (7-cat equal-weighted) | 95% CI |
|---|---|---|
| **treatment** B4 − B2 | **+0.0004** | [−0.0090, +0.0097] |
| **negative control** B2′ − B2 | **+0.0007** | [−0.0084, +0.0096] |

Read as **two draws from the same null**, not as a variance estimate — one replicate cannot estimate a
variance, and +0.0007 is consistent with an underlying SD of 0.01. What it licenses is the comparison itself:

> B4 differs from B2 in **which trajectories it trains on** (a +0.124 supervision-gap perturbation). B2′
> differs from B2 only in training randomness. **The treatment comparison carries strictly more perturbation
> than the seed comparison, and produced less movement.**

**⚠ Correction this forces on our own reporting.** Per-category seed-only deltas reach **numeracy −0.0211**,
spatial +0.0147, color +0.0093 — comparable to the `shape` **−0.0284** that §1 reported as significantly
negative. **Per-category effects of ~0.02–0.03 are therefore not distinguishable from training-seed noise even
when their within-run bootstrap CI excludes zero.** The `shape` result is demoted accordingly, and any
sign-pattern argument must rest on the two well-separated components (in-objective VQA up, pooled primary flat)
rather than on a third per-category term. Only the **equal-weighted primary**, which averages seven categories
and suppresses this noise, is safe at the ±0.009 level.

**What this does and does not license.** Together with §1–§3 it licenses *the treatment produced nothing
measurable*. It does **not** license *nothing measurable was available* — that requires the ceiling measurement
(`reports/phaseG_theory_and_framing.md` §1). Conflating the two would mistake an underpowered experiment for a
structural result.

## 4. Historical mechanism — superseded

> **Do not cite the explanation in this section.** The defensible current statement is:
>
> Under this audited selected-trajectory consistency recipe, a large candidate-level intervention produced
> \(+0.0004\;[-0.0090,+0.0097]\) independent CompBench transfer, no more than the calibrated control contrast
> of \(+0.0007\;[-0.0084,+0.0096]\). This establishes an objective- and setting-specific empirical limitation,
> not population-optimum invariance.
>
> The exact fraction of teacher best-of-four headroom excluded by the interval requires a matched or joint
> bootstrap. The shortcut \(0.0097/0.0717\approx13.5\%\) is a point-denominator approximation, not yet a formal
> \(95\%\) upper bound. BOND does not prove that forward-KL imitation cannot reproduce best-of-\(N\), and the
> Wasserstein proximal map in arXiv:2605.11361 is not an exact KL-tilt representation.

The pilot's B2/B4 supervise the student with **consistency distillation**: at a noisy state `z_{k−Δ}` on a
selected teacher trajectory, the student predicts `x̂₀` and is regressed onto the teacher's Tweedie target
`x̃₀` on that trajectory ([train_pilot.py:200–222](../phaseC/train_pilot.py#L200)). Two structural facts make
this objective **near-invariant to oracle selection**:

1. **The initial noise is part of what selection picks.** B4 rolls out from `seed_base + oracle_idx`, B2 from
   `seed_base + random_idx` ([train_pilot.py:183–194](../phaseC/train_pilot.py#L183)). Both seeds draw
   `z_T ~ N(0, I)`. So B4 and B2 train on *different but equidistributed* inputs, each paired with its own
   teacher-consistent target. Selection changes **which inputs** the student sees — not the target for any
   fixed input.

2. **Consistency distillation regresses to the teacher's per-state conditional mean.** Every `(noisy state →
   x̃₀)` pair B4 selects is already on the teacher manifold. Reweighting a self-consistent map toward its
   high-scoring draws leaves the regression target — the teacher's conditional mean at a standard-marginal
   noisy state — unchanged. At inference the student draws fresh `z_T` and applies the learned
   teacher-approximating map, identically for B2 and B4.

What *does* leak (the +0.014 VQA) is the sliver of the selector's idiosyncrasy that survives conditional-mean
matching — and that sliver is, by the transfer ratio, almost entirely non-transferable. **Selecting better
teacher outputs cannot enter a conditional-mean-matching objective through the input; reward can only shift
the student by shifting the *target marginal* it regresses onto.** That is a property of the objective, not of
the budget — which is why more steps of the same recipe converge to the same place.

## 5. Decision (pre-committed, §5)

The §5 decision table offers two moves on a null: **(a)** one pre-specified 2× budget retry (12k steps), or
**(b)** pivot.

**Recommendation: (b) pivot — do not spend the 2× retry on this recipe.** The retry exists to rule out
*undertraining*; Section 3 shows the selector-specific signal was delivered and measured while the independent
estimate stayed near zero. The result does not justify more compute on an unchanged recipe. B5/B6/B3 are
**not** run under this recipe (§5) — they are re-scoped into the pivot's *target-modification* axis, under a
new pre-registration ([phaseD_reward_target_preregistration.md](phaseD_reward_target_preregistration.md)).

## 6. What this means for the paper

C1 is not a dead end — it is the **backbone and control** of a sharper contribution, and it is credible
*because* the evaluation was locked before any B4 number existed.

- **A pre-registered quantitative negative:** best-of-\(N\) *trajectory selection* does not measurably
  amortize into a few-step student under this consistency objective at the experiment's resolution
  (delivered \(+0.124\) selector-space gap, independent transfer approximately \(0\), \(T\approx0.03\)).
- **It converts into the control condition for Phase D.** Reusing the identical, validated harness, C1
  (input-selection, null) vs. Phase-D (target-modification) becomes a clean pre-registered A/B that isolates
  an explicit objective-channel intervention. The contrast is empirical; the mechanism must be tested by the
  fixed-pair and shared-versus-independent-noise ablations rather than asserted post hoc.

---

### Artifacts
- Verdicts: `phaseC/eval_verdict_s{4,8,28}.md` (+ `.json`), writer: `phaseC/eval_analyze.py`, job 85647.
- Independent scores: `phaseC/eval_scores/{model}_s{step}_{cat}/scores.json` (`phaseC/compbench_eval.py`,
  GenEval2 `phaseC/geneval2_eval.py`).
- In-objective VQA: `phaseC/eval/scores/scores_rank*.jsonl`. Fidelity: `phaseC/fidelity_report.md`.
  Diversity: `phaseC/diversity_report.md`. Selection cache: `phaseC/selection_rank*.jsonl`.
- Pre-registration + deviations: [phaseC_eval_preregistration.md](phaseC_eval_preregistration.md);
  assumptions/limitations: [phaseC_eval_assumptions_and_limitations.md](phaseC_eval_assumptions_and_limitations.md).
- Companion design doc (pre-result): [phaseC_pilot_report.md](phaseC_pilot_report.md).
