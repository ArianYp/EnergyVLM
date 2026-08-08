# Phase FP — fixed-unordered-pair, label-only causal pilot

**Locked 2026-08-07, before any Phase-FP training arm produced a checkpoint.**
Cache `phaseFP/fixedpair_101185` (LSF 101185). Smoke gate `phaseFP/smoke_101190` — passed.
Training arms: 101193 correct, 101194 counterfactual, 101195 random, 101196 inverted,
101197 correct-with-independent-noise. Evaluation pool 101203.

## Question

The completed Phase-I counterfactual campaign changed the labelling text, which changed *both* the
membership of the preference pair (60.75% of records) and its orientation. It therefore estimates
the total effect of the text-conditioned pair-construction policy. This pilot asks the remaining
question:

> Holding the two images fixed, does the **orientation** of the preference label change what the
> four-step student generates — and is that orientation effect specific to the prompt's composition
> rather than a generic image-quality prior?

## Design

Every arm reads the same serialized candidate indices from the same cache root. The trainer runs
with `--pair_source fixed_indices`, asserts an `idx → pair_key` manifest, and never recomputes an
extremum. The teacher is rolled out in the pair's canonical `(a_i, b_i)` order in every arm, so the
endpoint tensors are bit-identical and only the loss sign differs. Prompt, seeds, data order,
optimizer, initialization, update count, and noise draw are identical.

The smoke gate verified this empirically rather than by assertion: across 24 logged steps all four
arms produced **identical SHA-256 hashes of the teacher endpoints**, the `inverted` arm's first
preference logit was the exact negation of the other three, and the `counterfactual` arm disagreed
with `correct` on exactly the cache's reversal-flagged records.

### Arms and their dose

| arm | orientation on the fixed pair | records with a different preferred image | mean assigned margin | **dose** (correct − arm) |
|---|---|---:|---:|---:|
| `correct` | original-prompt sign | 0.000 | +0.2865 | 0.0000 |
| `counterfactual` | sign implied by `R(·, c̃)` on the same pair | 0.272 | +0.1783 | 0.1082 |
| `random` | deterministic, exactly balanced sign | 0.500 | −0.0002 | 0.2867 |
| `inverted` | opposite of the original-prompt sign | 1.000 | −0.2865 | 0.5729 |

On the 1,373-record reversal subset, `counterfactual` and `inverted` assign identical labels. The
sharp orientation-only comparison at maximum dose is therefore the full-population
`correct` versus `inverted` contrast; no additional arm is required.

## Primary analysis — monotone dose response, not a single contrast

Scaling the Phase-I preference-minus-placebo effect (+0.0459 at dose 0.287) linearly predicts a
`correct` minus `counterfactual` effect near **+0.017** at dose 0.108, against a typical paired CI
half-width of ±0.009. A single pairwise contrast is therefore underpowered by design. The
preregistered primary is the **trend across all four arms**, which uses every arm's information:

For each held-out prompt \(p\) and arm \(k\) with dose \(d_k\), let \(S_k(p)\) be the official
CompBench score. The primary statistic is the equal-category-weighted slope

\[
\hat\beta_{\text{dose}}
=\frac{\sum_k (d_k-\bar d)\,\overline{S_k}}{\sum_k (d_k-\bar d)^2},
\qquad
\overline{S_k}=\frac{1}{|C|}\sum_{c\in C}\frac{1}{|P_c|}\sum_{p\in P_c} S_k(p),
\]

over the seven primary categories \(C\). Prompts are resampled within category, paired across arms,
10,000 draws.

**The orientation hypothesis passes only if the 95% interval for \(\hat\beta_{\text{dose}}\) lies
entirely below zero.**

Co-primary, reported with the slope and not substituted for it:

1. `correct` − `inverted` (maximum-dose orientation-only contrast), 95% CI excludes zero.
2. Bootstrap probability of the full ordering
   \(S_{\text{correct}}>S_{\text{counterfactual}}>S_{\text{random}}>S_{\text{inverted}}\).
3. GenEval2 Soft-TIFA agrees in direction on the slope.

## Text-specificity — the question the pilot exists to answer

`correct` − `counterfactual` is the text-specificity contrast: same images, same student prompt,
orientation differing only where an edited text reorders the pair. It is reported with its interval
regardless of significance, and interpreted against the dose ladder rather than in isolation.

The zero-training audit already found a supporting dissociation that this pilot can confirm or
refute: against M1, the Phase-I counterfactual arm gained on the BLIP-VQA family (+0.0383) but was
null on the architecturally disjoint UniDet detection family (−0.0070), whereas the correct arm
gained on both (+0.0789, +0.0620). **Prediction:** in the fixed-pair pilot the `correct` −
`counterfactual` effect will again be positive on UniDet. If the fixed-pair effect is null overall
*and* the UniDet dissociation disappears, the Phase-I positive result is attributable to pair
mining rather than orientation, and the paper reframes accordingly.

## Evaluator families are mandatory

Every CompBench result is reported split by evaluator architecture — BLIP-VQA (colour, shape,
texture), UniDet (spatial, 3d_spatial, numeracy), 3-in-1 (complex), with CLIPScore (non_spatial)
secondary and never pooled. `audit/evaluator_family_bootstrap.py` produces paired intervals. A
result carried by BLIP-VQA alone is reported as such and does not support a general alignment claim,
because that is the family closest to the VQAScore selector that produced the labels.

Category reversal rates are highly heterogeneous (spatial 57.7%, shape 9.2%), so per-category
effects are reported alongside the pooled primary.

## Mechanism falsification (roadmap task N)

Arm 101197 is identical to `correct` except that the negative branch receives independent
corruption noise. Because the extra noise tensor is drawn *after* all other RNG consumption,
\(z^+\) is bit-identical between the shared and independent conditions and only \(z^-\) differs.

The shared-noise coalescence hypothesis predicts that competition is strongest at high \(\sigma\),
where \(z^+-z^-=(1-\sigma)(x^+-x^-)\to 0\) while the targets stay separated. It is supported only if
**both** hold:

1. shared noise outperforms independent noise on the primary endpoint; and
2. the per-\(\sigma\) telemetry (`telemetry.jsonl`, logged every step) shows the preference logit and
   gradient contribution growing with \(\sigma\).

If independent noise performs identically and the \(\sigma\) profile is flat, shared-noise
coalescence is **removed from the paper's claims**, and the paper is written without a mechanism
rather than with a speculative one. σ-band-restricted training arms are run only if the telemetry
shows a σ-dependence worth confirming.

## Fidelity and diversity

Fidelity uses the **COCO-caption** pool against COCO val2017 reals — the standard matched-caption
protocol — for every arm including a freshly generated M1. The Phase-I job-99955 fidelity numbers
were generated from CompBench compositional prompts and are not comparable; they are not used.

Diversity uses four seeds on 400 stratified held-out prompts, DINO and LPIPS. The preregistered
comparison is **two-sided**:

- against M1, as before; and
- against the **teacher best-of-4 diversity price** measured in the same units by
  `audit/oracle_diversity_price.py` (LSF 101198).

A student that loses no more diversity than the selection policy it amortizes has not become worse
than what it imitates. A student that loses more has paid a cost the policy does not explain, and
that becomes a central reported result rather than a footnote. The analytic
\(\mathrm{KL}_4=\log 4-3/4\approx0.636\) nats is in different units and is not compared to either.

## β is not swept in this pilot

The audit found β = 100 places roughly half of all steps in the inert |logit| < 0.5 region, and that
gradient clipping fires on **100%** of steps, so β changes the direction mix between the preference
and anchor gradients rather than the update magnitude. All four arms therefore share β = 100 to keep
the causal comparison clean. A calibrated sweep (β ∈ {0, 100, 175, 350}, with β = 0 the exact
M1-anchored null) runs only after the orientation question is settled, and will report clipping rate
alongside every result.

## What each outcome means

| outcome | conclusion |
|---|---|
| slope < 0, ordering holds, UniDet agrees | orientation is causal and prompt-specific; strongest paper |
| slope < 0 but `correct` ≈ `counterfactual` | orientation is causal but not demonstrably text-specific; report as an orientation effect |
| slope null, Phase-I effect stands | the Phase-I result is an effect of text-conditioned pair **mining**, not orientation; reframe |
| slope > 0 | the objective is not doing what the loss states; stop and diagnose |

Diversity failing against **both** M1 and the oracle price blocks any "better generator" claim under
every outcome above, and makes the alignment–diversity Pareto failure a central result.
