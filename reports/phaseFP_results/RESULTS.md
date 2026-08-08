# Phase FP — fixed-pair causal pilot: results

**Campaign complete 2026-08-08.** Cache 101185, smoke 101190, training 101193/101194/101232/101233/
101234, evaluation pool 101203, evaluations 101217/101257/101258/101259/101260/101261.
Analysis plan locked in `phaseFP/PREREGISTRATION.md` before any arm produced a checkpoint.

**Seeds.** Sections up to and including the first fidelity/diversity analysis report training seed
20260806 unless stated otherwise. The primary CompBench orientation result was subsequently
replicated across **three** independent training seeds (20260806 / 20260807 / 20260808); see
*Three-seed replication*. Where a single-seed and a three-seed number both exist, **cite the
three-seed mean** — the across-seed range exceeds the within-seed prompt-bootstrap width. Every
table below states its seed count.

## The design held: only the label sign differed

*(1 seed per arm for this verification; the same check passes for the seed-2 arms.)*

All four orientation arms trained on **bit-identical images**, verified after the fact rather than
asserted: across all 3,000 steps, the SHA-256 of the teacher endpoint tensors matched in every arm
on every row (`reports/phaseFP_results/identity_4arm_full.json`). `inverted` was opposed on every
row, and `counterfactual` disagreed with `correct` on exactly the cache's reversal-flagged records
(826/3,000 = 27.5%, against a cache reversal rate of 27.23%).

Image identity alone does not establish that *every* stochastic training input matched, so the
stronger check is functional. The frozen reference error
\(e_0=\lVert v_{\xi}(z,\sigma,c)-u\rVert^2\) is a deterministic function of the image, the sampled
\(\sigma\), the corruption noise and the conditioning prompt. If two arms differ only in labelling,
each step's pair \((e_0^+,e_0^-)\) must be either an exact **match** or an exact **swap** of the
`correct` arm's. Over all 3,000 steps:

| arm | match | swap | neither |
|---|---:|---:|---:|
| `correct` | 3000 | 0 | **0** |
| `counterfactual` | 2174 | 826 | **0** |
| `random` | 1541 | 1459 | **0** |
| `inverted` | 0 | 3000 | **0** |

**Zero unexplained steps in any arm.** The swap counts are exactly what the cache dictates:
`inverted` swaps every step, `counterfactual` swaps on 826 — precisely the cache-flagged reversals —
and `random` on 1459 ≈ 50%. Independently, the per-step batch order (`idx`) and the sampled
\(\sigma\) sequence are byte-identical across all four arms.

Together these establish that the images, the noise level, the corruption noise, the conditioning
text and the data order were all identical, and that **the label assignment was the only thing that
varied** — an end-to-end functional check rather than an assertion about inputs.

A reproducibility check worth recording: `M1Fixed` scores **0.49482** on the CompBench primary,
identical to the M1 value from Phase-I job 99955 — different job, different node, weeks apart. The
frozen prompt manifests really are byte-identical and the evaluation path is deterministic.

## Definition of "orientation dose"

For arm \(a\), with \(R(\cdot,c_i)\) the VQAScore under the original prompt and
\((p_{ai},n_{ai})\) the preferred/rejected candidate that arm assigns to record \(i\):

\[
m_a=\frac1n\sum_{i=1}^{n}\Bigl[R(x_{i\,p_{ai}},c_i)-R(x_{i\,n_{ai}},c_i)\Bigr],
\qquad
d_a=m_{\text{correct}}-m_a .
\]

Dose is the **mean signed VQAScore margin an arm's labels ask the model to prefer**, measured
relative to the correct arm — *not* the fraction of records flipped. That is why `counterfactual`,
which flips 27.2% of records, has dose 0.108 rather than 0.272: the flipped records are
disproportionately near-ties (mean original margin 0.199 versus 0.319 elsewhere), so flipping them
moves the mean margin less than a random 27% would. `inverted` has dose 0.573 rather than 1.0
because the scale is margin units (\(2\times0.2865\)), not a fraction.

Properties, all fixed before any outcome was observed: computed from the **cache alone**
(`phaseFP/fixedpair_101185/dose_audit.json`, LSF 101185, built before training); prompt-normalised
rather than dataset-normalised; and expressed in VQAScore margin units, which are the *label source*
rather than an evaluation metric.

**Consequence for interpretation.** The response is not linear in \(d_a\) (see the linearity check
below), so the fitted slope is an **ordered-treatment summary** — it certifies a monotone dose
response — and must not be read as a universal effect per unit dose.

## Preregistered primary: PASS

Monotone dose slope over the seven primary categories, prompts resampled within category and paired
across arms, 10,000 draws:

\[
\hat\beta_{\text{dose}} = -0.13972,\qquad 95\%\ \text{CI}\ [-0.15719,\,-0.12210],
\qquad P(\hat\beta<0)=1.000
\]

The preregistered gate was "95% interval entirely below zero". **It passes.**
Probability of the full ordering correct > counterfactual > random > inverted: **0.9254**.

| arm | orientation dose | CompBench primary | 95% CI | versus M1 |
|---|---:|---:|---|---:|
| `correct` | 0.000 | **0.55697** | [0.54599, 0.56846] | +0.06215 |
| `counterfactual` | 0.108 | 0.50913 | [0.49808, 0.52024] | +0.01431 |
| `random` | 0.287 | 0.50314 | [0.49175, 0.51460] | +0.00832 |
| `inverted` | 0.573 | 0.46556 | [0.45388, 0.47733] | **−0.02926** |
| `M1` (untrained) | — | 0.49482 | — | — |

**Inverted labels make the model worse than not training at all** (−0.02926 [−0.03700, −0.02177],
every family negative and excluding zero). This establishes that the pair orientation is **causally
consequential** — the label carries directional information that the model acts on.

> **Correction.** An earlier draft argued that "a generic image-quality prior would move nothing
> when inverted". That is invalid. If the labels encoded generic quality, inverting them would
> train *toward* lower-quality images and could depress both quality and compositional scores.
> The inverted arm therefore cannot distinguish a prompt-specific signal from a
> prompt-independent one. **Prompt specificity is established separately**, by the fixed-pair
> `correct`-versus-`counterfactual` intervention — which changes orientation using a one-atom
> edited prompt while holding the unordered image pair fixed — together with the disjoint
> evaluator families.

## Text-specificity: the question the pilot existed to answer

| contrast | dose gap | Δ | 95% CI |
|---|---:|---:|---|
| `correct` − `counterfactual` | 0.108 | **+0.04784** | [+0.03923, +0.05625] |
| `correct` − `random` | 0.287 | +0.05383 | [+0.04520, +0.06235] |
| `correct` − `inverted` | 0.573 | +0.09141 | [+0.08095, +0.10197] |

Orienting the preference by a **one-atom-edited** prompt destroys most of the benefit, even though
it changes only 27% of the labels. Since the images are identical across arms, this cannot be a
pair-mining effect. The comparative signal is specific to the prompt's composition, not a
prompt-independent quality prior.

### Every evaluator family agrees, and the disjoint one agrees most

| family | dose slope | 95% CI | `correct` − `counterfactual` | 95% CI |
|---|---:|---|---:|---|
| BLIP-VQA | −0.20075 | [−0.22691, −0.17447] | +0.04083 | [+0.03098, +0.05102] |
| **UniDet** | −0.10138 | [−0.13198, −0.07102] | **+0.06599** | [+0.04936, +0.08278] |
| 3-in-1 | −0.07167 | [−0.09046, −0.05261] | +0.01439 | [+0.00732, +0.02130] |
| CLIPScore *(secondary)* | −0.01009 | [−0.01356, −0.00661] | +0.00005 | [−0.00124, +0.00131] |

UniDet is detection-based and shares no architecture with the CLIP-FlanT5 selector that produced the
training labels. It shows the **largest** text-specificity effect of any family. The zero-training
audit predicted exactly this from the Phase-I data; the prediction held on new arms.

## GenEval2 co-primary: independent replication

800 prompts, Soft-TIFA, a different evaluator on a different prompt set from CompBench.

\[
\hat\beta_{\text{dose}} = -0.14317,\qquad [-0.17532,\,-0.11140],\qquad P(\hat\beta<0)=1.000
\]

Ordering probability **0.9840**.

| arm | GenEval2 | `correct` − arm | 95% CI |
|---|---:|---:|---|
| `correct` | 0.26605 | — | — |
| `counterfactual` | 0.22589 | +0.04015 | [+0.02247, +0.05793] |
| `random` | 0.20680 | +0.05925 | [+0.04050, +0.07806] |
| `inverted` | 0.17651 | +0.08953 | [+0.07053, +0.10915] |
| `M1` | 0.21294 | — | — |
| `indepnoise` | 0.25553 | +0.01052 | — |

The slope magnitude (−0.143 versus −0.140), the ordering, and all three contrasts agree with
CompBench to within their intervals. Two benchmarks with different evaluators, different prompts and
different scoring architectures give the same answer.

One difference worth noting: on GenEval2 the `random` arm falls **below** M1 (0.20680 versus
0.21294), whereas on CompBench it sat slightly above. So the sign of the near-null random effect is
not stable across benchmarks, which reinforces reporting it as noise around zero rather than as a
real gain.

## Three things that complicate the story, stated plainly

**1. The random arm is not a clean null — and that gives a double dissociation.** `random` − `M1` is
+0.00832 [+0.00109, +0.01567], carried by BLIP-VQA (+0.01408, excludes zero) while UniDet (+0.00632)
and 3-in-1 (−0.00298) span zero. Phase-I's placebo was −0.0039 (null), and on GenEval2 the sign
flips, so this is noise around zero rather than a gain.

But the *pattern* is worth stating explicitly, because it is the strongest anti-circularity evidence
available:

| | BLIP-VQA (VQA-style, near the selector) | UniDet (detection, disjoint) |
|---|---:|---:|
| spurious `random` − M1 drift | **+0.01408** (excludes zero) | +0.00632 (null) |
| genuine orientation effect (`correct` − `counterfactual`) | +0.04083 | **+0.06599** (largest) |

**The artefact appears on the selector's own family and vanishes on the disjoint one; the real effect
is largest on the disjoint one.** A single mechanism — leakage from the VQAScore labels into
VQA-style evaluation — cannot produce both rows.

**2. The dose response is not linear.** Anchoring a line on `correct` and `inverted`:

| arm | dose | observed | linear prediction | residual |
|---|---:|---:|---:|---:|
| `counterfactual` | 0.108 | 0.50913 | 0.53972 | **−0.03058** |
| `random` | 0.287 | 0.50314 | 0.51124 | −0.00810 |

`counterfactual` behaves almost like `random` despite flipping only 27% of labels versus 50%.
Flipping the labels on the reversal subset is disproportionately damaging. Those are precisely the
near-tie records (mean original margin 0.199 versus 0.319 elsewhere), so the hardest, most
informative comparisons are the ones the edited text corrupts. The monotone-trend primary was chosen
partly because it does not assume linearity; the ordering is what was preregistered, not the spacing.

**3. The effect is category-heterogeneous, and only partly explained by reversal rate.**

| category | reversal rate | `correct` − `counterfactual` |
|---|---:|---:|
| spatial | 0.577 | +0.0931 |
| numeracy | 0.137 | +0.0474 |
| color | 0.391 | +0.0440 |
| shape | 0.092 | +0.0416 |
| 3d_spatial | 0.271 | +0.0575 |
| texture | 0.164 | +0.0369 |
| complex | 0.238 | +0.0144 |
| non_spatial | 0.327 | +0.0000 |

Correlation between a category's reversal rate and its effect is **+0.466** across eight categories —
positive, as expected, but far from deterministic. Notably the effect persists in low-reversal
categories (`shape` +0.0416 at a 9.2% reversal rate), which suggests corrupting labels anywhere
degrades the model broadly rather than only in the categories whose own labels were flipped. With
eight categories and one seed this is an observation, not an established claim. `non_spatial` is
CLIPScore-scored on a ~10× compressed scale and is reported as secondary, never pooled.

## Mechanism: coalescence refuted, variance reduction is the leading untested account

See `reports/phaseFP_results/mechanism_verdict.md`. In summary:

- Shared noise **does** beat independent noise on the endpoint: +0.01413 [+0.00656, +0.02184].
- But the σ-profile that coalescence predicts is **identical** in both conditions
  (corr +0.2210 [+0.2085, +0.2336] versus +0.2260 [+0.2140, +0.2378]; band ratio 4.02× versus
  4.37×), so it carries no evidential weight for the hypothesis.
- Shared noise is a common-random-numbers construction and does show lower gradient variance
  (1.72× ratio on gradient-norm variance, mean pre-clip norm 86.5 versus 110.9), which explains all
  three observations where coalescence explains one. **Marked as hypothesis, not result.**

Per the preregistration, shared-noise coalescence is removed from the paper's claims.

## Fidelity and diversity (aggregation job 101263)

All six models scored in **one process against a single shared COCO val2017 reference sample**, which
is the strongest form of the "same-job M1" requirement and the thing whose absence made the two
Phase-I fidelity jobs incomparable.

| arm | dose | alignment | DINO | LPIPS | CMMD | FID |
|---|---:|---:|---:|---:|---:|---:|
| `correct` | 0.000 | 0.55697 | 0.31586 | 0.67255 | **42.52** | 43.58 |
| `counterfactual` | 0.108 | 0.50913 | 0.35548 | 0.67468 | 50.34 | 44.25 |
| `random` | 0.287 | 0.50314 | 0.36711 | 0.66217 | 42.73 | 42.97 |
| `inverted` | 0.573 | 0.46556 | 0.38799 | 0.66001 | 48.41 | 44.52 |
| `M1` | — | 0.49482 | 0.37906 | 0.66419 | 45.58 | 42.83 |
| `indepnoise` | — | 0.54284 | 0.32993 | 0.66285 | 44.04 | 43.23 |

*CMMD is reported as CLIP-ViT-L/14 MMD with sigma=10, **scaled by 1000** (`phaseC/fidelity_eval.py`, `scale=1000.0`); raw CMMD values are ~0.042-0.050.*

**The preregistered fidelity gate passes.** That is narrower than "fidelity is fine", and the
distinction matters. Two separate statements, which must not be run together:

- The **preregistered gate** is the treatment-versus-control pair, `correct` versus
  `counterfactual`: `d_cmmd` **−7.811**, `d_fid` −0.668, `d_precision` +0.040, `d_recall` +0.036,
  **pass = true**. The tolerance (`cmmd_tol = 2.0`) permits CMMD to *worsen* by up to 2.0; here it
  improved, so the gate is passed rather than merely satisfied.
- Separately, and descriptively, `correct` versus **M1** is +0.75 FID (marginally worse), −3.06
  CMMD (better), +0.020 precision, +0.027 recall. This comparison is not the preregistered gate and
  the 2.0 tolerance does not apply to it.

**Orientation dose has no detectable effect on fidelity.** An earlier draft of this document claimed
that wrongly-oriented labels damage fidelity, on the basis that `counterfactual` had the worst CMMD.
That claim is **withdrawn** — it does not survive the dose ordering or the extra seeds. Averaged over
all three seeds:

| dose | arm | CMMD (3-seed mean) | per-seed |
|---:|---|---:|---|
| 0.000 | `correct` | 43.04 | 42.5, 45.8, 40.8 |
| 0.108 | `counterfactual` | 48.73 | 50.3, 49.4, 46.5 |
| 0.287 | `random` | 43.31 | 42.7, 43.8, 43.4 |
| 0.573 | `inverted` | 46.98 | 48.4, 45.1, 47.4 |

corr(dose, CMMD) = **+0.264** — non-monotone, and `inverted` (maximally wrong) has *better* CMMD than
`counterfactual`. The spread across arms (5.7) is not organised by dose. FID spans only 1.55 across
all six original arms, well inside noise at n≈2,398.

This is a **better** result than the withdrawn one: the alignment gain is not bought with fidelity.
Lead with CMMD rather than FID, and add KID (roadmap task Q, still missing).

### The alignment–diversity trade-off is monotone in the same knob

| correlation across the four orientation arms | value |
|---|---:|
| dose vs alignment | −0.9322 |
| dose vs DINO diversity | +0.9166 |
| **alignment vs DINO diversity** | **−0.9906** |

This is the cleanest statement the campaign produces about cost: **the orientation dose that buys
alignment sells diversity, almost perfectly linearly.** It is not that training happens to reduce
diversity — the more correctly oriented the labels, the less diverse the output, monotonically
across the whole range. `inverted` is *more* diverse than untrained M1 (0.38799 versus 0.37906) and
also the worst aligned.

### The preregistered gate fails; the oracle-referenced comparison does not

| comparison | DINO | verdict |
|---|---:|---|
| `correct` − `counterfactual` (**preregistered**) | −0.03961 [−0.04590, −0.03342] | **FAIL** |
| `correct` − `counterfactual`, LPIPS | −0.00213 [−0.00471, +0.00040] | PASS (non-inferior) |
| `correct` − `M1` | −0.06320 | — |
| teacher best-of-4 price (same units) | −0.12672 | — |

The preregistered DINO gate **fails**, replicating Phase-I's −0.04136 almost exactly (−0.03961
here). That is not rewritten after the fact.

But the oracle-referenced picture is quite different. The student's diversity loss against M1 is
**exactly half** the price best-of-4 selection itself pays (−0.0632 versus −0.1267), and in absolute
terms the student remains **more diverse than the policy it amortizes** (0.31586 versus 0.24711;
teacher-random is 0.37383). So the student did not collapse toward the oracle's concentration — it
captured a comparable alignment gain at roughly half the diversity cost.

Two caveats on that comparison, as before: it is not matched (different estimand, prompt population,
CFG and step count — see `reports/audit_2026-08-07/README.md` §7), and the alignment magnitudes
(+0.0622 student-versus-M1 against +0.0717 teacher-versus-random) sit on different baselines and
populations, so "comparable gain" is an order-of-magnitude statement, not an equality.

**The diversity loss is concentrated in semantic/structural feature space rather than in low-level
perceptual variation.** LPIPS moves by −0.002 and passes non-inferiority while DINO moves by
−0.040. Note this is weaker than "purely semantic": DINOv2 features capture semantic *and*
structural similarity, and LPIPS is the more locally perceptual of the two. The practical reading
is that a diversity remedy should target mode coverage rather than pixel-level variation.

## Three-seed replication (promotion gate 3: MET)

Seeds 20260806 / 20260807 / 20260808. Data order differs completely across seeds (5/200 overlap in
the first 200 records), and within each seed the four arms still train on bit-identical images.

| seed | dose slope | 95% CI (prompts) | ordering probability |
|---|---:|---|---:|
| s1 | −0.13972 | [−0.15731, −0.12259] | 0.9254 |
| s2 | −0.09591 | [−0.11090, −0.08038] | 0.9968 |
| s3 | −0.13591 | [−0.15175, −0.12004] | 0.9998 |
| **across seeds** | **−0.12385** | range [−0.13972, −0.09591] | *(not averaged — see below)* |

| contrast | s1 | s2 | s3 | mean | range |
|---|---:|---:|---:|---:|---:|
| `correct` − `counterfactual` | +0.04784 | +0.03208 | +0.04063 | **+0.04018** | 0.01576 |
| `correct` − `random` | +0.05383 | +0.04297 | +0.05511 | +0.05064 | 0.01214 |
| `correct` − `inverted` | +0.09141 | +0.06163 | +0.08616 | +0.07973 | 0.02978 |

Every contrast holds its sign in every seed, the arm ordering is identical in all three, `inverted`
sits below M1 in all three, and all three evaluator families keep their sign per seed.

> **Do not average the ordering probabilities.** An earlier draft reported 0.9740 as an
> "across-seed ordering probability". That figure is the mean of the three *within-seed* prompt
> bootstraps (0.9254, 0.9968, 0.9998) and is **not** a probability integrating over training
> randomness — nothing in the bootstrap resamples seeds. The correct statement is that the full
> ordering is reproduced in **all three of three** seeds, with within-seed probabilities of
> 0.9254, 0.9968 and 0.9998.

**The dominant uncertainty is training randomness, not prompt sampling.** The across-seed range
(0.0438) exceeds any single seed's prompt-bootstrap width (~0.032). This is precisely why the two
are reported separately and never pooled. Seed 2 is the consistent outlier, driven by its `inverted`
arm landing at 0.48028 against ~0.467/0.473 elsewhere. With n=3, quote the range, not an SD.

## Best-of-four selection headroom on the held-out split (task T)

*Called a "teacher ceiling" in earlier drafts and in the roadmap. That is a misnomer and is
retired: this is the headroom of **one particular selection policy** — best-of-4 by VQAScore over
this teacher and sampler. A larger \(N\), a different scorer or a different sampler could exceed
it, so it bounds nothing.*

LSF 101493 (after 101377 was killed by `TERM_MEMLIMIT`; the 9,592 generated images were reused).
Frozen base SD3.5-M, 8 steps, CFG 7, 4 candidates per prompt, on **the same 2,398 val prompts the
students are scored on** — the previous +0.0717 figure was measured on the exp0 pool, which is the
**train** split.

| selection policy | CompBench primary |
|---|---:|
| `TeacherBest` (best-of-4 by VQAScore) | 0.53078 |
| `TeacherUniform` (uniform over the 4) | 0.47792 |
| `TeacherRandom` (uniform over the 3 non-oracle) | 0.45589 |

**Two corrections follow, and the second matters.**

1. The old train-split figure replicates on val under the *matched* control definition:
   best − random(non-oracle) = **+0.07489** here versus +0.0717 before. The ceiling is a stable
   property, not an artefact of the split.
2. **That control is biased, by exactly \(N/(N-1)=33.3\%\).** This is algebra, not measurement:
   \(\text{Uniform}=\tfrac1N\text{Best}+(1-\tfrac1N)\text{NonOracle}\) implies
   \(\text{Best}-\text{Uniform}=\tfrac{N-1}{N}(\text{Best}-\text{NonOracle})\) identically.

   Every candidate slot has now been CompBench-scored (LSF 101942), so each policy is an **exact
   per-prompt expectation** rather than a single random draw:

   | policy | CompBench primary |
   |---|---:|
   | best-of-4 (VQAScore argmax) | 0.53078 |
   | uniform over all 4 | **0.47226** |
   | uniform over the 3 non-oracle | 0.45276 |

   - **Unbiased headroom: +0.05852 [+0.05148, +0.06542]**
   - Biased (non-oracle control): +0.07803 [+0.06865, +0.08723]
   - Mixture-identity residual: **−1.11e−16**; observed bias ratio **1.3333** = \(N/(N-1)\) exactly.

   > **Correction.** An earlier version reported "about 42%", from the empirical ratio
   > 0.07489/0.05287 = 1.4165. Both terms were single-draw estimates, so the identity failed by
   > ~1 SE and the excess was noise. The exact figure is **33.3%**, and the unbiased headroom is
   > **+0.05852**, not +0.05287.

### The amortization ratio, now genuinely matched

The C1 transfer contrast's 2,098 primary prompts are an exact subset of this val pool, so a **joint**
bootstrap is now possible: prompts are resampled once and both numerator and denominator recomputed
on the same resample, capturing their correlation.

| quantity | Δ | 95% CI |
|---|---:|---|
| numerator: B4 − B2 @ 4 steps | +0.00045 | [−0.00889, +0.00975] |
| denominator: best-of-4 − uniform @ 8 steps (**exact expectation**) | +0.05852 | [+0.05158, +0.06553] |
| **ratio** | **0.0077** | **[−0.154, +0.166]** |

**One-sided 95% upper bound: 0.142.**

This *supersedes* the 11.7% reported in `reports/audit_2026-08-07/README.md` §3, and it is looser,
not tighter. The earlier figure was optimistic for two reasons that have now been fixed: it used the
inflated non-oracle denominator, and it combined independent bootstraps across different prompt
populations. Cite **~14%**, from a joint bootstrap on 2,098 matched prompts with an exact-expectation denominator.

## Two comparisons that change the framing

### The offline-PSO configuration is not worse than ours — if anything slightly better

Same cache, same seed, same step count; only PSO's choices substituted (β=50, no anchor, independent
pair noise, ratio clipping 0.1).

| arm | CompBench primary |
|---|---:|
| `PSOBaseline` | **0.56418** |
| `correct` (s1) | 0.55697 |

`correct` − `PSO` = **−0.00721 [−0.01494, +0.00033]** — the pooled interval marginally spans zero,
so this is approximately a tie, with BLIP-VQA favouring PSO (−0.01832 [−0.02653, −0.01038]) and
UniDet null (+0.00528 [−0.01053, +0.02057]).

Taken with `reports/audit_2026-08-07/pso_equivalence.md`, which shows the objective *reduces to* the
PSO equation, this confirms that **the loss is not a contribution**.

**But the alignment comparison alone is misleading, and the diversity measurement settles it.** The
PSO arm has the **lowest DINO diversity of all fifteen models** (0.29985, against 0.31586/0.33239/
0.31020 for the three `correct` seeds and 0.37906 for M1). It drops the M1 anchor, and that is where
the anchor was earning its place — not on alignment.

Fitting the alignment–diversity frontier on all twelve orientation arms (3 seeds × 4 arms):

\[
\text{alignment} = 0.9359 - 1.1994 \times \text{DINO},
\qquad r = -0.9622,\qquad \text{residual SD} = 0.00828.
\]

| model | alignment | DINO | frontier prediction | residual |
|---|---:|---:|---:|---:|
| `PSOBaseline` | 0.56418 | 0.29985 | 0.57622 | **−0.01204** |
| `CorrectIndepNoise` | 0.54284 | 0.32993 | 0.54014 | +0.00270 |
| `M1` | 0.49482 | 0.37906 | 0.48121 | +0.01361 |

> **Correction — the residual argument does not survive a Pareto analysis.** An earlier draft claimed
> "PSO buys alignment by spending diversity at a slightly worse exchange rate", from its −0.012
> regression residual (~1.5 residual SDs). That inference assumes the linear fit *is* the frontier.
> Computing the **nondominated set** instead — no functional form assumed — puts **PSOBaseline on
> the Pareto front**: it has the highest alignment of all fifteen models, and nothing beats it on
> both axes. **Neither PSO's configuration nor ours Pareto-dominates the other.** The claim is
> withdrawn.

Empirical Pareto front (maximise alignment *and* DINO diversity), 11 of 15 models nondominated:

| model | alignment | DINO |
|---|---:|---:|
| `PSOBaseline` | **0.56418** | 0.29985 |
| `CorrectS3` | 0.55947 | 0.31020 |
| `CorrectFixed` | 0.55697 | 0.31586 |
| `CorrectIndepNoise` | 0.54284 | 0.32993 |
| `CorrectS2` | 0.54191 | 0.33239 |
| … | … | … |
| `M1Fixed` | 0.49482 | 0.37906 |
| `InvertedFixed` | 0.46556 | 0.38799 |

The four dominated models are each beaten by a **different seed of the same arm**
(`CounterfactualFixed` by `CounterfactualS2`, `RandomFixed` and `RandomS2` by `RandomS3`,
`InvertedS2` by `M1Fixed`). Seed noise is therefore comparable to the frontier's own structure at the
low-alignment end — further reason not to read the fitted line as a tight frontier.

Per-seed slopes over the four orientation arms differ noticeably: **−1.226** (r = −0.991),
**−1.373** (r = −0.879), **−1.128** (r = −0.977). Seed 2 is both the flattest fit and the least
linear.

The useful statement for the paper is therefore not "PSO's settings are better" but: **alignment and
diversity trade along a single frontier, and every configuration tested so far — ours, PSO's, shared
versus independent noise — lands on or below it.** Nothing yet moves the frontier outward, which is
the thing a genuine methodological contribution would have to do.

### The frontier is not yet earned, and one knob is the reason

**This claim is currently under-identified and must not be written up as established.** Twelve of the
fifteen points come from a *single* knob: orientation dose. Since dose→alignment is −0.9322 and
dose→DINO is +0.9166, the alignment↔DINO correlation of −0.9622 is close to tautological *within that
family* — one parameter moving two collinear quantities traces a **path**, not a frontier. Only three
points sit off that family: PSO (below the line), `indepnoise` (on it), and M1 (above it).

A second, independent knob is what turns the path into a frontier. **Task K is therefore reinstated
as the highest-priority experiment**, reversing the earlier decision to deprioritise it — that
decision assumed the very conclusion the sweep is meant to test. Running (LSF 101930–101934):
β ∈ {10, 30, 300} at fixed anchor, plus anchor ∈ {0.25, 4.0} at β = 100.

- If those points trace the same line, the frontier is established and it is the paper's spine.
- If they trace a different line, there is a two-parameter surface and the frontier claim is wrong.

Either outcome is informative, and the answer is needed *before* drafting.

### Shared versus independent noise is a move along the frontier, not an improvement

| | alignment | DINO | frontier residual |
|---|---:|---:|---:|
| `correct` (s1) | 0.55697 | 0.31586 | −0.0001 |
| `CorrectIndepNoise` | 0.54284 | 0.32993 | +0.0027 |

Both sit on the line, and their residuals differ by 0.0028 against a residual SD of 0.0083 —
statistically indistinguishable positions. The +0.01413 alignment gain is bought with −0.01407 DINO,
essentially the frontier exchange rate. So **shared noise is a reparameterisation of effective
preference strength, not a design choice with independent value** — which is a sharper and more
damaging statement than "coalescence is refuted", and supersedes the framing in
`mechanism_verdict.md`.

This also gives the common-random-numbers hypothesis a cheap decisive test: if the advantage is
purely CRN variance reduction, raising the number of sampled timesteps or using antithetic noise in
the independent condition should close the gap. One run either confirms or kills it.

One seed for the PSO arm; the frontier fit uses three seeds per orientation arm.

### The student matches teacher best-of-4 — but only ties it on the disjoint evaluator

On the same 2,098 held-out primary prompts:

| deployable system | steps | CompBench primary | inference cost |
|---|---:|---:|---|
| `correct` student (mean of 3 seeds) | 4 | 0.55278 | 1 generation |
| teacher best-of-4 | 8 | 0.53078 | 4 generations + a VQAScore call |
| teacher, no selection | 8 | 0.47792 | 1 generation |
| `M1`, no preference training | 4 | 0.49482 | 1 generation |

**The pooled comparison is misleading and must not be reported alone.** Split by evaluator family:

| family | student (3-seed) | teacher best-of-4 | difference |
|---|---:|---:|---:|
| BLIP-VQA | 0.74441 | 0.70065 | +0.04376 |
| **UniDet** | 0.41630 | 0.41941 | **−0.00311** |
| 3-in-1 | 0.38736 | 0.35530 | +0.03206 |

On the **architecturally disjoint detection family the student does not lead — it is a dead heat**.
The pooled margin is carried entirely by the VQA-style families, which are the ones closest in kind
to the VQAScore model that produced the training labels.

Two explanations were available for a student exceeding the oracle: (a) the distilled base is simply
better at this operating point, or (b) preference training partly over-optimises toward what
VQA-style scoring rewards. The family split does **not** kill (b) — it is exactly the signature (b)
predicts. The honest claim is therefore: *the four-step student reaches teacher-best-of-4 alignment,
with the margin concentrated on VQA-style evaluators and a tie on the detection family.*

### The compute claim, accounted properly

| system | candidates | steps | CFG | forwards/step | **total NFE** | extra |
|---|---:|---:|---:|---:|---:|---|
| `correct` student | 1 | 4 | 1.0 | 1 | **4** | — |
| teacher best-of-4 | 4 | 8 | 7.0 | 2 | **64** | 1 VQAScore call (11B) over 4 images |

**16× fewer denoiser evaluations per returned image**, plus no selection-model call.

Paired over prompts, with the three student seeds averaged per prompt first:
**+0.02200 [+0.01260, +0.03135]**, P(>0) = 1.0000. Per seed: s1 +0.02619, s2 +0.01113, s3 +0.02868 —
all positive, but seed 2 is less than half of seed 3, so quote the interval rather than any single
seed.

> **Correction.** An earlier draft said "a quarter of the generation cost", counting *generations*
> rather than denoiser evaluations. That is wrong twice over: it ignores the 4-versus-8 step count,
> and it ignores that classifier-free guidance doubles the forwards per step — `euler_cfg_sample`
> sets `use_cfg = cfg > 1.0`, so the CFG-7 teacher runs two passes per step while the CFG-1 student
> runs one. Parallel candidate generation would change latency but not total compute.

The remaining caveats stand: different operating points (4 steps / CFG 1 versus 8 / CFG 7), and M1
alone already outscores the unselected teacher, so part of any margin belongs to distillation rather
than preference training. The comparison also still needs a **paired interval over the three student
seeds** before it is quoted as a result. None of this contradicts the C1 null, which concerns
*trajectory selection transferring through consistency regression* — a different channel.

## Reading order (headline numbers are the 3-seed means)

Seed 1 is reported first in several sections above for historical reasons, and **s1 happens to be the
most favourable of the three seeds** for the headline contrast (`correct` − `counterfactual`:
s1 +0.04784 is the maximum of {0.04784, 0.03208, 0.04063}; the 3-seed mean is +0.04018, 16% lower).

Since the across-seed range (0.0438) exceeds the within-seed prompt-bootstrap width (~0.032),
**training randomness dominates and the 3-seed mean is the number to cite**. Any table quoting a
single seed should be read as illustrative. This is a presentation hazard rather than a substantive
one, but it is the kind that costs credibility for free.

## Outstanding

- **Task K — the knob sweep (HIGHEST PRIORITY).** Running, LSF 101930–101934. The frontier claim is
  under-identified until a second knob is tested; see the frontier section above.
- **Exact policy expectations.** Running, LSF 101942 — makes the headroom and amortization numbers
  exact rather than draw-based.
- **REPA isolation.** Running, LSF 101912–101917.
- **Human edit audit.** 210 items, packet rendering under LSF 101922. Needs a human.
- **KID** and satisfaction-stratified diversity (roadmap task Q) — still missing.
- **CRN test for shared noise** — raise the timestep count or use antithetic noise in the
  independent condition; one run confirms or kills the variance-reduction hypothesis.
- **A canonical sCM/rCM baseline** (roadmap task M).

*(The teacher ceiling and the offline-PSO baseline were listed here in an earlier draft; both are now
complete and have full sections above.)*
