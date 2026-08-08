# Amortizing best-of-K compositional selection into a four-step generator

**Report date: 2026-08-08.** Self-contained summary of what the project has established.
Every number here is current and recomputed from raw artifacts; detail documents are linked per
section.

---

## 1. The question

Stable Diffusion 3.5-Medium produces semantically different images from different sampling
trajectories for the same prompt. Some candidates satisfy the requested attributes, counts and
relations better than others. Selecting the best of four candidates at inference is worth
**+0.05852 [+0.05148, +0.06542]** on held-out T2I-CompBench — but it costs four teacher generations
plus a scorer call.

Can that per-prompt selection advantage be **amortized** into a single deterministic four-step
student, so it is paid once at training time instead of every time an image is generated?

The project answers this by asking *through which channel* reward can enter — the sampled
trajectory, the regression target, the initial noise, or the loss itself — and showing these are
not equivalent.

---

## 2. Headline results

| # | Result | Evidence |
|---|---|---|
| 1 | Selecting better teacher **trajectories** does not transfer through ordinary consistency regression | ≤ **14%** of the selection headroom, one-sided 95% |
| 2 | An explicit **comparative loss** does transfer, and the effect is **causal in the label orientation** | dose slope **−0.124**, 3 seeds, all same sign |
| 3 | The signal is **prompt-specific**, not a generic quality prior | `correct` − `counterfactual` = **+0.040** on identical images |
| 4 | Alignment and diversity trade along one empirical frontier | r = **−0.96**; nothing tested moves it outward |
| 5 | The objective is **not novel** — it reduces to the offline PSO equation | see §8 |

---

## 3. The negative result: trajectory selection does not amortize

Training a four-step student on *selected* teacher trajectories (B4) versus random ones (B2), with
everything else matched, moves the held-out benchmark by **+0.00045 [−0.00889, +0.00977]** — a null.

Against the selection headroom it is meant to capture:

| quantity | Δ | 95% CI |
|---|---:|---|
| numerator: selected − random trajectory training | +0.00045 | [−0.00889, +0.00977] |
| denominator: best-of-4 − no selection, exact expectation | +0.05852 | [+0.05158, +0.06553] |
| **transfer ratio** | **0.0077** | [−0.154, +0.166] |

**At most 14.2% of the available headroom transfers through this channel** (one-sided 95% bound,
ratio 0.0077), from a joint bootstrap over 2,098 prompts shared by numerator and denominator — the
same prompts resampled once, so their correlation is captured.

The denominator deserves care. "No selection" must mean a **uniform draw over all four candidates**.
Using a control that *excludes* the winner inflates the measured headroom by exactly
\(N/(N-1)=33.3\%\), since
\(\text{Best}-\text{Uniform}=\tfrac{N-1}{N}(\text{Best}-\text{NonOracle})\) identically. Every
candidate was therefore scored individually so each policy is an exact per-prompt expectation; the
mixture identity then holds to −1.1e−16 and the bias ratio is 1.3333 exactly.

| selection policy | CompBench primary |
|---|---:|
| best-of-4 (VQAScore argmax) | 0.53078 |
| uniform over all 4 — *the correct control* | 0.47226 |
| uniform over the 3 non-winners — *biased* | 0.45276 |

This is a limitation of **this objective in this setting**, not a theorem that best-of-K can never
be amortized.

---

## 4. The positive result: orientation is causal

An explicit pairwise comparative loss *does* move fresh samples. The decisive experiment freezes the
image pair and varies **only the sign of the label**.

For each prompt, the highest- and lowest-scoring teacher candidates form a fixed unordered pair
\((a_i,b_i)\). Four arms receive that identical pair with different orientations: `correct`
(original-prompt sign), `counterfactual` (sign implied by a one-atom-edited prompt), `random`
(balanced), `inverted` (opposite).

### The design verifiably held

Not asserted — measured after the fact. Across all 3,000 training steps:

- teacher endpoint tensors had **identical SHA-256** in all four arms;
- the batch order and the sampled noise level σ were **byte-identical**;
- each step's frozen-reference error pair was an exact **match or swap** of the `correct` arm's, with
  **zero unexplained steps** — and since that error is a deterministic function of image, σ, noise
  and prompt, this establishes that all of them matched and only the *label* differed.

| arm | match | swap | neither |
|---|---:|---:|---:|
| `correct` | 3000 | 0 | **0** |
| `counterfactual` | 2174 | 826 | **0** |
| `random` | 1541 | 1459 | **0** |
| `inverted` | 0 | 3000 | **0** |

The swap counts are what the cache dictates: `counterfactual`'s 826 are exactly its flagged
reversals, `random`'s 1459 ≈ 50%.

### Orientation dose

Arms are ordered by how far their labels sit from the correct ones, in VQAScore margin units:

\[
m_a=\frac1n\sum_i\bigl[R(x_{i\,p_{ai}},c_i)-R(x_{i\,n_{ai}},c_i)\bigr],\qquad d_a=m_{\text{correct}}-m_a
\]

giving `correct` 0.000, `counterfactual` 0.108, `random` 0.287, `inverted` 0.573. This is computed
from the cache before any training. Note dose is a *margin*, not a flip fraction — `counterfactual`
flips 27.2% of records but scores 0.108 because the records it flips are disproportionately
near-ties.

### The result, replicated across three seeds

The preregistered primary is the monotone slope of score against dose.

| seed | dose slope | 95% CI (prompts) | ordering probability |
|---|---:|---|---:|
| 20260806 | −0.13972 | [−0.15731, −0.12259] | 0.9254 |
| 20260807 | −0.09591 | [−0.11090, −0.08038] | 0.9968 |
| 20260808 | −0.13591 | [−0.15175, −0.12004] | 0.9998 |
| **mean** | **−0.12385** | range [−0.13972, −0.09591] | reproduced **3 of 3** |

Absolute scores, all arms trained on identical images:

| seed | `correct` | `counterfactual` | `random` | `inverted` |
|---|---:|---:|---:|---:|
| 20260806 | 0.55697 | 0.50913 | 0.50314 | 0.46556 |
| 20260807 | 0.54191 | 0.50983 | 0.49894 | 0.48028 |
| 20260808 | 0.55947 | 0.51884 | 0.50435 | 0.47330 |

with untrained M1 at 0.49482. Contrasts against `correct`, three-seed means:

| contrast | mean | range | same sign in all seeds |
|---|---:|---:|:-:|
| − `counterfactual` | **+0.04018** | 0.01576 | yes |
| − `random` | +0.05064 | 0.01214 | yes |
| − `inverted` | +0.07973 | 0.02978 | yes |

**Training on reversed labels leaves the model worse than not training at all** — `inverted` sits
below untrained M1 in every seed (seed 1: −0.02926 [−0.03700, −0.02177], every evaluator family
negative). The label sign carries directional information the model acts on.

**Uncertainty is dominated by training randomness, not prompt sampling.** The across-seed range
(0.0438) exceeds any single seed's prompt-bootstrap width (~0.032). The two are therefore reported
separately and never pooled, and three-seed means are the numbers to cite.

### Independent replication on a second benchmark

GenEval2 Soft-TIFA — different evaluator, different prompts, different scoring architecture:

slope **−0.14317 [−0.17532, −0.11140]**, ordering probability 0.9840, and every contrast agreeing
with CompBench within its interval (`correct` − `counterfactual` = +0.04015 [+0.02247, +0.05793]).

---

## 5. Is it prompt-specific, or a generic quality prior?

This is what the fixed pair exists to answer. `counterfactual` orients the same two images using a
prompt with **one atom changed** (a colour, count, relation or verb). It destroys most of the
benefit — **+0.04018** three-seed mean — even though it flips only 27% of labels. Since the images
are identical, this cannot be an effect of which pairs were mined.

### The evaluator families agree, and the disjoint one agrees most

T2I-CompBench is not one metric. Its categories are scored by architecturally distinct instruments,
and the training labels come from a fourth (VQAScore / CLIP-FlanT5).

| family | dose slope (3-seed mean) | `correct` − `counterfactual` | 95% CI |
|---|---:|---:|---|
| BLIP-VQA | −0.17702 | +0.04083 | [+0.03098, +0.05102] |
| **UniDet** (detection) | −0.09081 | **+0.06599** | [+0.04936, +0.08278] |
| 3-in-1 | −0.06344 | +0.01439 | [+0.00732, +0.02130] |

UniDet is detection-based and shares no architecture with the label source — and it shows the
**largest** text-specificity effect. All three families keep their sign in every seed.

### A double dissociation

The `random` arm drifts very slightly against M1 (+0.00832 [+0.00109, +0.01567] on CompBench, with
the sign flipping on GenEval2 — i.e. noise around zero). Where that drift appears is informative:

| | BLIP-VQA (VQA-style, near the label source) | UniDet (disjoint) |
|---|---:|---:|
| spurious `random` drift | **+0.01408** (excludes zero) | +0.00632 (null) |
| genuine orientation effect | +0.04083 | **+0.06599** (largest) |

**The artefact appears on the label source's own family and vanishes on the disjoint one; the real
effect is largest on the disjoint one.** A single leakage mechanism cannot produce both rows.

### Where the effect lives

It is present in every category but far from uniform, and correlates only moderately with how often
each category's labels were actually flipped (r = +0.466 across eight categories):

| category | flip rate | `correct` − `counterfactual` |
|---|---:|---:|
| spatial | 0.577 | +0.0931 |
| 3d_spatial | 0.271 | +0.0575 |
| numeracy | 0.137 | +0.0474 |
| colour | 0.391 | +0.0440 |
| shape | 0.092 | +0.0416 |
| texture | 0.164 | +0.0369 |
| complex | 0.238 | +0.0144 |

The effect persists in low-flip categories (`shape` at 9.2%), suggesting corrupting labels anywhere
degrades the model broadly rather than only where they were corrupted.

---

## 6. How the loss actually works

Two mechanistic findings, both measured rather than assumed.

**The loss pushes away from the rejected sample rather than pulling toward the preferred one.**
Relative to the frozen reference, the student's error on the *preferred* endpoint gets **worse**
(−0.0078) while its error on the rejected one gets worse faster (+0.0146). About 60% of the achieved
margin comes from degrading the loser. This is repulsion-dominated, it is uniform across noise
levels, and it is a plausible proximate cause of the diversity cost in §7.

**Shared corruption noise helps, but not for the proposed reason.** Using the same noise for both
branches beats independent noise by +0.01413 [+0.00656, +0.02184]. The proposed explanation was
high-noise input coalescence — as σ→1 the two inputs converge while their targets stay separated.
That prediction was tested and **falsified**: the σ-profile is statistically identical with and
without shared noise (correlation +0.2210 [+0.2085, +0.2336] versus +0.2260 [+0.2140, +0.2378];
band ratios 4.02× versus 4.37×). The σ-dependence is a property of the rectified-flow
parameterisation, not of the pair construction, so coalescence is **removed from the project's
claims**.

A common-random-numbers variance-reduction account fits all three observations — the endpoint
benefit, the absent σ-signature, and a 1.72× lower gradient-norm variance — but it has **not** been
tested directly and is recorded as a hypothesis.

Also relevant to any strength sweep: gradient clipping fires on **100%** of steps, so the update is
a normalised direction with fixed step length. β changes the direction mix between the preference
and anchor terms, not the step size.

---

## 7. What it costs

### Fidelity: no detectable cost

All fifteen models were scored in one process against a single shared COCO val2017 reference
(n = 2,398). The preregistered gate passes: `correct` versus `counterfactual` gives CMMD −7.811,
FID −0.668, precision +0.040, recall +0.036.

Orientation dose has **no detectable effect on fidelity**: CMMD by dose (three-seed means) runs
43.04 → 48.73 → 43.31 → 46.98, non-monotone, corr(dose, CMMD) = +0.264. The alignment gain is not
bought with fidelity. *(CMMD is CLIP-ViT-L/14 MMD ×1000; raw values ≈0.043–0.049.)*

### Diversity: a real cost, correctly referenced

The preregistered DINO gate **fails**: `correct` − `counterfactual` = −0.03961 [−0.04590, −0.03342]
against a tolerance of 0.02. LPIPS passes non-inferiority (−0.00213 [−0.00471, +0.00040]), so the
loss is concentrated in semantic/structural feature space rather than low-level perceptual variation.

But the honest reference is the policy being amortized, not the untrained model. Best-of-4 selection
*itself* costs diversity:

| policy | DINO | LPIPS |
|---|---:|---:|
| teacher, no selection | 0.37383 | 0.59087 |
| teacher best-of-4 | 0.24711 | 0.43089 |
| **selection's own price** | **−0.12672** | **−0.15998** |

The student gives up **−0.06320** against M1 — about **half** what the selection policy pays — and
remains **more diverse in absolute terms** (0.31586) than the policy it amortizes (0.24711). The gate
still fails, and that is reported as a failure; but the student did not collapse toward the oracle.

*(Caveat: these are the same estimators but different estimands and populations — selection over a
fixed bank on 1,200 prompts at CFG 7/8 steps versus trained models on 400 prompts at CFG 1/4 steps.
It is an order-of-magnitude reference, not a matched contrast.)*

### Alignment and diversity trade along one frontier

Across the twelve orientation arms (4 arms × 3 seeds), alignment and DINO diversity are almost
perfectly anti-correlated: **r = −0.9622**, fit `alignment = 0.9359 − 1.1994 × DINO`, residual SD
0.0083. Per-seed slopes −1.226, −1.373, −1.128.

Computing the **nondominated set** — no functional form assumed — 11 of 15 models are on the Pareto
front, spanning `PSOBaseline` (highest alignment, lowest diversity) through the `correct` arms and M1
to `inverted`. The four dominated models are each beaten by *a different seed of the same arm*, so
seed noise is comparable to the frontier's own structure at the low-alignment end.

**Every configuration tested lands on or below one frontier.** Nothing yet moves it outward — which
is what a genuine methodological contribution would have to do. This reframes the open problem from
*tune the objective* to *escape the frontier*.

**This claim is not yet fully identified**: twelve of the points come from a single knob
(orientation dose), so alignment↔DINO being collinear is partly built in. A β and anchor-weight
sweep is running to supply an independent knob; if those points trace the same line the frontier is
established, and if they trace a different one the claim is wrong.

---

## 8. What this is not

**The loss is not novel.** Under the stated Gaussian transition bridge, the objective *reduces
exactly* to the offline PSO / Diffusion-DPO equation — there is no loss-level difference to defend.
The surviving differences from PSO are the M1 anchor, shared pair noise, and β; PSO's ratio clipping
is inert at our scale (0% of examples fall outside its window).

Running PSO's own configuration (β=50, no anchor, independent noise, clipping 0.1) on the same cache
gives **0.56418** versus our 0.55697 — and it sits *on* the Pareto front, with the lowest diversity
of all fifteen models (0.29985). Neither configuration dominates the other. Our particular
hyperparameters confer no advantage; the anchor earns its place on diversity, not alignment.

**The contribution is therefore the measurement, not the method**: the channel distinction, the
fixed-pair causal identification, and the alignment–fidelity–diversity characterisation.

---

## 9. Amortization achieved

On the same 2,098 held-out prompts, comparing deployable systems:

| system | NFE per image | CompBench primary |
|---|---:|---:|
| `correct` student, 4 steps (3-seed mean) | **4** | 0.55278 |
| teacher best-of-4, 8 steps, CFG 7 | **64** | 0.53078 |
| teacher, no selection | 16 | 0.47226 |
| M1, no preference training | 4 | 0.49482 |

Paired over prompts: **+0.02200 [+0.01260, +0.03135]**, P(>0) = 1.0, at **16× fewer denoiser
evaluations** and no scorer call.

Two honest qualifications. Split by evaluator family, the student leads on BLIP-VQA (+0.04376) and
3-in-1 (+0.03206) but is a **dead heat on the disjoint UniDet family (−0.00311)** — consistent with
some over-optimisation toward VQA-style scoring. And this is a system-level comparison at different
operating points (4 steps/CFG 1 versus 8 steps/CFG 7), not a controlled ablation; M1 alone already
outscores the unselected teacher, so part of the margin belongs to distillation.

---

## 10. Status and what remains

**Complete:** result registry; fixed-pair cache and dose audit; smoke/identity gate; the four-arm
causal pilot; three-seed replication; the noise-coupling falsification; the offline-PSO baseline;
the selection-headroom measurement on matched prompts; cross-model fidelity and diversity; the
non-cherry-picked visual audit.

**Running:** β and anchor-weight sweep (frontier identification); REPA isolation at matched seeds;
edit-audit pair rendering; a JVP feasibility probe for a canonical sCM baseline.

**Outstanding:**

| item | why it matters |
|---|---|
| **Blinded human edit audit** (210 items, packet prepared) | the prompt-specificity claim ultimately rests on the edits being valid; requires a human |
| Canonical sCM/rCM baseline | establishes whether M1 is a fair stand-in for a modern method |
| KID; satisfaction-stratified diversity | completes the quality frontier |
| Direct test of the variance-reduction hypothesis | one run confirms or kills it |
| Second backbone; unseen natural-prompt distribution | generality |

**Promotion gate: 8 of 9 conditions met.** The exception is the diversity gate, which failed — and
which the frontier result in §7 explains rather than excuses.

---

## Detail documents

| subject | file |
|---|---|
| full results, with per-seed tables | `reports/phaseFP_results/RESULTS.md` |
| three-seed replication | `reports/phaseFP_results/seeds.md` |
| mechanism / noise-coupling verdict | `reports/phaseFP_results/mechanism_verdict.md` |
| zero-training audit | `reports/audit_2026-08-07/README.md` |
| PSO reduction | `reports/audit_2026-08-07/pso_equivalence.md` |
| selection diversity price | `reports/audit_2026-08-07/oracle_diversity_101231/best_of_4.md` |
| preregistered analysis plan | `phaseFP/PREREGISTRATION.md` |
| sCM baseline design | `phaseM/PREREGISTRATION.md` |
| immutable result registry | `reports/registry/registry.md` |
| A–Z roadmap and gates | `reports/research_verdict_and_roadmap.md` |
| visual audit | `reports/phaseFP_results/visual_audit/contact_sheet.html` |
| supervisor deck | `reports/architecture.html` |

## Standing cautions

- **Join evaluations on prompt text, never on `idx`** — `idx` is assigned per job, and two jobs can
  use the same index for different prompts.
- **Never compare fidelity across generation prompt pools.** Only the COCO-caption protocol counts.
- Keep the three selection quantities apart: **+0.05852** is best-of-4 over the teacher on held-out
  prompts (the amortization denominator); **+0.0485** is best-of-4 over the *student's own noise
  seeds*; **+0.1220** is in-objective cache scoring. They never share a denominator.
- Cite **three-seed means**; across-seed range exceeds prompt-bootstrap width.
- Do not use BOND as an impossibility theorem, or the Wasserstein proximal map of arXiv:2605.11361
  as a KL-tilt representation theorem.
