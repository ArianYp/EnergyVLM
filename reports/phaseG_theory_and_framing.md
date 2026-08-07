# Phase-G theory audit — historical hypotheses and the surviving claim

**Status: SUPERSEDED THEORY NOTE, corrected 2026-08-07.** The measurements below are retained for provenance.
The conditional-mean, reward-curvature, target-variance, and optimum-invariance mechanisms are not current
claims. The source-of-truth synthesis is `reports/project_handoff_2026-08-07.md`.

## Current theory status

### What the experiments establish

The Phase-C and Phase-G measurements establish objective-specific empirical contrasts:

1. selected-trajectory consistency regression produced
   \(+0.0004\;[-0.0090,+0.0097]\) independent CompBench transfer;
2. its control contrast was \(+0.0007\;[-0.0084,+0.0096]\);
3. selected-target variants moved the model differently, but the successive conditional-mean, reward-curvature,
   and target-extremity explanations were not sustained by all controls; and
4. an explicit Phase-I pairwise objective did change fresh samples relative to a randomized-label placebo.

The defensible conclusion is therefore a measured **channel difference under the audited objectives**, not a
universal theorem about deterministic generators or best-of-\(N\) distillation.

### Shared-noise hypothesis that remains open

For the Phase-I pair, shared corruption noise gives

\[
z^+-z^-=(1-\sigma)(x^+-x^-),
\qquad
u^+-u^-=x^- - x^+.
\]

As \(\sigma\rightarrow1\), the two student inputs coalesce while their targets remain distinct. If the model
outputs are locally similar for the coalesced pair, the relative-error objective creates explicit competition
between the outcomes exactly where the first four-step update operates. This is a limiting argument, not a
proof of the full training dynamics. It is admitted into the paper only if:

1. shared noise outperforms independent pair noise at matched compute;
2. the effect is concentrated in the predicted high-\(\sigma\) region; and
3. gradient-variance measurements agree with the proposed mechanism.

If those tests fail, report the positive comparative-objective result without a mechanistic explanation.

### Statements that are retired

- A deterministic generator cannot represent an aligned distribution.
- The selected-trajectory population optimum is invariant under reweighting.
- The dose and irreducible regression error are the same conditional-variance quantity.
- Reward curvature or target extremity explains the Phase-G outcome.
- BOND proves that forward-KL imitation cannot reproduce best-of-\(N\).
- The deterministic proximal map in arXiv:2605.11361 represents the KL exponential tilt.

BOND actually treats imitation on best-of-\(N\) samples as a forward-KL distribution-matching route. It
motivates a soft all-candidate baseline but does not prove the C1 null. Likewise, arXiv:2605.11361 separates
the KL exponential tilt from a Wasserstein proximal transport; its deterministic proximal-argmax statement
belongs to the Wasserstein construction.

### Quantitative claim discipline

The corrected teacher best-of-four oracle is \(+0.0717\;[+0.0573,+0.0865]\), while the selected-trajectory
effect is \(+0.0004\;[-0.0090,+0.0097]\). The descriptive ratio

\[
\frac{0.0097}{0.0717}\approx13.5\%
\]

is not yet a formal \(95\%\) upper bound because the two quantities use different experiment populations.
Compute a matched or joint bootstrap before making the bound a headline. The older sections below explain how
the previous hypotheses arose; they must not override this correction.

---

## Historical note retained for provenance

---

## 1. What each experiment licenses (keep these separate)

The two arguments below are often conflated. They are not the same, and only the second closes the channel.

| evidence | licenses | does NOT license |
|---|---|---|
| C1 null (+0.0004 [−0.0090,+0.0097]) **+ negative control** (B2′−B2 = +0.0007 [−0.0084,+0.0096]) | *the treatment produced nothing measurable* | any claim that nothing **was available** |
| **Ceiling** (MAP dose at k) | *nothing measurable was available* — a non-identifying design | — |
| **RESULT (88069):** MAP dose on the PRIMARY = **+0.0127 [−0.0011, +0.0263]** | Phase G k=3 **was** effectively non-identifying on the primary metric | — |

If the first is allowed to do the second's work, the correct referee response is that we mistook an
underpowered experiment for a structural result. That is precisely the error the ceiling measurement exists to
prevent, and the writeup must keep them in separate paragraphs.

## 2. The seed control is a NEGATIVE CONTROL ARM, not a variance estimate

**Do not write "run-to-run variance is small."** One replicate cannot estimate a variance; a single
control–control draw of +0.0007 is entirely consistent with an underlying SD of 0.01.

What it does license is stronger rhetorically anyway: **a treatment–control difference and a control–control
difference are the same size.** Report them side by side as two draws from the same null, each with its paired
bootstrap CI:

| comparison | Δ (7-cat equal-weighted) | 95% CI |
|---|---|---|
| **treatment** B4 − B2 | **+0.0004** | [−0.0090, +0.0097] |
| **negative control** B2′ − B2 | **+0.0007** | [−0.0084, +0.0096] |

**The asymmetry is the point, and it must be stated explicitly** or a referee will supply it:

> B2′ differs from B2 only in data order, the per-step Δ draws, and kernel nondeterminism — initialisation is
> deterministic from base weights and the selection cache is shared. B4 differs from B2 in **which trajectories
> it trains on**, a +0.124 supervision-gap perturbation. The treatment comparison therefore carries strictly
> *more* perturbation than the seed comparison, and still produced *less* movement.

Symmetry would have been suggestive; the asymmetry is the result.

## 3. The theory paragraph: an ESTIMATOR obstruction, not a representability one

An earlier formulation of ours — "reward tilting is a property of a distribution; distillation learns a map" —
is rhetorically loaded toward a **false** reading, and [arXiv 2605.11361](https://arxiv.org/abs/2605.11361) is
its counterexample. Their **Thm 4.1**: the aligned law *is* a deterministic pushforward,

  Q_λ = (T_λ)#P,  T_λ(y) ∈ argmax_x { r(x) − λ‖x − y‖² }

so a deterministic map suffices to represent the tilt. (They also show KL-tilting is NP-hard, NP⊆BPP, for
rank-one negative-semidefinite quadratic rewards, via Moitra et al. 2026. They assume oracle access to the
score and to the argmax, and contain **no** theorem about learning a pushforward by regression and **no**
mention of distillation, few-step, or consistency models.)

The correct claim is narrower and better:

> The tilted law is representable by a deterministic map. But the map obtained by **regressing on
> reward-selected samples** converges to a **conditional central tendency**, not to the proximal **argmax**.

**⛔ RETRACTED (2026-07-27): the "variance identity".** We previously wrote that the shortfall *equals* the dose,
i.e. that conditional variance is simultaneously the harvestable tilt and the irreducible error. That is
**dimensionally wrong** — the dose scales as c_M·σ (linear), the irreducible error as σ² (quadratic), so they
cross at a single point rather than being one quantity. Do not restate it.

**✅ WHAT REPLACES IT — reward curvature (a Jensen gap), and it is now measured.** How much of the tilt survives
the central-tendency step is set by the **curvature of the reward over the winner cloud**, not by the variance
alone. Smooth rewards survive averaging; conjunctive / thin-band rewards can invert, because averaging four
*correct-count* images yields a *wrong-count* image. Measured on the same candidates (job 88069):

| reward | MAP dose (survives averaging) | |
|---|---|---|
| **VQAScore** — smooth continuous likelihood | **+0.0421** [+0.0316, +0.0529] | ~50% survives |
| **CompBench primary** — BLIP-VQA + hard UniDet thresholds | **+0.0127** [−0.0011, +0.0263] | **CI spans 0** |

Per-category on the primary: smooth BLIP categories retain the tilt (shape **+0.0356** sig, color +0.0268,
texture +0.0188); hard-thresholded ones collapse (numeracy +0.0056, complex +0.0048, spatial −0.0170).

This single mechanism predicts the **sign** of the Phase-G result, its **category profile** (worst in counting),
and **Phase F's** binding-vs-counting dissociation — while remaining consistent with **Exp-0b** (+0.0717), where
selection helps at the *sample* level on the very same metric. The map step is what destroys it.

**Attribution:** P-GRAFT's **Lemma 3.3** already proves the monotone-conditional-variance half and measures it on
a VQA reward (App. E.1.2). Do **not** claim that component is unstated. What is ours is the *measured* contrast
between a smooth and a conjunctive reward on the same candidate sets.

**Precision on our own estimator.** The loss is pseudo-Huber on the *norm*,
`d = sqrt(‖Δ‖² + c²) − c` with `c ≈ 0.138`, while observed ‖Δ‖ is O(50). In that regime `d ≈ ‖Δ‖ − c`, and
minimising `E‖f(z) − Y‖` gives the conditional **geometric median**, not the mean.

State this **neutrally**: the identity is unchanged, and under best-of-N selection the two central tendencies
shift within a few percent of each other, so **no part of the argument's magnitude depends on which one it is.**
Concretely, for n=4 standard normals E[max] = 1.0294σ while the median of the max-of-4 law sits at
F(x)⁴ = 0.5 ⇒ F(x) = 0.8409 ⇒ x = 0.9981σ — a **3.0% difference** (verified numerically). Do **not** claim the
median's tail-robustness makes the obstruction "sharper": that would matter if the tilt were an upper-tail
reweighting, but a hard best-of-N selection shifts and compresses the whole distribution, so both statistics
track it. A referee who runs the order-statistic calculation will find "sharper" unsupported, and the argument
does not need it.

**Consequence propagated to the measurement.** Because the estimand is the geometric median, the ceiling
estimator aggregates group winners by **Weiszfeld geometric median**, not arithmetic mean — otherwise the
decisive number would compute a different object from the one this paragraph names. With 8 points in ~65k
dimensions the two are close but not identical, and the emitted conclusion has a boundary at 0.010, so the
report prints the arithmetic-mean aggregation alongside as an auditable cross-check.

## 3b. Disclose the two normalisations together

Because `d ≈ ‖Δ‖` rather than `‖Δ‖²`, the gradient w.r.t. the student output is the **unit** vector
`Δ/‖Δ‖` — constant magnitude regardless of residual size. Stacked on gradient clipping (which fired on
essentially every step), the optimiser sees a **doubly-normalised** signal: the target gap sets the update
*direction* and contributes nothing to its *magnitude*.

This does **not** threaten C1. The invariance argument is about the population optimum, which does not depend
on the loss's scaling, and both arms are treated identically. But it does mean **"we delivered a +0.124
supervision gap" is a statement about targets that is two normalisations away from being a statement about
parameter movement.** Disclose both halves in the same sentence — the clipping caveat and the loss-scaling
caveat — so the pair is presented together rather than discovered separately.

## 4. The blur penalty IS the Thm 4.1 gap

The third quantity in `measure_ceiling.py` is not a nuisance diagnostic. The proximal map picks an argmax under
a proximity penalty; regression picks a central tendency; and

  `r(mean of non-winners) − r(single non-winner)`

is the empirical estimate of how far apart those two objects are at that σ. **The theory paragraph and the
third number are one thing viewed twice** — which is what makes a short theory paragraph load-bearing rather
than decorative.

## 5. The constructive closure (state it; do not run it)

If the obstruction is that regression is **central-tendency-seeking** while the tilt sits at a **mode/argmax**,
the escape is an objective that is not central-tendency-seeking. That is exactly what every working recipe
uses:

- **DMD-family reverse KL** — mode-seeking by construction. Continuous-Time Distribution Matching
  ([2605.06376](https://arxiv.org/abs/2605.06376)) complains about precisely this property, describing reverse
  KL as inherently mode-seeking and biasing the student toward dominant modes.
- **Adversarial distillation** — likewise not a pointwise regression.
- **Reward gradients** (RTDMD, LaSRO) — gradient-following rather than estimating a central tendency.
- **GRAFT** — generative-from-noise training on a filtered dataset, i.e. distribution matching rather than
  pointwise regression, which is why its target-channel selection *does* transfer.

Stated as one sentence:

> Supervised regression on reward-selected samples cannot reach the tilted pushforward, because it targets a
> conditional central tendency while the tilt sits at a proximal argmax; the shortfall equals the conditional
> variance, which is also the only dose selection can harvest. The recipes that work — DMD-family reverse KL,
> adversarial distillation, reward gradients — are all mode-seeking or gradient-following rather than
> central-tendency-seeking.

**Falsifiable prediction (we do not run it):** the same selection signal delivered under a mode-seeking
objective should transfer. GRAFT's positive is already evidence for it.

This is what makes the negative generative rather than terminal: it does not merely close a door, it says
which doors are load-bearing and why.
