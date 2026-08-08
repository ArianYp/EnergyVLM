# Task N verdict: shared-noise coalescence is refuted as the mechanism

**Status: the preregistered falsification fired.** Recorded 2026-08-07 from LSF jobs 101193
(`correct`, shared noise) and 101234 (`correct`, independent negative-branch noise), 12,000
per-example telemetry rows each.

## What was predicted

The shared-noise construction gives

\[
z^+-z^-=(1-\sigma)(x^+-x^-),
\qquad
u^+-u^-=x^--x^+ .
\]

As \(\sigma\to1\) the two inputs coalesce while their targets stay separated, which was the proposed
reason the preference signal should be strongest at the student's first, high-noise step.

The sharp prediction is a **difference between conditions**, not a trend on its own. Under
independent noise the two branches are corrupted by different draws, so they do *not* coalesce as
\(\sigma\to1\). If coalescence is the mechanism, the \(\sigma\)-growth must weaken or disappear
there.

## What happened

Scale-normalised \(|{\Delta_\theta-\Delta_0}|\) divided by the branch error scale — the quantity that
removes the confound that flow errors are simply larger at high \(\sigma\):

| \(\sigma\) band | shared | independent | independent − shared |
|---|---:|---:|---:|
| [0.0, 0.2) | 0.0102 | 0.0099 | −0.0003 |
| [0.2, 0.4) | 0.0150 | 0.0143 | −0.0007 |
| [0.4, 0.6) | 0.0198 | 0.0195 | −0.0003 |
| [0.6, 0.8) | 0.0229 | 0.0242 | +0.0013 |
| [0.8, 1.0) | 0.0410 | 0.0431 | +0.0021 |

| statistic | shared | independent |
|---|---|---|
| corr(\(\sigma\), normalised \|gap\|) | +0.2210 [+0.2085, +0.2336] | +0.2260 [+0.2140, +0.2378] |
| highest / lowest band ratio | 4.02× | 4.37× |
| overall mean gap | −0.00739 | −0.00652 |

The two profiles are indistinguishable. The confidence intervals on the correlation overlap, and at
high \(\sigma\) the independent condition is if anything *marginally stronger*.

The \(z^+\) branch is bit-identical between the two conditions by construction — the extra noise
tensor is drawn after all other RNG consumption — so this is a matched comparison in which only
\(z^-\) differs.

## Conclusion

**The \(\sigma\)-dependence is real, large, and has nothing to do with input coalescence.** It
survives intact when coalescence is removed, so it is a property of the rectified-flow
parameterisation rather than of the shared-noise pair construction.

Per `phaseFP/PREREGISTRATION.md`, shared-noise coalescence is therefore **removed from the paper's
claims**, and the paper is written without a mechanism rather than with a speculative one. The
\(\sigma\)-profile itself remains reportable as a measured property of the objective; it just does
not license the coalescence story.

This also retires the "analytically interesting high-noise limit" framing carried in the handoff's
list of safe statements. The algebra is still correct; what fails is the inference from it to the
training behaviour.

## Both preregistered conditions have now resolved

1. **Shared noise does outperform independent noise on the held-out endpoint.**
   `CorrectFixed` − `CorrectIndepNoise` on the CompBench primary is
   **+0.01413 [+0.00656, +0.02184]** (absolute 0.55697 versus 0.54284), from evaluations 101257 and
   101261. By family: BLIP-VQA +0.0180 [+0.0094, +0.0269]; UniDet +0.0149 [−0.0006, +0.0305],
   marginally spanning zero; 3-in-1 +0.0002, null.
2. **The \(\sigma\)-profile is satisfied but non-diagnostic**, as shown above.

So the design choice is mildly beneficial — about a quarter of the correct-versus-random orientation
effect — while its proposed explanation is refuted. Per the preregistration, a positive (1) with a
non-diagnostic (2) must **not** be reported as support for coalescence.

## A more parsimonious account, consistent with everything measured but not itself tested

Shared corruption noise is a **common-random-numbers** construction: because \(z^+\) and \(z^-\) are
corrupted by the same draw, the noise contribution partially cancels in the difference
\(e^+_\theta-e^-_\theta\) that the loss consumes. This is the classic paired-sampling variance
reduction, and it predicts a lower-variance gradient estimate with **no particular \(\sigma\)
signature** — which is exactly the pattern observed.

Measured over 3,000 steps per arm:

| quantity | shared | independent | ratio (indep / shared) |
|---|---:|---:|---:|
| SD of the loss gap | 0.02278 [0.02050, 0.02535] | 0.02622 [0.02117, 0.03271] | 1.33× variance |
| mean pre-clip gradient norm | 86.48 | 110.94 | — |
| variance of pre-clip gradient norm | — | — | 1.72× |

The gradient-norm evidence is the cleaner of the two; the bootstrap intervals on the gap SD overlap,
so that comparison is suggestive rather than conclusive on its own.

This interacts with a finding from the zero-training audit: gradient clipping fires on **100%** of
steps, so the update is a normalised direction with a fixed step size. Lower gradient *variance*
therefore does not change the step length — it makes the step *direction* more stable across
updates, which is a coherent route to the observed endpoint benefit.

**Status: hypothesis, not result.** No controlled test of the variance-reduction account has been
run. It is offered because it explains all three observations at once (endpoint benefit, absent
\(\sigma\)-signature, lower gradient variance) where coalescence explains only the first. The
write-up should state the measured effect, state that coalescence is refuted, and mark variance
reduction as the leading untested explanation rather than a conclusion.

## Side observation, independent of the above

The winner share stays in the range 0.40–0.46 across every \(\sigma\) band in **both** conditions —
always below 0.5. The repulsion-dominated behaviour reported in
`reports/audit_2026-08-07/README.md` §5 is therefore uniform in noise level and independent of the
noise coupling. It is a separate finding from the \(\sigma\)-profile and does not fall with it.
