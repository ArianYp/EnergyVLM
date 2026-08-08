# Phase M — canonical continuous-time consistency baseline (sCM) on SD3.5-M

**Locked 2026-08-08, before any baseline weights exist.** Roadmap task M.

## Why this exists

Every preference result in this project is trained on top of **M1**, a *custom* CM/LCM-inspired
consistency student. If M1 is a weak base, two things are in doubt: the headline alignment numbers,
and — more seriously — whether the orientation effect would replicate on a competently distilled
student. A reviewer is entitled to ask whether we compared preference training against a strawman.

This phase answers only the first question: **is M1 a fair stand-in for a canonical modern few-step
method under a matched budget?** The second question (does the orientation result survive a change of
base student) is a separate, larger experiment and is *not* claimed here.

## What is being implemented, and what is not

**sCM** — the continuous-time consistency model of arXiv:2410.11081 (*Simplifying, Stabilizing and
Scaling Continuous-Time Consistency Models*).

We use the **rectified-flow-native** form of the loss from NVIDIA's rCM release
(`third_party/rcm`, Apache-2.0, `rcm/models/t2v_model_distill_rcm.py`), because SD3.5-M already
predicts a velocity and needs no TrigFlow change of variables:

\[
g=-\bigl(v_\theta^{\text{sg}}-v_{\text{teacher}}\bigr)-r\,t\,\partial_t v_\theta,
\qquad
\hat g=\frac{g}{\lVert g\rVert_2+0.1},
\qquad
\mathcal L_{\text{sCM}}=\bigl\lVert v_\theta-v_\theta^{\text{sg}}-\hat g\bigr\rVert_2^2 ,
\]

with \(r\) the tangent-warmup ratio rising linearly from 0 to 1. rCM also ships
`RectifiedFlow_TrigFlowWrapper` giving the exact RF↔TrigFlow map, retained as a cross-check.

**Not implemented: the "r" in rCM** — the score-regularization term, which requires a second
trainable fake-score network and roughly triples cost. Consequence to state plainly in the paper:
**rCM would likely be stronger than the baseline we run**, so a finding of "M1 ≈ sCM" does not
license "M1 ≈ state of the art".

## Fairness conditions

Held identical to M1 by construction:

| dimension | value |
|---|---|
| teacher | frozen SD3.5-M, identical checkpoint |
| student initialization | SD3.5-M weights (M1's own init) |
| resolution | 512 px |
| teacher guidance for targets | CFG 7.0 |
| training prompts | the same T2I-CompBench train split M1 used |
| inference | 4 steps, CFG 1.0 |
| evaluation | the frozen, hash-asserted manifests in `phaseFP/eval_pool_101203` |

**The baseline is given advantages M1 never had**, deliberately, so that any deficit cannot be
attributed to our tuning:

1. **Its own hyperparameters.** There are no published sCM settings for SD3.5-M, so we run a small
   learning-rate search ({1e-6, 3e-6, 1e-5}) at 2,000 steps and promote the best by held-out
   CompBench. M1 received no such search — it used a single configuration.
2. **Paper-recommended internals**: tangent normalization \(c=0.1\); tangent warmup over the first
   10k iterations; logit-normal timestep proposal; identity \(c_{\text{noise}}\).
3. **Both budget definitions reported** (below), rather than whichever flatters us.

## Budget: report both, because JVP changes the exchange rate

M1 used **20,000 optimizer steps** in **11.2 h** on 4 GPUs. An sCM step costs more because of the
JVP. Reporting equal-steps alone would penalise sCM's wall-clock; reporting equal-hours alone would
penalise its step count. Both arms are trained and both reported:

- **M-steps**: 20,000 optimizer steps, however long that takes.
- **M-hours**: as many steps as fit in 11.2 h × 4 GPUs.

If the two agree, the conclusion is robust to the accounting choice. If they disagree, that
disagreement *is* the finding and is reported as such.

## Matched-NFE reference points

The comparison table also carries, on the same prompts and manifests:

| system | NFE per returned image |
|---|---:|
| sCM student, 4 steps, CFG 1 | 4 |
| M1 student, 4 steps, CFG 1 | 4 |
| unmodified SD3.5-M teacher, 4 steps, CFG 7 | 8 |
| unmodified SD3.5-M teacher, 8 steps, CFG 7 (`TeacherUniform`, already measured) | 16 |

The 4-step teacher is added because it is the honest same-step reference; the existing 8-step number
is not.

## Preregistered interpretation

Primary endpoint: equal-category-weighted official CompBench primary at 4 steps, paired prompt
bootstrap, on `phaseFP/eval_pool_101203`. Co-primary: GenEval2 Soft-TIFA.

| outcome | conclusion |
|---|---|
| sCM ≈ M1 (interval contains 0) | M1 is a fair stand-in; the preference results are not an artifact of a weak base |
| sCM > M1 | our base is weak. The orientation contrast **must** be re-run on the sCM student before the headline claims stand |
| sCM < M1 | report as *"we could not reproduce an advantage for sCM at this scale under matched budget"* — **not** as "sCM is worse". A negative result about someone else's method, produced by our own reimplementation, is weak evidence and will be labelled so |

Three seeds for whichever arm is promoted, matching the standard already set in Phase FP.

## Correctness gates before any training run

1. **JVP validity.** The tangent must match a finite-difference estimate to <5% relative error
   (`phaseM/probe_jvp_sd3.py`). PyTorch's fused attention backends have no forward-mode rule, so the
   math backend is forced; if that proves too slow or memory-hungry, the fallback is porting rCM's
   Triton flash-attention JVP kernel, and that decision is recorded rather than silently taken.
2. **Loss sign.** At initialization the student equals the teacher, so \(g\) reduces to
   \(-r\,t\,\partial_t v_\theta\) and the loss must be finite and non-zero.
3. **Teacher parity.** The sCM trainer must reproduce M1's teacher rollout bit-for-bit given the
   same seed, so the two students differ only in objective.
4. **No evaluation contamination.** The eval manifest hashes are asserted, as in every Phase-FP job.

## Threats to validity, stated in advance

- **Reimplementation risk.** This is our code, not the authors'. A poor result may reflect our
  implementation rather than the method. Gate 1 and the lr search mitigate but do not remove this.
- **No published SD3.5-M reference.** Nothing external tells us what sCM *should* reach on this
  backbone, so there is no target to validate against.
- **sCM, not rCM.** The stronger method is not being run; see above.
- **Attention backend.** Forcing the math SDPA backend changes numerics slightly relative to M1's
  fused-kernel training. Reported, and the teacher rollout is verified identical.
