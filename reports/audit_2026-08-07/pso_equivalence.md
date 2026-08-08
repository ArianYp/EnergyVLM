# Our objective and the offline PSO equation are the same equation

**Roadmap task O.** The audit asked for an independent implementation of the published PSO equation
for fixed offline pairs, as the closest mandatory baseline. Writing it out shows there is less to
implement than expected, and that is itself the result.

## The reduction

PSO/Diffusion-DPO/D3PO share the reference-relative Bradley–Terry form

\[
\mathcal L
=-\log\sigma\!\left(
\beta\left[
\log\frac{p_\theta(\tau^+\mid c)}{p_0(\tau^+\mid c)}
-\log\frac{p_\theta(\tau^-\mid c)}{p_0(\tau^-\mid c)}
\right]\right).
\]

Under the Gaussian transition/constant-variance bridge already stated in
`reports/pso_d3po_reference_audit.md`,

\[
\log p_\theta(u\mid z_t,c)-\log p_0(u\mid z_t,c)\;\propto\;-\bigl(e_\theta(u)-e_0(u)\bigr),
\]

so with \(\ell^\pm=-(e_\theta^\pm-e_0^\pm)\),

\[
\beta\bigl(\ell^+-\ell^-\bigr)
=-\beta\bigl[(e_\theta^+-e_\theta^-)-(e_0^+-e_0^-)\bigr]
=-\beta\bigl(\Delta_\theta-\Delta_0\bigr),
\]

which is **exactly** the audited Phase-I logit. `phaseI/train_preference.py` now computes the
per-branch log ratios explicitly and forms the logit as \(\beta(\ell^+-\ell^-)\); this was verified
bit-equivalent to the previous expression over 2,000 random inputs.

So the Phase-I objective is not merely "a member of the PSO family": under the stated bridge it *is*
the offline PSO equation. The claim to retire is stronger than previously written — there is no
surviving loss-level difference to defend.

## What actually differs from PSO, and by how much

| dimension | PSO (bundled DMD2 config) | ours | still a real difference? |
|---|---|---|---|
| loss form | reference-relative logistic | identical under the bridge | **no** |
| ratio clipping | \(\epsilon=0.1\) | none | **inert — see below** |
| pair noise | separately sampled trajectories | shared corruption noise | **yes** |
| preservation | reference ratio only | explicit \(M1\) function-space anchor (+ optional REPA) | **yes** |
| \(\beta\) | 50 | 100 (uncalibrated; audit says under-driven) | yes, but see the clipping note |
| pair source | on-policy student samples | cached frozen-teacher endpoints | **yes** |

### The clipping is inert at our scale

PSO's \(\epsilon=0.1\) clips the log ratio to \([-0.1054, +0.0953]\). Measured on the smoke run's
per-example telemetry, the observed \(|e_\theta-e_0|\) has median \(2.9\times10^{-4}\) and **0%** of
examples fall outside the clip window. The clipping branch would therefore never fire for a
rectified-flow student anchored this close to its reference. It is implemented
(`--ratio_clip`, default 0) for completeness and to make the comparison explicit, not because it
changes anything.

This also reframes \(\beta\): PSO can afford \(\beta=50\) *with* clipping as a safety net; we have no
active safety net, and the audit separately shows gradient clipping fires on 100% of steps, so the
step size is set by `grad_clip` rather than by \(\beta\) in either case.

## The baseline arm to run

Given the reduction, the honest "offline PSO baseline" is our trainer with PSO's choices substituted
one at a time. The bundled-config point is:

```
--labels fixed --pair_source fixed_indices --label_source correct_prompt \
--noise_coupling independent --beta 50 --lambda_anchor 0 --ratio_clip 0.1
```

launchable as `PHASEFP_NOISE_COUPLING=independent PHASEFP_BETA=50 PHASEFP_ANCHOR=0
PHASEFP_RATIO_CLIP=0.1` through `ablations/phaseFP_train.lsf`. Because the loss form is shared, this
arm isolates **anchor + noise coupling + β**, which is exactly the set of surviving differences —
a cleaner ablation than a separately written "PSO implementation" would have produced.

It is queued behind the four decisive orientation arms rather than run alongside them: the
orientation question determines whether the paper needs this baseline in its main table or its
appendix, and the reservation has limited concurrent capacity.

## Consequence for the write-up

The paper must say that the objective is the offline PSO/Diffusion-DPO equation applied to a
rectified-flow student, with an added function-space anchor and shared pair noise. The contribution
is the controlled channel measurement, the fixed-pair counterfactual identification, and the
alignment–fidelity–diversity characterisation — not the loss, and not a "compatibility variant" of
the loss either.
