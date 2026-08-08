# Zero-training audit (roadmap task E) — findings

**Run on:** 2026-08-07. **No GPU training. No new checkpoints.**
**Scripts:** `audit/*.py`. **Artifacts:** this directory plus `reports/registry/`.

Everything below is recomputed from raw JSON/JSONL. Where a number disagrees with earlier prose, the
recomputation wins.

---

## 1. The unexplained fidelity shift is explained: different generation prompt pools

The handoff recorded an "unexplained" cross-job FID/CMMD shift and made it a blocking correction.
It is not an anomaly, a seed effect, or a preprocessing difference.

| job | generation prompts | reference | M1-family FID | precision |
|---|---|---|---:|---:|
| 99955 | **T2I-CompBench compositional prompts** (`compbench/images`) | 2,398 COCO val2017 reals | 68.1 | 0.40 |
| 100593 | **COCO captions** (`fidelity/images`, `build_eval_pool.py coco`) | 2,398 COCO val2017 reals | 43.6 | 0.67 |

`ablations/phaseI_eval.lsf` line 98 passes `--gen_root "$ROOT/compbench/images"`;
`ablations/phaseI_counterfactual_eval.lsf` line 108 passes `--gen_root "$ROOT/fidelity/images"`.
The reference construction is identical in both (`n_ref 2398`, same seed, same square crop). Images
generated from COCO captions are simply far closer to the COCO real distribution than images
generated from short synthetic templates such as *"a white piano and a black bench"*.

**Consequences.**

- The two jobs' fidelity numbers must never be compared, and the ~24-point gap needs no further
  investigation. Blocking correction 3 is closed.
- Job 99955's numbers are a *prompt-mismatched* FID. They remain internally valid across its four
  arms (identical prompts) but are not a standard FID and should be labelled as such.
- Only job 100593's protocol (COCO captions vs COCO reals) is the standard matched-caption design.
  Every future fidelity measurement uses it.

## 2. The prompt key `idx` is not stable across jobs

`idx` is assigned per evaluation job. In `phaseC/eval_scores` colour `idx 0` is *"a white piano and a
black bench"*; in `exp0/primary_headroom/scores` colour `idx 0` is *"a blue backpack and a green
bottle"*. Any cross-job analysis joined on `idx` silently compares different prompts.

All audit scripts join on prompt **text**. Two evaluation jobs were then checked explicitly:

- `phaseI/eval_99955` and `phaseI/counterfactual_eval_100593` share **2,398/2,398** CompBench
  prompts *and* an identical `idx`→prompt map, so cross-job contrasts between them are valid.
- `phaseC/eval_scores` and `exp0/primary_headroom/scores` share **0** prompts.

## 3. Amortization ratio: a real interval, and why no joint bootstrap exists

Because the C1 transfer contrast and the teacher-oracle contrast share zero prompts, the requested
"matched or joint bootstrap" is impossible with the current artifacts. Disjointness does make the
two estimates independent, so a ratio interval from independent bootstraps is legitimate.

| quantity | contrast | prompts | Δ | 95% CI |
|---|---|---:|---:|---|
| numerator | B4 − B2 @ 4 steps | 2,098 | +0.00045 | [−0.00889, +0.00975] |
| denominator | teacher best-of-4 − random @ 8 steps | 1,050 | +0.07172 | [+0.05709, +0.08655] |

Both counts are **primary-category** prompts (the seven categories that enter the equal-weighted
primary). The CompBench pool is 2,398 prompts, but `non_spatial` is CLIPScore-scored and is reported
as secondary rather than pooled, so it does not enter either contrast.

**Ratio: 0.0062, 95% CI [−0.126, +0.139]; one-sided 95% upper bound 0.117.**

The handoff's shortcut (0.0097/0.0717 ≈ 0.135) divides an interval endpoint by a point estimate and
is neither a bound on the ratio nor a confidence statement about it. Cite **11.7%** as the one-sided
95% upper bound, with the stated caveat that numerator and denominator describe different prompt
populations. Closing that caveat needs the teacher oracle re-estimated on the same held-out split
(roadmap task T) — not a different statistical procedure.

## 4. CompBench by evaluator family: the anti-circularity evidence holds

Paired prompt bootstrap, 10,000 draws, prompts resampled within category, family effect = equal-
weighted mean of its category means (matching the preregistered primary).

| contrast | pooled primary | BLIP-VQA | UniDet | 3-in-1 |
|---|---:|---:|---:|---:|
| correct − counterfactual construction | +0.04922 | +0.0405 [+0.0301,+0.0513] | **+0.0689 [+0.0524,+0.0860]** | +0.0161 |
| preference − placebo | +0.04587 | +0.0608 [+0.0490,+0.0728] | **+0.0376 [+0.0205,+0.0547]** | +0.0260 |
| preference − M1 | +0.04196 | +0.0583 | +0.0315 [+0.0154,+0.0477] | +0.0243 |
| correct − M1 *(cross-job)* | +0.06446 | +0.0789 | **+0.0620** | — |
| counterfactual − M1 *(cross-job)* | +0.01524 | +0.0383 | **−0.0070 (null)** | — |
| REPA increment | +0.00833 | **+0.0169 [+0.0093,+0.0249]** | **+0.0017 [−0.0122,+0.0159] (null)** | +0.0025 (null) |
| placebo − M1 | −0.00390 | −0.0025 (null) | −0.0061 (null) | −0.0017 (null) |

Three results follow.

**(a) The headline effects are not confined to VQA-style scoring.** UniDet is a detection-based
family sharing no architecture with the CLIP-FlanT5 selector that produced the training labels, and
its interval excludes zero for every headline contrast. For correct-versus-counterfactual the
detection family moves *more* than the VQA family: BLIP − UniDet = **−0.0284 [−0.0488, −0.0086]**.
The handoff flagged this as needing a paired bootstrap before it could be cited; it now has one and
it survives.

**(b) The counterfactual arm's residual gain is VQA-family-only.** Against M1 it gains +0.0383 on
BLIP-VQA but is null on UniDet (−0.0070). The correct arm gains on both (+0.0789, +0.0620). This is
a clean dissociation obtained with no new training: whatever prompt-independent quality prior the
counterfactual labels carry shows up in VQA-style scoring but not in detection-based compositional
scoring, while the prompt-specific signal shows up in both.

**(c) The REPA semantic increment does not replicate across families.** Its +0.00833 is carried
entirely by BLIP-VQA; UniDet and 3-in-1 are null. This supports the roadmap's instruction to retire
"REPA improves compositional alignment" and keep REPA as a fidelity regulariser.

**(d) The placebo is a clean negative control.** Randomized-label training is null against M1 on
every family, so the preference effect is not a generic consequence of extra training.

## 5. β = 100 is under-driven, and the loss works by pushing away from the loser

Recovered from the completed Phase-I runs' local W&B datastores (`audit/wandb_history.py`), 301
logged steps per run.

**Where β = 100 puts the sigmoid.** The loss is `−log σ(−β(Δ_θ − Δ_0))`.

| run | median \|logit\| | frac \|logit\|>4 (saturated) | frac \|logit\|<0.5 (inert) | responsive |
|---|---:|---:|---:|---:|
| preference | 0.268 | 0.060 | 0.450 | 0.490 |
| placebo | 0.023 | 0.017 | 0.623 | 0.360 |
| correct_matched | 0.285 | 0.077 | 0.423 | 0.500 |
| counterfactual | 0.181 | 0.027 | 0.557 | 0.417 |

β = 100 is not saturating; it is **too small**. Roughly half of all steps sit in the near-linear
region where the preference term contributes almost nothing beyond a constant. Median |Δ_θ − Δ_0| is
≈ 0.0057, so putting the median logit at 1 needs β ≈ 175 and at 2 needs β ≈ 350.

**Which branch moves?** Both quantities are measured against the frozen M1 reference.

| run | winner error improvement | loser error increase | winner share of gap change |
|---|---:|---:|---:|
| preference | **−0.00778** | **+0.01461** | 0.403 |
| placebo | −0.00408 | +0.00461 | 0.477 |
| correct_matched | −0.00928 | +0.01834 | 0.431 |
| counterfactual | −0.00467 | +0.00818 | 0.449 |

The winner improvement is **negative**: relative to M1 the student gets *worse* at reconstructing the
preferred teacher endpoint too. The objective is satisfied because it gets worse on the rejected
endpoint roughly twice as fast. About 60% of the achieved margin comes from degrading the loser.

This is a repulsion-dominated mechanism, not an attraction toward preferred outputs, and it is a
plausible proximate cause of the observed DINO diversity collapse: an objective that mainly pushes
probability away from specific outputs removes modes rather than adding them. It also explains why
the M1 anchor is load-bearing — it is the only term resisting the winner-side degradation. The
placebo shows the same degradation pattern but symmetric (0.477 share), which is exactly the null
signature.

**Gradient clipping is saturated.** Median pre-clip gradient norm is 50–68 against `grad_clip 1.0`,
and the clipping rate is **1.000** in every run: every single logged step is clipped. The effective
update is therefore a normalised direction with a constant step size, so β does **not** scale the
update magnitude — it only changes the direction mix between the preference and anchor gradients.
Any β sweep must be interpreted that way, and reporting the clipping rate is mandatory.

## 6. Fixed-pair cache (roadmap tasks C and D)

Built by `phaseFP/build_fixed_pair_selection.py` in LSF job **101185** →
`phaseFP/fixedpair_101185`, W&B run `phaseFP_cache_101185`. CPU only: the completed counterfactual
cache already stores both score vectors over the same four candidates, so no teacher regeneration
was needed.

All gates pass, and `phaseFP/test_fixed_pair.py` proves cross-arm identity from the raw indices,
not merely from a hash.

| diagnostic | value |
|---|---:|
| records | 5,042 |
| fixed pairs that reverse under the edited text | 1,373 (27.23%) |
| mean original-prompt pair margin | +0.2865 |
| counterfactual ties on the fixed pair | 4 (kept as +1) |
| random arm balance | exactly 0.5 |

**Effective dose ladder.** The arms differ only in the sign applied to a fixed pair, and their
assigned original-prompt margins give a monotone, preregisterable dose:

| arm | records with a different preferred image | mean assigned margin | contrast vs correct |
|---|---:|---:|---:|
| correct | 0.000 | +0.2865 | 0.0000 |
| counterfactual | 0.272 | +0.1783 | 0.1082 |
| random | 0.500 | −0.0002 | 0.2867 |
| inverted | 1.000 | −0.2865 | 0.5729 |

Two design facts follow.

- **On the reversal subset the counterfactual arm *is* the inverted arm** (both give −0.1986 mean
  assigned margin there). The sharp orientation-only test is therefore already available at maximal
  dose as the full-population correct-versus-inverted contrast; it does not need a fifth arm.
- **The counterfactual contrast is the weakest of the three** (0.108 versus 0.287 for random). If
  the dose response is roughly linear in assigned margin, the Phase-I preference-minus-placebo
  effect of +0.0459 scales to a predicted fixed-pair correct-minus-counterfactual effect near
  **+0.017**, against a typical CI half-width of ±0.009. That is detectable but not comfortably so,
  which is why the preregistered primary analysis is the **monotone trend across all four arms**
  rather than any single pairwise contrast.

**Reversals are not uniformly distributed.** They concentrate on near-ties — mean original margin
0.1986 on reversed records versus 0.3193 on non-reversed (difference −0.1207) — and vary sharply by
category: spatial 57.7%, colour 39.1%, non-spatial 32.7%, 3d_spatial 27.1%, complex 23.8%, texture
16.4%, numeracy 13.7%, shape 9.2%. Category-level heterogeneity must be reported; a pooled effect
alone would hide that the intervention barely touches `shape` and `numeracy`.

## 7. The diversity failure looks very different against the right reference

`audit/oracle_diversity_price.py`, LSF **101231**, 1,200 prompts from the frozen-teacher `cfg7_s8`
bank, four images per prompt per policy, using the same DINO and LPIPS estimators as the student
gate.

| policy | DINO diversity | LPIPS diversity | mean VQAScore | distinct images / prompt |
|---|---:|---:|---:|---:|
| teacher random | 0.37383 | 0.59087 | 0.74739 | 3.339 |
| teacher best-of-4 | 0.24711 | 0.43089 | 0.87530 | 2.388 |
| **price** | **−0.12672** [−0.13583, −0.11756] | **−0.15998** [−0.17142, −0.14815] | +0.12791 | −0.951 |

Set against the student numbers from job 100593 (400 held-out prompts, four seeds):

| quantity | DINO | LPIPS |
|---|---:|---:|
| correct-preference student, absolute | 0.31591 | 0.67166 |
| counterfactual student, absolute | 0.35727 | 0.68440 |
| student difference (the failed gate) | −0.04136 | −0.01273 |
| **teacher best-of-4 price** | **−0.12672** | **−0.15998** |

Two things follow.

1. **The student's diversity reduction is about one third of the selection policy's own price**
   (−0.041 versus −0.127 on DINO), and roughly one twelfth on LPIPS.
2. **The student is more diverse in absolute terms than the policy it amortizes** — DINO 0.316
   versus 0.247, LPIPS 0.672 versus 0.431.

So "diversity collapse" overstates it: the student did not concentrate to anything like the
best-of-4 oracle's degree. The preregistered gate still **failed**, because that gate was defined
against the counterfactual control arm, and a failed preregistered gate is not rewritten after the
fact. The correct write-up reports both comparisons and says plainly which was preregistered.

**Caveats, which are real.** The two measurements are not the same estimand and not the same
population: the teacher price is best-versus-random *selection over a fixed 8-candidate bank* on
1,200 `exp0` prompts at CFG 7 / 8 steps, whereas the student difference is *two trained models* on
400 held-out CompBench prompts at CFG 1 / 4 steps. They are directly comparable only in the sense
that both use identical DINO and LPIPS estimators. It helps that the teacher-random level (0.374)
sits close to the counterfactual student level (0.357), but the comparison should be described as
an order-of-magnitude reference, not a matched contrast. Making it matched requires the teacher
policy evaluated on the same 400 prompts (roadmap task T).

The best-of-8 row in the same job is a limiting sanity anchor rather than a policy: with K equal to
the bank size the argmax is deterministic, so all four draws return the same image (distinct images
1.010, DINO 0.0015). That the estimator collapses to zero exactly where it should is evidence the
measurement is behaving, not a result about a usable policy.

## 8. What remains open

- **Same-population teacher ceiling** (roadmap task T) — needed to remove the caveat on §3 and to
  turn the diversity comparison above into a matched one.
- **Citation audit** — `reports/related_work_2026_novelty_audit.md` and
  `reports/pso_d3po_reference_audit.md` cover PSO/D3PO; see also
  `reports/audit_2026-08-07/pso_equivalence.md`, which shows the objective *reduces to* the offline
  PSO equation. The BOND, Noise Hypernetworks and KL-versus-Wasserstein corrections are recorded in
  the handoff and must be applied in the write-up.
- **Citation audit** — `reports/related_work_2026_novelty_audit.md` and
  `reports/pso_d3po_reference_audit.md` already cover PSO/D3PO; the BOND, Noise Hypernetworks and
  KL-versus-Wasserstein corrections are recorded in the handoff and must be applied in the write-up.
