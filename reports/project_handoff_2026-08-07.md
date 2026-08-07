# Project handoff: evidence, lessons, and implementation plan

**Frozen on:** 2026-08-07

**Purpose:** source-of-truth context for the next implementation session.
**Scope:** consistency distillation, REPA, sample selection, Phase-I preference training, the MMDiT ranking
branch, the counterfactual campaign, the four external reviews, and the subsequent artifact and literature
audit.

## Executive decision

The project should be written as an **analysis and mechanistic study**, not as a new preference-optimization
method. The Phase-I loss is an offline, teacher-sourced, rectified-flow-compatible member of the
Diffusion-DPO/D3PO/Pairwise Sample Optimization family. Its value is as an experimental instrument.

The strongest surviving question is:

> When a frozen teacher exposes compositionally different outcomes for the same prompt, why does selecting a
> better trajectory fail to transfer through ordinary consistency regression, while an explicit comparative
> objective changes the fresh-sample distribution—and what fidelity and diversity price does that change
> incur?

Do not launch a new publication-scale campaign yet. First complete the zero-training audit, then run two
decisive pilots in parallel:

1. the **fixed-unordered-pair label intervention**; and
2. the **shared-noise mechanism falsification**.

The correct paper claim depends on their outcomes.

## The project in plain language

Stable Diffusion \(3.5\)-Medium can produce semantically different images from different sampling trajectories
for the same prompt. Some candidates satisfy the requested attributes, counts, and relations better than
others. Best-of-\(K\) selection proves that this variation contains useful compositional headroom, but it costs
\(K\) teacher generations at inference.

The project asks whether that per-prompt selection advantage can be **amortized** into one deterministic
four-step student:

1. **Consistency distillation** compresses the teacher into few sampling steps.
2. **REPA** aligns an internal diffusion-transformer representation with frozen DINOv2 image features and is
   intended to protect fidelity.
3. **Preference supervision** tells the student which of two teacher outcomes better matches the prompt.

The key distinction discovered by the experiments is that the reward can enter through different channels:
the sampled trajectory, the regression target, the initial noise, or the loss itself. These channels are not
equivalent.

## Mathematical definitions

### Consistency student

Let \(v_\theta(z_t,t,c)\) be the four-step student velocity, \(v_\xi\) the frozen teacher, and
\(\widetilde z_s\) a teacher-propagated state at a cleaner time \(s<t\). The audited \(M1\) baseline is a custom
CM/LCM-inspired consistency objective rather than a canonical implementation of \(s\mathrm{CM}\) or
\(r\mathrm{CM}\):

\[
\mathcal L_{M1}
=\mathbb E\!\left[
d\!\left(
f_\theta(z_t,t,c),
\operatorname{stopgrad} f_{\bar\theta}(\widetilde z_s,s,c)
\right)
\right],
\]

where \(f_{\bar\theta}\) is the EMA target and \(d\) is the repository's pseudo-Huber-style distance. The
objective contains no comparison among the \(K\) teacher candidates unless an additional preference term is
introduced.

### REPA branch

For a clean real image \(x\), frozen DINOv2 patch features \(\phi(x)\), a chosen MMDiT hidden state
\(h_\theta^{(\ell)}\), and trainable projector \(P\), the REPA term is

\[
\mathcal L_{\mathrm{REPA}}
=1-\frac{1}{N}\sum_{n=1}^{N}
\frac{P(h_{\theta,n}^{(\ell)})^\top\phi_n(x)}
{\lVert P(h_{\theta,n}^{(\ell)})\rVert_2\,\lVert\phi_n(x)\rVert_2}.
\]

REPA is currently supported as a possible fidelity regularizer, not as the primary semantic mechanism.

### Phase-I preference objective

For a preferred teacher endpoint \(x^+\), rejected endpoint \(x^-\), shared corruption noise \(\epsilon\), and
noise level \(\sigma\), the current implementation uses

\[
z^+=(1-\sigma)x^+ + \sigma\epsilon,
\qquad
z^-=(1-\sigma)x^- + \sigma\epsilon,
\]

with flow targets

\[
u^+=\epsilon-x^+,
\qquad
u^-=\epsilon-x^-.
\]

Define the dimension-normalized errors and their gaps

\[
e_\theta^\pm
=\frac{1}{D}\left\lVert v_\theta(z^\pm,\sigma,c)-u^\pm\right\rVert_2^2,
\qquad
\Delta_\theta=e_\theta^+-e_\theta^-,
\qquad
\Delta_0=e_0^+-e_0^-.
\]

The preference and total losses are

\[
\mathcal L_{\mathrm{pref}}
=-\log\operatorname{sigmoid}\!\left[-\beta(\Delta_\theta-\Delta_0)\right],
\]

\[
\mathcal L
=\mathcal L_{\mathrm{pref}}
+\lambda_{\mathrm{anchor}}\mathcal L_{M1}
+\lambda_{\mathrm{REPA}}\mathcal L_{\mathrm{REPA}}.
\]

The Bradley–Terry/DPO structure and reference-relative comparison are prior art. The mapping from a
rectified-flow error difference to a transition log ratio is a modelling surrogate that must be disclosed and
tested. The pilot value \(\beta=100\) was not derived from the literature and is not yet calibrated.

## Correct source-of-truth numbers

### Three different kinds of selection headroom

These numbers must never be merged:

| Intervention | Candidate generator and evaluation | Difference |
|---|---|---:|
| Teacher endpoint scoring inside the training cache | Frozen teacher, VQAScore, top-of-four minus candidate mean | \(+0.1220\;[+0.1191,+0.1250]\) |
| Teacher best-of-four at inference | Frozen teacher candidates, held-out official CompBench, best versus random | **\( +0.0717\;[+0.0573,+0.0865] \)** |
| Frozen-student noise best-of-four | Four independent \(M1\) initial seeds, held-out official CompBench, best versus random | **\( +0.0485\;[+0.0386,+0.0585] \)** |

The \(+0.0717\) result is the teacher best-of-four oracle in `exp0/primary_headroom/verdict.md`. The \(+0.0485\)
result is the Phase-H noise-channel oracle in `reports/phaseH_gate_report.md`. Earlier prose accidentally
presented the latter under a teacher-endpoint description. Every amortization ratio must be recomputed after
this correction.

### Phase C1: selected trajectories through ordinary consistency regression

| Comparison | Official CompBench difference | Interpretation |
|---|---:|---|
| selected teacher trajectory \(B4\) minus random trajectory \(B2\) | \(+0.0004\;[-0.0090,+0.0097]\) | no measurable independent transfer |
| control replicate \(B2'\) minus \(B2\) | \(+0.0007\;[-0.0084,+0.0096]\) | calibrated negative control, not a variance estimate |
| \(B4-B2\), in-objective VQAScore | \(+0.0142\;[+0.0075,+0.0207]\) | selector-specific movement without independent transfer |
| \(B4-B2\), GenEval2 | \(+0.0065\;[-0.0109,+0.0239]\) | null |

This is a strong empirical limitation for the audited objective and setting. It is not a theorem that
best-of-\(K\) selection can never be amortized.

A tempting approximation divides the upper CompBench effect bound by the teacher oracle point estimate:

\[
\frac{0.0097}{0.0717}\approx 0.135.
\]

Do not yet call this a formal \(95\%\) upper bound. The numerator and denominator were estimated on different
experimental populations. A matched or joint bootstrap is required. Using the oracle's lower confidence bound
as a crude conservative denominator gives \(0.0097/0.0573\approx0.169\).

### Phase G and Phase H: lessons from alternative channels

The Phase-G selected-target experiments produced real observations, but every proposed explanation based on
conditional means, reward curvature, target variance, or covariate-shift optimum invariance was later broken
or rendered non-discriminating by controls. Preserve the measurements; do not resurrect those mechanisms.

Phase H established that best-of-four selection among the student's own seeds has \(+0.0485\) of headroom, but
the raw noise is not predictively rankable by the shallow audited probes:

\[
r_{\mathrm{OOF}}(z,R)=-0.0123,
\]

which caps that particular learned-selector route at approximately \(+0.0015\). This does **not** refute Noise
Hypernetworks: shallow selection among fixed noises and end-to-end learned noise modulation are different
problems.

### Phase I: explicit pairwise preference

The matched placebo comparison used the same top and bottom images, starting checkpoint, data order, pair
noise, optimizer, and update count. The placebo randomized only the orientation.

| Four-step arm or contrast | CompBench | GenEval2 |
|---|---:|---:|
| frozen \(M1\) absolute | \(0.49482\) | \(0.21294\) |
| random-label placebo absolute | \(0.49092\) | \(0.20203\) |
| correct preference absolute | \(0.53679\) | \(0.25162\) |
| correct preference \(+\) REPA absolute | \(0.54511\) | \(0.25804\) |
| preference minus placebo | **\( +0.04587\;[+0.03660,+0.05483] \)** | **\( +0.04958\;[+0.02910,+0.06943] \)** |
| preference \(+\) REPA minus preference | \(+0.00833\;[+0.00140,+0.01540]\) | \(+0.00642\;[-0.01112,+0.02416]\) |

This establishes that the explicit comparative objective moves fresh samples relative to a randomized-label
control. It is one training seed per arm and does not establish method novelty or publication-grade variance.

The original fidelity run reported \(M1\) FID/CMMD \(68.12/66.14\), preference \(68.96/66.99\), and
preference \(+\) REPA \(67.99/66.30\). A later counterfactual evaluation reported much lower absolute values
despite matching prompt hashes. That cross-job shift is unexplained. Do not compare fidelity across those jobs
until the prompt manifest, real reference pool, preprocessing, seed, feature extractor, and sample count are
audited or \(M1\) is regenerated inside the same job.

### Evaluator-family evidence already present

CompBench is not one VQA evaluator:

- colour, shape, and texture use BLIP-VQA;
- spatial, three-dimensional spatial, and numeracy use UniDet-style detection;
- non-spatial uses CLIP;
- the complex category combines several components.

For preference minus placebo, the unbootstrapped family-level means are approximately

\[
\Delta_{\mathrm{BLIP}}\approx+0.0608,
\qquad
\Delta_{\mathrm{UniDet}}\approx+0.0376.
\]

For correct versus counterfactual pair construction they are approximately

\[
\Delta_{\mathrm{BLIP}}\approx+0.0405,
\qquad
\Delta_{\mathrm{UniDet}}\approx+0.0689.
\]

The second result is especially useful: the detection-family point effect is larger than the VQA-family point
effect. Compute paired family-level bootstrap intervals before using it as formal evidence. It partially
defuses, but does not eliminate, scorer-circularity concerns.

### Completed counterfactual campaign

The original and minimally edited prompts shared the four-candidate bank, but each text independently selected
its own top and bottom images. Across (5{,}042) records:

| Cache diagnostic | Value |
|---|---:|
| editable prompt coverage | \(90.70\%\) |
| best candidate changes | \(55.04\%\) |
| unordered top-bottom pair changes | \(60.75\%\) |
| ordered top-bottom pair changes | \(67.53\%\) |
| fixed original top-bottom pair reverses | \(27.23\%=1{,}373/5{,}042\) |
| some reversible pair exists | \(74.65\%\) |

The completed comparison is:

| Endpoint | correct construction minus counterfactual construction |
|---|---:|
| official CompBench | **\( +0.04923\;[+0.04069,+0.05805] \)** |
| GenEval2 | **\( +0.04100\;[+0.02412,+0.05824] \)** |
| FID | \(-0.388\), favorable point estimate |
| CMMD | \(-6.735\), favorable point estimate subject to the cross-job audit |
| DINO diversity | **\(-0.04136\;[-0.04794,-0.03472]\)**, preregistered gate failed |
| LPIPS diversity | \(-0.01273\), within the preregistered tolerance |

The correct causal estimand is the **total effect of the text-conditioned pair-construction policy**. Pair
identity is downstream of the labelling-text intervention, so the result is not simply discarded as
confounded. It does not, however, identify an orientation-only direct effect.

The fixed-pair experiment is required to make that second claim.

### MMDiT ranking branch

The frozen-MMDiT and trainable-fusion rankers select better images than random, but controlled correct-versus-
wrong/blank-prompt interventions are null on official CompBench. Decoded CLIP also recovered held-out
VQAScore ordering better than the fusion readout in the audited comparison. The rankers appear to exploit
generic image completeness or quality more than prompt-specific composition.

This branch should remain in the appendix as a useful negative supervision-identifiability result. It is not
abandoned as an architectural possibility—diffusion-native reward models show that MMDiT features can work—but
it is not the shortest route to the main paper.

### RTDMD audit

The released RTDMD configuration explicitly contains a GenEval reward and uses GenEval prompts. Its reward
stage, evaluated with our independent harness, improved the aggregate CompBench primary by
\( +0.0116\;[+0.0018,+0.0213] \) but reduced numeracy and reduced performance on its own GenEval counting prompt
distribution by \( -0.0592\;[-0.0842,-0.0342] \). This is a reward-composition failure, not a train-to-test
generalization failure.

An earlier claim that RTDMD moves SD(3.5)-M GenEval from approximately (0.50) to (0.95) was a
mis-citation. The SD\(3.5\)-M table contains no GenEval column; the located \(0.7722\rightarrow0.9046\) values
refer to FLUX.2 overall GenEval. The released-checkpoint RTDMD comparison is already our reward-plus-few-step
baseline and should be promoted in the paper instead of hidden.

## What the literature audit changed

1. **No new loss claim.** Diffusion-DPO, D3PO, and especially PSO cover the reference-relative pairwise
   objective family. PSO is the closest prior because it explicitly tunes one-to-four-step models.
2. **Offline versus online is a design difference, not an advantage.** Our pairs come from the frozen teacher;
   PSO ordinarily samples from the current student. Offline teacher pairs keep the causal instrument clean but
   introduce a support mismatch relative to the student's deployed four-step distribution. Keep offline as
   the primary causal design and add an online arm as robustness if compute permits.
3. **REPA is prior art for representation fidelity.** Our contribution can only be its controlled interaction
   with the preference channel, not the idea of representation alignment.
4. **The ranking head is not the novelty.** LRM, SLRM, and DiNa-LRM demonstrate stronger diffusion-native
   reward-model designs.
5. **Reward plus few-step distillation is crowded.** RTDMD, LaSRO, GRAFT/P-GRAFT, GORS, reward-guided
   distillation, PSO, and recent distribution-matching approaches prevent a broad method claim.
6. **Noise Hypernetworks is not contradicted by Phase H.** It learns to modulate noise end to end; Phase H
   tested shallow pre-selection from raw noise features.

The local PSO clone is at commit `85e09df883a93782b64379097489abd9ba85d50a` and has no license file, so it
must remain a read-only reference. The D3PO clone is at commit `40d31758048f62ff49814eaa0403d820493ad374`
and is MIT licensed. Neither is a drop-in implementation for SD(3.5)-M rectified flow.

## Theory status: what survives and what is retracted

### Safe statements

- The three reward-entry channels we tested are empirically different under the audited objectives.
- Selected-trajectory consistency regression yielded no measurable independent benefit despite large
  candidate-level headroom.
- Explicit pairwise relative-error training changed the fresh-sample distribution.
- Shared noise creates an analytically interesting high-noise limit:

\[
z^+-z^-=(1-\sigma)(x^+-x^-),
\qquad
u^+-u^-=x^- - x^+.
\]

  As \(\sigma\rightarrow1\), the two inputs coalesce while their targets remain distinct. This is a falsifiable
  hypothesis for why competition is strong at the student's first, high-noise step.

### Retracted or prohibited statements

- A deterministic generator cannot represent a reward-aligned distribution.
- Reweighting teacher-consistent trajectories leaves the optimum invariant in our misspecified four-step
  student.
- Conditional variance equals both the harvestable reward dose and the irreducible regression error.
- Reward curvature or a Jensen gap explains the Phase-G result.
- Target extremity or variance is the established mechanism; later controls did not support making it causal.
- BOND proves that forward-KL training cannot reproduce best-of-(N).
- arXiv:2605.11361 proves that a deterministic proximal map represents a KL tilt.
- The current counterfactual campaign changed only the label orientation.
- Phase H refutes Noise Hypernetworks.

The last two theoretical corrections are particularly important:

- BOND treats imitation of best-of-(N) samples as a forward-KL route toward the best-of-(N) distribution;
  it does not prove impossibility. It is useful motivation for soft all-candidate supervision, not a theorem
  explaining C1.
- arXiv:2605.11361 distinguishes the KL exponential tilt from a Wasserstein proximal transport. Its
  deterministic proximal-argmax result belongs to the Wasserstein construction and cannot be imported as an
  exact representation theorem for the KL-tilted law.

Therefore C1 should remain a quantitative empirical limitation unless a new, correct derivation is supplied.

## Diversity: how to interpret and test it

The DINO diversity failure is scientific content, not a footnote. Pairwise objectives can concentrate the
student around reward-favored modes. However, best-of-\(K\) selection itself has a diversity price, so the
student should be compared with both \(M1\) and the oracle policy it is meant to amortize.

The analytic best-of-four KL displacement

\[
\operatorname{KL}_4=\log 4-\frac{3}{4}\approx0.636
\]

is not an entropy theorem and cannot be numerically compared with DINO or LPIPS. Use the existing eight-
candidate teacher bank to form repeated or randomly partitioned quartets, select one winner from each quartet,
and measure teacher-best versus teacher-random diversity directly in the same DINO and LPIPS units.

Report:

1. unconditional within-prompt diversity;
2. prompt-satisfaction rate;
3. diversity within preregistered satisfaction strata as a secondary diagnostic; and
4. alignment versus diversity over calibrated \(\beta\).

Satisfaction-conditioned diversity must not replace the unconditional endpoint, because conditioning on a
post-treatment success variable can itself distort comparisons.

## Data and evaluation decisions

- Use compositional prompt datasets for preference construction; COCO alone is insufficient for targeted
  attribute, count, and relation interventions.
- Use COCO real images and captions for a matched fidelity branch if the complete prompt/reference manifest is
  immutable and shared across arms.
- A stronger generator may supply candidate diversity but must not serve as its own oracle. Labels and final
  evaluation must come from separate instruments.
- Split all CompBench results by evaluator family rather than presenting it as one VQA score.
- Keep official CompBench and GenEval2 for comparability, but add a held-out non-VQA judge and a blinded human
  pairwise study with atomic questions.
- Report absolute scores for \(M1\), placebo, correct preference, counterfactual, REPA, teacher, and
  best-of-four—not only favorable pairwise differences.
- A second backbone is required for a broad mechanism claim. It need only replicate the teacher-headroom
  versus selected-consistency-transfer contrast, not the full campaign.

## Exact next implementation sequence

### Tier (0): no new training

1. Create an immutable result registry that distinguishes the \(+0.1220\), \(+0.0717\), and \(+0.0485\)
   oracles and records prompts, candidate generator, selector, evaluator, sample count, hashes, job ID, and
   W&B run ID.
2. Recompute every oracle-recovery ratio and produce a matched/joint-bootstrap transfer bound.
3. Audit the two fidelity jobs; regenerate \(M1\) in the same job if exact equivalence cannot be proven.
4. Compute absolute results versus \(M1\) for every admitted arm under identical evaluation conditions.
5. Produce paired bootstrap intervals for BLIP-VQA, UniDet, and CLIP evaluator families.
6. Recover the existing distributions of
   \(\beta(\Delta_\theta-\Delta_0)\), \(e_\theta^+\), \(e_\theta^-\), \(e_0^+\), \(e_0^-\), gradient norm, and
   per-\(\sigma\) contribution. Check whether the loss improves winners, worsens losers, or both.
7. Measure the teacher oracle's empirical DINO and LPIPS diversity price using repeated quartets.
8. Complete the citation audit, including PSO, BOND, Noise Hypernetworks, RTDMD tables, and the KL-versus-
   Wasserstein distinction in arXiv:2605.11361.

### Tier (1): decisive pilots

#### Fixed-unordered-pair causal pilot

Store once

\[
(a_i,b_i)=\left(
\operatorname*{arg\,max}_{j}R(x_{ij},c_i),
\operatorname*{arg\,min}_{j}R(x_{ij},c_i)
\right).
\]

Every arm receives the same unordered images, student-conditioning prompt (c_i), noise, timestep, batch
order, optimizer, initialization, and update count. Training reads serialized indices and asserts equality;
it never recomputes an extremum.

| Arm | Orientation on the fixed pair |
|---|---|
| correct | original-prompt sign |
| counterfactual | sign induced by \(R(\cdot,\widetilde c_i)\) on the same pair |
| random | deterministic, balanced random sign |
| inverted | opposite of the original-prompt sign |

Primary analysis: all \(5{,}042\) prompts, interpreted as a policy assignment.

Sharp secondary analysis: the (1{,}373) records where the same pair reverses.

#### Shared-noise mechanism pilot

At matched compute compare:

- shared \(\epsilon\) versus independent \(\epsilon^+,\epsilon^-\);
- low-, middle-, and high-\(\sigma\) bands; and
- gradient variance and outcome metrics, not training loss alone.

If independent noise performs identically and the high-\(\sigma\) band is not special, retire shared-noise
coalescence as the mechanism.

#### Calibrated strength pilot

Choose \(\beta\) values after observing the existing logit distribution. Include \(\beta=0\) or the exact
\(M1\)-anchored null. Measure the alignment–diversity Pareto rather than choosing the lowest preference loss.

### Tier (2): publication-grade confirmation

1. At least three training seeds for correct, random, and counterfactual; preferably five for the two headline
   arms.
2. Offline PSO equation as the closest mandatory baseline.
3. Online PSO-style arm as a support-mismatch robustness check if resources permit.
4. One canonical \(s\mathrm{CM}\) or \(r\mathrm{CM}\)-family baseline alongside the custom \(M1\).
5. RTDMD released-checkpoint result as the reward-plus-few-step positive control.
6. REPA isolation at synchronized initialization and matched seeds.
7. A soft all-candidate variant

\[
w_j=\frac{\exp(r_j/\tau)}{\sum_k\exp(r_k/\tau)}
\]

   as the first principled diversity remedy after the oracle-diversity audit.
8. A contrastive label

\[
S(x;c,\widetilde c)=R(x,c)-R(x,\widetilde c)
\]

   as a later arm that discounts generic image quality.
9. Independent non-VQA judge, blinded human evaluation, and one second backbone.

## Go or no-go logic

By approximately 2026-08-21, proceed with an ICLR-scale paper only if:

1. the fixed-pair correct arm beats random and counterfactual on independent metrics;
2. the reversal subset agrees;
3. the shared-noise or \(\sigma\)-band ablation supplies a reproducible mechanism—or the paper is explicitly
   written without one;
4. diversity is either non-inferior or lies on a convincing oracle-referenced Pareto frontier;
5. fidelity is measured against a same-job \(M1\); and
6. the result is no longer dependent on a single evaluator family.

If the fixed-pair effect survives and the shared-noise mechanism survives, the project supports a strong
mechanistic paper. If the fixed-pair effect survives but shared noise does not, write a causal analysis paper
and leave the mechanism open. If the fixed-pair effect disappears, reframe the positive result as an effect of
text-conditioned pair mining rather than orientation. If diversity remains substantially worse than both
\(M1\) and oracle selection, do not call the resulting model better; make the Pareto failure a central result.

Official ICLR 2027 dates: abstract registration 2026-09-18 AOE and paper submission 2026-09-25 AOE. The
second-backbone and human-evaluation package can move to the ICML 2027 strengthening cycle if it cannot be
completed credibly in time.

## W&B and artifact contract for every future job

Every scheduler job must create one unique W&B run and one immutable local output directory. Resume and
overwrite are disabled by default.

Required configuration and lineage:

- scheduler job ID and dependency chain;
- W&B entity, project, group, run ID, and human-readable arm name;
- git branch, commit, and dirty diff or patch artifact;
- full command and resolved arguments;
- base, teacher, reference, and initialization checkpoint hashes;
- cache paths and hashes, candidate indices, prompt-manifest hash, evaluator versions, and dataset split;
- random seeds for training, sampling, label randomization, and bootstrap;
- environment snapshot and GPU topology.

Required training metrics:

- total, preference, anchor, and REPA losses;
- \(e_\theta^+\), \(e_\theta^-\), \(e_0^+\), \(e_0^-\) separately;
- \(\Delta_\theta\), \(\Delta_0\), \(\Delta_\theta-\Delta_0\), and
  \(\beta(\Delta_\theta-\Delta_0)\) histograms;
- winner improvement and loser degradation separately;
- pre-clip and post-clip gradient norms, clipping rate, and gradient variance;
- metrics by \(\sigma\)-band and label arm;
- anchor drift, REPA projector synchronization, throughput, memory, and numerical failures.

Required qualitative and evaluation artifacts:

- fixed prompts and fixed sampling seeds at step (0) and regular intervals;
- side-by-side preferred/rejected training pairs with both original and counterfactual scores;
- fresh CompBench, GenEval2, fidelity, and diversity samples;
- non-cherry-picked wins, ties, and failures;
- final checkpoint plus all analysis JSON, Markdown verdicts, and bootstrap samples.

Before launch, assert that no target output path or W&B run ID already exists. After completion, write one
machine-readable manifest linking cache \(\rightarrow\) training \(\rightarrow\) checkpoint \(\rightarrow\)
generation \(\rightarrow\) evaluator \(\rightarrow\) report.

## Repository and compute state at handoff

- Branch: `codex/phase-l-publication-campaign` at commit `4431e8b`.
- The worktree contains substantial historical tracked and untracked material. Do not delete, reset, or stage
  unrelated files. Establish a narrow tracked artifact list for the next campaign.
- No EnergyVLM job was active at the handoff check. Scheduler jobs `100569` and `100570`
  belonged to the separate `bd3lms` working directory and must not be treated as this project's jobs.
- Lustre had approximately \(3.1\,\mathrm{PB}\) free globally; the NFS home allocation showed approximately
  \(23\,\mathrm{GB}\) free. Continue to place checkpoints and generated images on Lustre and keep only code,
  manifests, and compact reports in NFS-backed locations.
- This documentation update submits no jobs and changes no checkpoints or datasets.

## Canonical artifacts

| Subject | Artifact |
|---|---|
| teacher best-of-four oracle | `exp0/primary_headroom/verdict.md` |
| noise-channel oracle and predictability | `reports/phaseH_gate_report.md` |
| selected-trajectory null | `reports/phaseC1_results_report.md` |
| RTDMD audit | `reports/phaseF_rtdmd_reward_effect_report.md` |
| Phase-I placebo comparison | `phaseI/eval_99955/preference_vs_placebo.md` |
| counterfactual construction comparison | `phaseI/counterfactual_eval_100593/correct_vs_counterfactual.md` |
| counterfactual diversity | `phaseI/counterfactual_eval_100593/diversity.md` |
| Phase-I design and amendment | `phaseI/EXPERIMENT.md` |
| PSO/D3PO implementation audit | `reports/pso_d3po_reference_audit.md` |
| novelty audit | `reports/related_work_2026_novelty_audit.md` |
| runnable A–Z roadmap | `reports/research_verdict_and_roadmap.md` |
| supervisor slides | `reports/architecture.html` |

Where an older narrative disagrees with this handoff, prefer raw JSON/JSONL artifacts first, then this handoff,
then the updated roadmap. Historical theory notes are retained only to document how hypotheses were tested and
retracted.
