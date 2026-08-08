# Research verdict and A–Z publication roadmap

**Decision date:** 2026-08-07

**Scope:** four external reviews, the compiled review, the audited experiment artifacts, and the PSO/D3PO reference implementations.

**Status as of 2026-08-08 — the decision below has been executed.** The zero-training audit, the
fixed-pair causal pilot, the shared-noise falsification, three-seed replication, the teacher ceiling
and the offline-PSO baseline are all complete. Results: `reports/phaseFP_results/RESULTS.md`.

> **Current decision:** the preregistered primary **passed** — dose slope \(-0.13972\) on seed 1,
> replicated across three seeds (mean \(-0.12385\), all same sign), with GenEval2 agreeing
> (\(-0.14317\)). Orientation is causal and prompt-specific. **Eight of the nine promotion-gate
> conditions are met**; the exception is gate 5 (DINO diversity), which failed and which the
> alignment–diversity frontier result now explains rather than excuses.
>
> The open problem has moved. Every configuration tested — ours, PSO's, shared versus independent
> noise, all four orientations, three seeds — lands on or below a single alignment–diversity frontier
> (\(r=-0.9622\)). **Nothing moves the frontier outward.** That, not further hyperparameter search,
> is what the remaining work should attack.

> **Source of truth.** This roadmap is operational. The complete evidence and theory handoff is
> `reports/project_handoff_2026-08-07.md`, updated by `reports/phaseFP_results/RESULTS.md`. Raw
> JSON/JSONL artifacts take precedence over all prose.

## One-paragraph verdict

There is a strong paper here, but its defensible contribution is not a new pairwise preference loss. The Phase-I
objective is an offline, teacher-sourced, rectified-flow-compatible member of the Diffusion-DPO/D3PO/**Pairwise
Sample Optimization (PSO)** family. The strongest paper is instead a mechanistic and empirical account of how a
few-step generator acquires compositional preferences: input-side best-of-
\(K\) trajectory selection does not transfer through ordinary consistency regression; an explicit comparative
loss does transfer; REPA is a fidelity regularizer rather than the semantic mechanism; and the remaining causal
question is whether the comparative signal is prompt-specific rather than a generic image-quality prior. The
completed counterfactual experiment estimates the **total effect of the text-conditioned pair-construction
policy**: the text changed pair membership for \(60.75\%\) of records and orientation as well. Pair identity is
downstream of the intervention, so the result is not discarded; it simply does not identify the
orientation-only direct effect. The next decisive run freezes the candidate indices and changes only the
preference orientation. In parallel, shared versus independent corruption noise and a \(\sigma\)-band ablation
must falsify or support the proposed high-noise input-coalescence mechanism before it enters the paper.

## What the existing evidence establishes

| Finding | Evidence | Claim strength |
|---|---|---|
| Teacher trajectories contain usable semantic variation | **SUPERSEDED 2026-08-08.** The \(+0.0717\) figure was measured on the **train** split against a control that *excluded the oracle*, which inflates it by ~42%. Re-measured on the held-out split (LSF 101493): best-of-four minus a uniform draw over all four is \(+0.05287\;[+0.04336,+0.06247]\) | Still strong, but use the unbiased \(+0.05287\) as the amortization denominator |
| Frozen-student noise seeds also contain oracle headroom | Best-of-four over \(M1\)'s own initial seeds gives \(+0.0485\;[+0.0386,+0.0585]\), but shallow OOF reward prediction has \(r=-0.0123\) | Real but distinct oracle; not the teacher headroom and not cheaply selectable |
| Winner-only trajectory selection does not amortize through the audited \(M1\) consistency objective | Phase C1: CompBench \(+0.0004\;[-0.0090,+0.0097]\), versus seed control \(+0.0007\;[-0.0084,+0.0096]\) | Strong for this objective and setting; not a universal impossibility theorem |
| Explicit pairwise training changes fresh student samples | Preference versus randomized-label placebo: CompBench \(+0.04587\), GenEval2 \(+0.04958\); both paired intervals exclude zero | Strong one-seed causal evidence about correct orientation versus randomized orientation |
| REPA's semantic claim did not replicate | Phase-I increment \(+0.00833\) is carried entirely by BLIP-VQA; UniDet \(+0.0017\) and 3-in-1 \(+0.0025\) are null. Matched-seed isolation running (LSF 101912–101914) | Retired as an alignment claim; fidelity role still under test |
| The frozen-MMDiT ranker learns useful image quality but weak text dependence | Better-than-random selection, while correct versus wrong/blank text is null on official CompBench | Strong negative mechanism result for the current ranker |
| The completed correct-versus-counterfactual training comparison is positive | CompBench \(+0.04923\), GenEval2 \(+0.04100\), and favorable fidelity point estimates | Total effect of text-conditioned pair construction; not the orientation-only direct effect |
| Diversity is a live failure mode, but on a frontier | Preregistered DINO gate failed again (\(-0.03961\)). However the student loses ~half the diversity that best-of-4 selection itself costs (\(-0.0632\) vs \(-0.12672\)) and stays *more* diverse than that policy. Alignment and DINO trade at \(r=-0.9622\) across 12 arms | Established; reframed from "collapse" to a Pareto frontier nothing has yet escaped |
| The positive result is not confined to VQA evaluators | **CONFIRMED with paired intervals.** Fixed-pair `correct`−`counterfactual` by family: BLIP-VQA \(+0.04083\), **UniDet \(+0.06599\;[+0.04936,+0.08278]\)**, 3-in-1 \(+0.01439\). The architecturally disjoint detection family moves *most* | Established; the bootstrap the old row asked for was run |

## Blocking corrections before any new training

1. Never conflate teacher best-of-four \(+0.0717\) with frozen-student noise best-of-four \(+0.0485\).
2. Recompute amortization ratios with matched uncertainty. The shortcut \(0.0097/0.0717\approx13.5\%\) is a
   point-denominator approximation, not yet a formal \(95\%\) upper bound.
3. Audit the unexplained cross-job FID/CMMD shift and regenerate \(M1\) in the same job if equivalence cannot
   be proven.
4. Bootstrap CompBench by evaluator family: BLIP-VQA, UniDet, and CLIP.
5. Recover the already logged preference-logit, positive-error, negative-error, gradient, and
   per-\(\sigma\) distributions before choosing \(\beta\).
6. Measure the teacher oracle's empirical DINO/LPIPS diversity price from repeated cached quartets.
7. Remove the invalid KL-versus-Wasserstein use of arXiv:2605.11361 and do not use BOND as an impossibility
   theorem.

## The exact causal correction

For each prompt \(c_i\), keep the original-prompt top and bottom candidate indices fixed:

\[
(a_i,b_i)=\left(
\operatorname*{arg\,max}_{j}R(x_{ij},c_i),
\operatorname*{arg\,min}_{j}R(x_{ij},c_i)
\right).
\]

Every arm must receive the same unordered pair \(\{x_{ia_i},x_{ib_i}\}\), the same student-conditioning text
\(c_i\), and the same noise, timestep, batch order, optimizer, initialization, and compute. Only the sign
\(y_i\in\{-1,+1\}\) changes:

\[
\mathcal L_{\mathrm{pref}}^{(i)}
=-\log\sigma\!\left[
-\beta y_i
\left\{
\left(e_{\theta,ia_i}-e_{\theta,ib_i}\right)
-\left(e_{0,ia_i}-e_{0,ib_i}\right)
\right\}
\right].
\]

The four arms are:

1. **Correct:** \(y_i=+1\), the original-prompt ordering.
2. **Counterfactual:** reverse the sign only when \(R(x_{ia_i},\widetilde c_i)<R(x_{ib_i},\widetilde c_i)\);
   otherwise retain it. The all-prompt analysis is the assigned-policy effect.
3. **Random:** a deterministic balanced random sign, the matched placebo.
4. **Inverted:** \(y_i=-1\), the strongest sanity check.

The primary population is all \(5{,}042\) editable prompts. The sharp orientation-only sensitivity population
is the \(1{,}373\) prompts (\(27.23\%\)) for which the counterfactual actually reverses the fixed original pair.
Candidate indices must be serialized in the cache and asserted equal across arms; training must never recompute
an \(\arg\max\) or \(\arg\min\).

## Paper positioning after the reviews

### Main claim

**Under the audited four-step consistency estimator, selecting better teacher trajectories produced no
measurable independent transfer, whereas an explicit comparative objective did.** We measure this channel
distinction and its alignment–fidelity–diversity trade-off. We do not claim a universal invariance theorem.

### Method claim

Our implementation is a **shared-noise, anchored, offline PSO-style preference objective for a few-step
rectified-flow student**, optionally combined with REPA. The loss itself is not claimed as novel. Potential
novelty must come from the controlled channel measurement, fixed-pair counterfactual identification, and the
alignment–fidelity–diversity characterization. Shared-noise input coalescence is a hypothesis until the
shared-versus-independent and \(\sigma\)-band ablations pass.

### Claims to retire

- “We introduce pairwise preference optimization for few-step diffusion.”
- “The completed counterfactual run changes only the label text.”
- “A deterministic generator cannot represent a reward-tilted distribution.”
- “BOND proves selected-sample regression cannot reproduce best-of-\(N\).”
- “The deterministic Wasserstein proximal map in arXiv:2605.11361 is an exact KL-tilt representation.”
- “Best-of-\(K\) selection never works.” It works through target/noise/gradient channels in prior work; our null
  is for input-channel selection under the audited consistency estimator.
- “REPA improves compositional alignment.” Current evidence supports a fidelity role; semantic gains need
  isolated replication.
- Any publication claim based only on VQAScore, the training label source.

## A–Z execution list

Items are ordered by dependency. A phase advances only after its stated gate passes.

| ID | Task | Concrete output and gate | State |
|---|---|---|---|
| **A — Archive and reconcile** | Freeze every admitted run and distinguish the three oracle quantities | Immutable registry with hashes, prompts, selector, evaluator, scheduler and W&B identifiers | **DONE** — `reports/registry/` (`audit/build_registry.py`); every headline number recomputed from raw artifacts |
| **B — Baseline genealogy** | Cite Diffusion-DPO, D3PO, PSO, GORS, GRAFT/P-GRAFT, D-OPSD, REPA, and the recent few-step reward literature | Related-work matrix states exact overlap and surviving difference | Drafted |
| **C — Correct the cache** | Build fixed-index records containing \((a_i,b_i)\), both score vectors, edit metadata, and reversal flag | Cross-arm assertion: candidate indices and tensors are bit-identical | **DONE** — `phaseFP/fixedpair_101185` (LSF 101185). Needed no GPU: the completed counterfactual cache already stores both score vectors over identical candidates |
| **D — Dose audit** | Measure score margins, reversal rates, edit validity, category coverage, and effective label disagreement | Report all-prompts and reversal-only populations before training | **DONE** — `phaseFP/fixedpair_101185/dose_audit.md`. Dose ladder 0 / 0.108 / 0.287 / 0.573; on the reversal subset `counterfactual` **is** `inverted` |
| **E — Zero-training audit** | Analyze errors/logits; evaluator-family deltas; matched absolute baselines; fidelity lineage; oracle diversity price | Resolve every cheap reviewer question before a GPU pilot | **DONE** except the oracle diversity price (LSF 101198, running) — `reports/audit_2026-08-07/README.md` |
| **F — Fixed-pair smoke** | Run tiny correct/counterfactual/random/inverted arms end to end | Exact cache identity, finite gradients, distinct expected signs, unique W&B artifacts | **PASSED** — LSF 101190: 24/24 rows with bit-identical teacher-endpoint SHA-256 across all four arms; `inverted` first logit exactly negates the others |
| **G — Gate the causal pilot** | Train the four fixed-pair pilot arms with locked hyperparameters | Preregistered primary is the **monotone dose slope across all four arms**, not a single pairwise contrast (see `phaseFP/PREREGISTRATION.md`) | **RUNNING** — LSF 101193–101196; evaluation 101206–101210 |
| **H — Human edit audit** | Blindly review a stratified sample of original/counterfactual prompts and fixed pairs | Report edit validity and whether the fixed pair truly changes the edited concept | **PACKET READY, AWAITING A HUMAN** — 210 stratified items at `phaseFP/edit_audit/`; job 101922 renders the frozen training pair for each so the image-side question is answerable. Prompt order AND image side both blinded. 0/210 answered; this is the one item that cannot be automated |
| **I — Independent judges** | Evaluate with official CompBench, GenEval2, CLIP-family alignment, and at least one non-CLIP VLM/human audit | Direction agrees beyond VQAScore; disagreements are reported, not averaged away | **PARTIAL** — official CompBench and GenEval2 both done and agreeing (slope −0.13972 vs −0.14317). Still missing the non-CLIP VLM / human judge |
| **J — Judge calibration** | Quantify selector–evaluator dependence and pairwise human agreement | Correlation and agreement table; VQAScore never serves as sole endpoint | **PARTIAL** — evaluator-family split done with paired intervals (`audit/evaluator_family_bootstrap.py`); UniDet is architecturally disjoint from the VQAScore selector and moves most. Human agreement still absent, and depends on H |
| **K — Knob calibration** | Choose a small \(\beta\) sweep from the measured logit distribution; include \(\beta=0\) or exact null | Select by independent alignment subject to fidelity and diversity constraints | **PRECONDITION DONE, SWEEP DEPRIORITISED** — logit distribution measured (β=100 under-driven, ~half of steps inert, clipping fires 100% of the time). Sweep deliberately not run: the Pareto result implies a β sweep would produce more points on the same frontier, not move it |
| **L — Listwise comparison** | Compare top–bottom logistic, all-pairs, and soft listwise supervision at matched compute | Establish whether discarded candidates add signal; no method novelty claim | Not started. Deprioritised for the same reason as K; the soft all-candidate variant is now interesting as a **diversity** remedy rather than an alignment one |
| **M — Modern consistency baseline** | Add a canonical recent few-step baseline such as \(s\mathrm{CM}\) or \(r\mathrm{CM}\), plus unmodified SD\(3.5\)-M | Reviewer-facing table separates our custom \(M1\) from standard baselines | Not started. Unmodified SD3.5-M is covered at 8 steps by `TeacherUniform` (0.47792); a canonical sCM/rCM baseline is still missing |
| **N — Noise coupling** | Compare shared versus independent noising and low/mid/high-\(\sigma\) bands | Falsify or support high-noise coalescence; report gradient variance and outcomes | **RUNNING** — LSF 101197 is `correct` with independent negative-branch noise; \(z^+\) is bit-identical between conditions. The \(\sigma\) profile is measured from per-example telemetry first (`phaseFP/analyze_telemetry.py`); band-restricted training arms run only if that profile is non-flat |
| **O — Offline PSO baseline** | Independently implement the paper equation for fixed offline pairs | Reproduce expected signs and compare fairly with our anchored error-ratio form | **DONE** — LSF 101373. Our objective *reduces to* the PSO equation (`reports/audit_2026-08-07/pso_equivalence.md`); PSO's settings score 0.56418 vs our 0.55697 but land ~1.5 residual SDs **below** the alignment–diversity frontier. Neither the loss nor our configuration is a contribution |
| **P — Online PSO baseline** | Port the algorithmic idea, not unlicensed code, to the rectified-flow student | Small on-policy comparison quantifies the value of fresh student samples | Not started (roadmap marks optional) |
| **Q — Quality frontier** | Evaluate FID, KID, CMMD, precision, recall, alignment, unconditional diversity, satisfaction-stratified diversity, and oracle diversity | Report a Pareto frontier rather than a single scalar “win” | **MOSTLY DONE** — FID, CMMD, precision, recall, alignment, unconditional diversity and the oracle diversity price all measured across 15 models against one shared reference. **KID never computed**; satisfaction-stratified diversity not done |
| **R — REPA isolation** | Run preference with and without REPA using synchronized projectors and matched seeds | Replicated fidelity improvement with no significant semantic/diversity regression | **RUNNING** — LSF 101912–101914 train `correct`+REPA at the same three seeds as the REPA-free arms, so the increment is a matched within-seed contrast. Evals 101915–101917. Closes a real gap: Phase-FP had no REPA arm, and the Phase-I increment was BLIP-VQA-only |
| **S — Seeds and statistics** | Use at least three training seeds for finalists; paired prompt bootstrap within seed and seed-level variation across runs | Confidence intervals reflect both prompt and training randomness | **DONE — GATE MET** — 3 seeds × 4 arms; slope −0.13972/−0.09591/−0.13591, all same sign, ordering identical. Across-seed range (0.0438) EXCEEDS the prompt-bootstrap width (~0.032), so training randomness dominates; the two are reported separately, never pooled |
| **T — Teacher ceiling** | Evaluate teacher, base, \(M1\), best-of-\(K\), and oracle-selected samples on identical prompts | Decompose available headroom from amortization efficiency | **DONE** — LSF 101493, on the same val prompts as the students. Corrected a long-standing number: the +0.0717 headroom used a control that excluded the oracle, inflating it ~42%; unbiased headroom is **+0.05287**. Joint bootstrap now possible → amortization bound **~16%**, superseding 11.7% |
| **U — Unseen data** | Separate training prompts from natural held-out prompts and benchmark templates | Gains survive template, category, and prompt-source shifts | **PARTIAL** — val split is disjoint from training prompts and GenEval2 is a separate template source; a natural (non-benchmark) prompt distribution is still untested |
| **V — Visual audit** | Preselect random seeds and failure categories; create non-cherry-picked grids and blinded ratings | Include wins, ties, failures, counting, and relation examples | **DONE** — `reports/phaseFP_results/visual_audit/`, 35 items / 175 images. Selection rule fixed in advance: score-blind uniform samples plus the largest win, nearest tie and largest **loss** per category; column order blinded |
| **W — W&B contract** | One run per job; immutable names; config, git state, cache hashes, metrics, samples, gradients, and checkpoints logged | Automated collision check and complete artifact lineage | Required for every run |
| **X — Cross-backbone check** | Replicate the central channel result on one second few-step backbone if resources permit | Establish whether the mechanism generalizes beyond SD\(3.5\)-M | Not started; deferrable to the ICML cycle per the go/no-go logic |
| **Y — Yield artifacts** | Package configs, manifests, analysis scripts, evaluator versions, and legal provenance | Reproducibility bundle runs from a clean checkout; no unlicensed vendoring | Not started — scripts, manifests and hashes all exist but are not packaged as a clean-checkout bundle |
| **Z — Zero-overclaim write-up** | Write the paper around the channel mechanism and fixed-pair evidence | Every headline maps to an independent metric, control, and uncertainty estimate | Not started. Evidence package is now largely complete; see `reports/phaseFP_results/RESULTS.md` |

## The run order

\[
\boxed{A\rightarrow E\rightarrow C\rightarrow D\rightarrow F\rightarrow (G\parallel N\parallel K)}
\quad\Longrightarrow\quad
\begin{cases}
\text{fail: diagnose or stop},\\
\text{pass: }H,I,J,O,Q,R.
\end{cases}
\]

Only after that evidence package passes should we execute \(M,S,T,U,V\), and only then consider the
publication-scale campaign and optional \(P,X\). This ordering prevents a large run from precisely estimating
an incompletely identified or diversity-damaging effect.

## Promotion gate for a publication-scale run

All conditions are required:

1. The fixed-pair correct arm beats both counterfactual and randomized orientations on official CompBench with
   a positive paired \(95\%\) interval.
2. GenEval2 agrees in direction, and at least one independent evaluator or blinded human audit agrees.
3. The result persists across at least three training seeds.
4. Matched-prompt fidelity is non-inferior under a preregistered tolerance; KID is included alongside FID.
5. DINO and LPIPS diversity pass preregistered non-inferiority gates.
6. Correct, counterfactual, random, and inverted arms show the expected ordered dose response.
7. A canonical consistency baseline and an offline PSO-style baseline are included.
8. W&B and local artifacts give an immutable, collision-free lineage from cache to final figure.
9. The shared-noise mechanism is either supported by its falsification test or explicitly removed from the
   paper's claims.

If the fixed-pair effect is null, the paper should become the mechanistic negative-results study rather than
escalating the same objective. If alignment improves but diversity again fails, the next work is a constrained
alignment/Pareto study—not a larger unconstrained run.
