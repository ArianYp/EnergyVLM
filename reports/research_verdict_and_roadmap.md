# Research verdict and A–Z publication roadmap

**Decision date:** 2026-08-07

**Scope:** four external reviews, the compiled review, the audited experiment artifacts, and the PSO/D3PO reference implementations.
**Current decision:** **pause publication-scale training. Complete the zero-training audit, then run the
fixed-pair causal pilot and the shared-noise falsification together.**

> **Source of truth.** This roadmap is operational. The complete evidence and theory handoff is
> `reports/project_handoff_2026-08-07.md`. Raw JSON/JSONL artifacts take precedence over both documents.

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
| Teacher trajectories contain usable semantic variation | Frozen-teacher best-of-four improves independent held-out CompBench by \(+0.0717\;[+0.0573,+0.0865]\) over random | Strong sample-level evidence |
| Frozen-student noise seeds also contain oracle headroom | Best-of-four over \(M1\)'s own initial seeds gives \(+0.0485\;[+0.0386,+0.0585]\), but shallow OOF reward prediction has \(r=-0.0123\) | Real but distinct oracle; not the teacher headroom and not cheaply selectable |
| Winner-only trajectory selection does not amortize through the audited \(M1\) consistency objective | Phase C1: CompBench \(+0.0004\;[-0.0090,+0.0097]\), versus seed control \(+0.0007\;[-0.0084,+0.0096]\) | Strong for this objective and setting; not a universal impossibility theorem |
| Explicit pairwise training changes fresh student samples | Preference versus randomized-label placebo: CompBench \(+0.04587\), GenEval2 \(+0.04958\); both paired intervals exclude zero | Strong one-seed causal evidence about correct orientation versus randomized orientation |
| REPA may protect or improve fidelity | Earlier four-step CMMD \(69.7\to67.4\); Phase-I CompBench increment \(+0.00833\) | Promising, not established: one seed and incomplete matched fidelity inference |
| The frozen-MMDiT ranker learns useful image quality but weak text dependence | Better-than-random selection, while correct versus wrong/blank text is null on official CompBench | Strong negative mechanism result for the current ranker |
| The completed correct-versus-counterfactual training comparison is positive | CompBench \(+0.04923\), GenEval2 \(+0.04100\), and favorable fidelity point estimates | Total effect of text-conditioned pair construction; not the orientation-only direct effect |
| Diversity is a live failure mode | DINO diversity \(-0.04136\), preregistered gate failed; LPIPS \(-0.01273\), within tolerance | Must be resolved before any “better generator” claim |
| The positive result is not obviously confined to VQA evaluators | Correct-minus-counterfactual family means are about \(+0.0405\) on BLIP-VQA and \(+0.0689\) on UniDet categories | Promising architecture-disjoint evidence; paired family bootstrap is still required |

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
| **A — Archive and reconcile** | Freeze every admitted run and distinguish the three oracle quantities | Immutable registry with hashes, prompts, selector, evaluator, scheduler and W&B identifiers | **First** |
| **B — Baseline genealogy** | Cite Diffusion-DPO, D3PO, PSO, GORS, GRAFT/P-GRAFT, D-OPSD, REPA, and the recent few-step reward literature | Related-work matrix states exact overlap and surviving difference | Drafted |
| **C — Correct the cache** | Build fixed-index records containing \((a_i,b_i)\), both score vectors, edit metadata, and reversal flag | Cross-arm assertion: candidate indices and tensors are bit-identical | Next |
| **D — Dose audit** | Measure score margins, reversal rates, edit validity, category coverage, and effective label disagreement | Report all-prompts and reversal-only populations before training | Next |
| **E — Zero-training audit** | Analyze errors/logits; evaluator-family deltas; matched absolute baselines; fidelity lineage; oracle diversity price | Resolve every cheap reviewer question before a GPU pilot | Next; no new training |
| **F — Fixed-pair smoke** | Run tiny correct/counterfactual/random/inverted arms end to end | Exact cache identity, finite gradients, distinct expected signs, unique W&B artifacts | Blocked by C |
| **G — Gate the causal pilot** | Train the four fixed-pair pilot arms with locked hyperparameters | Correct beats counterfactual and random on independent metrics; reversal subset agrees | Blocked by F |
| **H — Human edit audit** | Blindly review a stratified sample of original/counterfactual prompts and fixed pairs | Report edit validity and whether the fixed pair truly changes the edited concept | Parallel after C |
| **I — Independent judges** | Evaluate with official CompBench, GenEval2, CLIP-family alignment, and at least one non-CLIP VLM/human audit | Direction agrees beyond VQAScore; disagreements are reported, not averaged away | After G |
| **J — Judge calibration** | Quantify selector–evaluator dependence and pairwise human agreement | Correlation and agreement table; VQAScore never serves as sole endpoint | After H |
| **K — Knob calibration** | Choose a small \(\beta\) sweep from the measured logit distribution; include \(\beta=0\) or exact null | Select by independent alignment subject to fidelity and diversity constraints | After E and F |
| **L — Listwise comparison** | Compare top–bottom logistic, all-pairs, and soft listwise supervision at matched compute | Establish whether discarded candidates add signal; no method novelty claim | After K |
| **M — Modern consistency baseline** | Add a canonical recent few-step baseline such as \(s\mathrm{CM}\) or \(r\mathrm{CM}\), plus unmodified SD\(3.5\)-M | Reviewer-facing table separates our custom \(M1\) from standard baselines | Before scale |
| **N — Noise coupling** | Compare shared versus independent noising and low/mid/high-\(\sigma\) bands | Falsify or support high-noise coalescence; report gradient variance and outcomes | **Run with G** |
| **O — Offline PSO baseline** | Independently implement the paper equation for fixed offline pairs | Reproduce expected signs and compare fairly with our anchored error-ratio form | After F |
| **P — Online PSO baseline** | Port the algorithmic idea, not unlicensed code, to the rectified-flow student | Small on-policy comparison quantifies the value of fresh student samples | After O; optional if compute-limited |
| **Q — Quality frontier** | Evaluate FID, KID, CMMD, precision, recall, alignment, unconditional diversity, satisfaction-stratified diversity, and oracle diversity | Report a Pareto frontier rather than a single scalar “win” | After G |
| **R — REPA isolation** | Run preference with and without REPA using synchronized projectors and matched seeds | Replicated fidelity improvement with no significant semantic/diversity regression | After K |
| **S — Seeds and statistics** | Use at least three training seeds for finalists; paired prompt bootstrap within seed and seed-level variation across runs | Confidence intervals reflect both prompt and training randomness | Only finalists |
| **T — Teacher ceiling** | Evaluate teacher, base, \(M1\), best-of-\(K\), and oracle-selected samples on identical prompts | Decompose available headroom from amortization efficiency | Before final table |
| **U — Unseen data** | Separate training prompts from natural held-out prompts and benchmark templates | Gains survive template, category, and prompt-source shifts | Before scale claim |
| **V — Visual audit** | Preselect random seeds and failure categories; create non-cherry-picked grids and blinded ratings | Include wins, ties, failures, counting, and relation examples | With each evaluation |
| **W — W&B contract** | One run per job; immutable names; config, git state, cache hashes, metrics, samples, gradients, and checkpoints logged | Automated collision check and complete artifact lineage | Required for every run |
| **X — Cross-backbone check** | Replicate the central channel result on one second few-step backbone if resources permit | Establish whether the mechanism generalizes beyond SD\(3.5\)-M | After main result |
| **Y — Yield artifacts** | Package configs, manifests, analysis scripts, evaluator versions, and legal provenance | Reproducibility bundle runs from a clean checkout; no unlicensed vendoring | Before submission |
| **Z — Zero-overclaim write-up** | Write the paper around the channel mechanism and fixed-pair evidence | Every headline maps to an independent metric, control, and uncertainty estimate | Final |

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
