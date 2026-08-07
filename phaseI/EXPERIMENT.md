# Phase I — preference signal × REPA fidelity pilot

> **Retrospective validity correction, 2026-08-07.** The pairwise loss below belongs to the
> Diffusion-DPO/D3PO/Pairwise Sample Optimization family; it is not a novel objective. Our implementation is an
> offline, teacher-sourced, shared-noise, anchored rectified-flow compatibility variant. Also, the completed
> correct-versus-counterfactual follow-up did not change only the label sign: it recomputed the top and bottom
> indices using each text. The unordered pair changed for \(60.75\%\) of the \(5{,}042\) records. Therefore its
> positive held-out result estimates the **total effect of the text-conditioned pair-construction policy**, not
> the orientation-only direct effect. Pair identity is a downstream mediator of that policy. Publication-scale
> promotion is paused until the fixed-pair experiment in the amendment below passes.
>
> **Second audit correction, 2026-08-07.** The independent frozen-teacher best-of-four oracle is
> \(+0.0717\;[+0.0573,+0.0865]\). The previously quoted
> \(+0.0485\;[+0.0386,+0.0585]\) belongs to best-of-four selection among the frozen \(M1\) student's own
> initial noise seeds. Both are valid, but they are different interventions and must not share an
> amortization denominator. The completed counterfactual campaign is retained as the total effect of the
> text-conditioned pair-construction policy; the fixed-pair amendment estimates the orientation-only effect.

## Evidence admitted into the design

These numbers were recomputed directly from the raw JSON/JSONL artifacts on 2026-08-06. The prose reports
were not treated as evidence.

1. The 5,559-prompt teacher candidate cache has real endpoint headroom: top-of-4 minus candidate mean
   VQAScore is **+0.1220** (prompt bootstrap 95% CI **[+0.1191,+0.1250]**).
2. On held-out CompBench prompts, selecting the frozen teacher's best-of-4 candidate with VQAScore and judging
   it with the official category evaluators gives **+0.0717** over a random candidate (95% CI
   **[+0.0573,+0.0865]**). Selection therefore helps at the sample level under an evaluator independent of
   the selector. Separately, selection among four \(M1\) initial-noise seeds gives **+0.0485**
   **[+0.0386,+0.0585]**; that noise-channel result is not the teacher oracle.
3. The one available winner-only consistency-distillation pair (B4 versus B2) gives **+0.00045** on the same
   official aggregate (95% CI **[-0.0088,+0.0096]**). This is evidence that *that training recipe* failed,
   not a theorem that preferences cannot be learned: there is one training seed per arm and the model never
   receives an explicit winner-versus-loser loss.
4. The prior REPA result is only exploratory. Its cached point estimate is CMMD **69.7→67.4** at four steps,
   with no CMMD confidence interval and one seed. A fresh code audit also found that the external REPA
   projector was not DDP-synchronized across ranks. Phase I fixes this explicitly.

## Question

Can an already-distilled four-step model shift probability toward compositionally better teacher outputs
when the competition is represented explicitly, while REPA protects fidelity?

Winner filtering is replaced by a pairwise preference objective. For the same prompt and the same cached
four-candidate pool, the highest-VQA and lowest-VQA teacher endpoints form a pair. The student starts exactly
from M1. At matched noising levels, it is trained to reduce its flow error on the winner relative to the loser,
measured against a frozen M1 reference:

\[
\mathcal L_{\mathrm{pref}}
=-\log\sigma\!\left[-\beta\left{
(e_\theta^+-e_\theta^-)-(e_0^+-e_0^-)
\right\}\right],
\qquad
\mathcal L
=\mathcal L_{\mathrm{pref}}
+\lambda_{\mathrm{anchor}}\mathcal L_{M1}
+\lambda_{\mathrm{REPA}}\mathcal L_{\mathrm{REPA}}.
\]

The pilot used per-dimension mean squared errors and \(\beta=100\). That value was a pilot hyperparameter, not
a literature-derived optimum. Before a sweep, inspect the empirical distribution of
\(\beta(\Delta_\theta-\Delta_0)\), the sigmoid saturation rate, \(e_\theta^+\) and \(e_\theta^-\) separately,
and each quantity by \(\sigma\)-band. A preference loss may improve the winner, worsen the loser, or do both;
the aggregate margin alone cannot distinguish those behaviours.

Shared corruption noise gives

\[
z^+-z^-=(1-\sigma)(x^+-x^-),
\qquad
u^+-u^-=x^- - x^+.
\]

As \(\sigma\to1\), the inputs coalesce while the targets remain separated. This is a mechanistic hypothesis,
not an established explanation. The fixed-pair pilot must compare shared with independent pair noise and
low/mid/high-\(\sigma\) bands.

The M1 anchor is the few-step-preservation term. REPA, when enabled, uses a separate noised-real-COCO forward
and frozen DINOv2 patch targets.

## Arms

| arm | pair labels | REPA | role |
|---|---|---:|---|
| `placebo` | top/bottom orientation randomly flipped per prompt | 0 | identical-compute causal control |
| `preference` | high score preferred to low score | 0 | tests the comparative semantic signal |
| `preference_repa` | high score preferred to low score | 0.5 | tests the proposed alignment + fidelity combination |

All arms start from the same M1 checkpoint and share data order, candidate pairs, optimizer, noise coupling,
budget, and W&B schema. The placebo uses the exact same top/bottom images and score-margin magnitudes; only the
sign of the label is randomized deterministically.

## Decision rule

The semantic method passes only if `preference > placebo` on held-out official CompBench at four steps with a
paired CI excluding zero. VQAScore is in-objective and cannot establish the result. The combined arm is useful
only if it preserves the independent alignment gain and improves fidelity relative to `preference`; a VQA-only
gain, CMMD loss, or visible diversity collapse is a failure.

## Run integrity

- Every LSF job ID maps to a unique output directory and a unique W&B run ID; existing directories are a hard
  error and no shared metric output path is used.
- W&B logs loss decomposition, preference margins, anchor drift, pre/post-clip gradients, memory, throughput,
  local checkpoint paths, and fixed-seed samples at step 0 and throughout training.
- Checkpoints are written atomically. `args.json` records the checkpoint, code hashes, git revision, raw cache
  statistics, W&B identity, and every live hyperparameter.
- The REPA projector is initialized identically, its gradients are explicitly all-reduced, and a cross-rank
  parameter-spread assertion is run after the first optimizer step.

---

## Pre-registered follow-up: text-specific counterfactual labels

**Locked before submission on 2026-08-07.** Phase I established that correct VQAScore preferences beat a
randomized-label placebo on independent CompBench and GenEval2 metrics. The remaining causal ambiguity is
whether those labels encode the prompt's composition or a prompt-independent image-quality prior.

For every training prompt that admits an auditable one-atom edit, regenerate the exact same four frozen-teacher
endpoints with the original prompt and the original seeds. Build two selection caches over the identical prompt
subset:

\[
j^+_{\mathrm{correct}}=\operatorname*{arg\,max}_{j}R(x_j,c),
\qquad
j^+_{\mathrm{cf}}=\operatorname*{arg\,max}_{j}R(x_j,\widetilde c),
\]

where \(\widetilde c\) changes exactly one color, shape, texture, count, spatial relation, depth relation, or
verb. In both training arms, the teacher candidates and student are conditioned on the original prompt \(c\).
Only the text used to orient the preference pair differs. This prevents the intervention from changing the
candidate distribution or the student's input text.

The rule-based edit system covers **5,042 of 5,559 prompts (90.7%)**. Because restricting only the
counterfactual arm would confound label source with training data, both arms train on exactly these 5,042
records:

| arm | student-conditioning text | pair-label text | REPA |
|---|---|---|---:|
| `preference_matched` | \(c\) | \(c\) | 0 |
| `counterfactual` | \(c\) | \(\widetilde c\) | 0 |

Before training, the full cache must pass all of the following informativeness checks: coverage at least
\(0.90\), top-candidate flip fraction at least \(0.20\), top/bottom-pair change fraction at least \(0.30\),
mean absolute candidate-score effect at least \(0.05\), and positive paired-bootstrap lower confidence bounds
for both the original-prompt penalty of the counterfactual winner and the counterfactual-score recovery of that
winner.

### Primary held-out decision

At four inference steps and paired noise seeds:

\[
\Delta_{\mathrm{text}}
=S_{\mathrm{independent}}(\theta_{\mathrm{correct}})
-S_{\mathrm{independent}}(\theta_{\mathrm{cf}}).
\]

The text-specific hypothesis passes only if the equal-category-weighted official CompBench confidence interval
has a lower bound above zero. GenEval2 Soft-TIFA is a co-primary replication. FID and CMMD use images generated
from a separate COCO-caption pool against the corresponding COCO real-image distribution; they must satisfy
the pre-set CMMD non-inferiority tolerance. Four-seed DINO and LPIPS diversity must not show a mode collapse.

All selection shards, code, and starting checkpoints are hashed. Data construction, both training arms, and
evaluation use unique scheduler job IDs in their local paths and W&B run IDs; overwrite and resume are disabled.

---

## Amendment: fixed-pair label-only causal test

**Locked after the 2026-08-07 review audit; this supersedes the causal interpretation above.** In the completed
cache, the correct arm selected its top and bottom candidates under \(c\), while the counterfactual arm selected
its own top and bottom under \(\widetilde c\). Audit of all \(5{,}042\) records found:

| diagnostic | value |
|---|---:|
| unordered candidate pair changed | \(60.75\%\) |
| ordered top–bottom tuple changed | \(67.53\%\) |
| original top versus bottom reverses under \(\widetilde c\) | \(27.23\%\), \(1{,}373/5{,}042\) |
| at least one reversible candidate pair exists | \(74.65\%\) |

The next cache freezes the original-prompt indices

\[
(a_i,b_i)=\left(
\operatorname*{arg\,max}_{j}R(x_{ij},c_i),
\operatorname*{arg\,min}_{j}R(x_{ij},c_i)
\right)
\]

for every arm. Candidate tensors, student prompt, training order, noise, timestep, optimizer, initialization,
and update count are identical. Only the orientation variable changes:

| arm | fixed pair | orientation |
|---|---|---|
| `correct_fixed` | \(\{x_{ia_i},x_{ib_i}\}\) | original-prompt sign |
| `counterfactual_fixed` | same | sign implied by \(R(\cdot,\widetilde c_i)\) on that same pair |
| `random_fixed` | same | deterministic balanced random sign |
| `inverted_fixed` | same | opposite of original-prompt sign |

The primary analysis is the assigned-policy effect over all \(5{,}042\) prompts. A preregistered sharp
orientation-only analysis uses the \(1{,}373\) fixed-pair reversals, where the same two images receive opposite
labels. Training code must read stored indices and assert equality across arms; it may not recompute candidate
selection.

The already completed total-policy comparison remains reportable with its proper label: correct pair construction
beats counterfactual pair construction on CompBench by \(+0.04923\) with \(95\%\) CI
\([+0.04069,+0.05805]\), and on GenEval2 by \(+0.04100\) with \(95\%\) CI
\([+0.02412,+0.05824]\). It also reduced DINO diversity by \(-0.04136\)
\([-0.04794,-0.03472]\), failing the preregistered diversity gate. Those facts prohibit an unconditional
“better generator” or label-only claim.

The next evaluation must additionally report CompBench by evaluator family. Existing category point estimates
already suggest the effect is not confined to VQA scoring: correct minus counterfactual averages approximately
\(+0.0405\) across BLIP-VQA categories and \(+0.0689\) across UniDet categories. These family aggregates need
paired bootstrap intervals before they become formal evidence. Fidelity must be regenerated against \(M1\)
inside the same job unless the unexplained absolute FID/CMMD difference between the two completed Phase-I
evaluation jobs is reconciled exactly.
