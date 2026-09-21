# What the 2023-2026 literature says would make our distillation recipe better

Scouting report, 2026-09-17. Read-only: nothing was run. Sources: alphaXiv full-text reads of every paper cited
in Sections 1-3 (arXiv ids verified against the arXiv API on 2026-09-17; one id from memory was wrong and is
corrected below), plus our own `README.md`, `paper/iclr2027_conference.tex` (Sections 2-3, 3.6),
`docs/report.tex` (ablation appendix) and `docs/rank/README.md` + `docs/rank/debug/per_prompt_analysis.md`.

## 0. Where our recipe sits, in the field's vocabulary

Our loss is **trajectory-preserving regression onto the frozen teacher's realised one-step Tweedie estimate,
from a noisier state**, with no self-bootstrapping, no EMA target, guidance absorbed from w=7 trajectories.
In the literature's taxonomy that is a "trajectory-based / forward-divergence" distillation
(rCM 2510.08431, CrossDistill 2609.14725), closest to the *trajectory-guidance pre-training stage* that
Stability's own SD3.5-Flash (2509.21318) runs for 2k iterations before switching to distribution matching, and
to Hyper-SD's segment-wise consistency (2404.13686). Four facts about our setting that the literature speaks to
directly:

1. **Forward-divergence trajectory losses are mode-covering: they preserve diversity but plateau on
   sharpness/fidelity** (rCM 2510.08431 Sec. 3.3; CrossDistill 2609.14725; AYF 2506.14603). Every 2025-26
   recipe that beats its teacher on GenEval/preference metrics adds a reverse-divergence term evaluated on the
   student's *own* samples (DMD/CA/adversarial). Our CMMD (0.84) vs the 28-step teacher (0.64) is that gap.
2. **Our student is never evaluated during training at the σ it is deployed at.** `docs/report.tex` (line
   1617) records that the 4-step inference grid shares only its endpoints with the 8-step training grid; with
   the converged window K={4..7} and Δ≤3 the student sees inputs at σ∈{0.947, 0.882, 0.800, 0.692, 0.545,
   0.333} and is deployed at σ∈{1.0, 0.857, 0.601, 0.003} (shift-3 schedule; confirm from `scheduler.sigmas`).
   Three 2025-26 papers report gains specifically from evaluating the objective on the student's deployment
   states (SD3.5-Flash "timestep sharing" 2509.21318; CDM's dynamic-schedule ablation 2605.06376; SenseFlow
   trains on the deployment anchors 2506.00523).
3. **Our pseudo-Huber is an L1-norm in practice.** c_h = 0.00054·sqrt(D) = 0.138 was designed (iCT
   2310.14189) for residuals between *adjacent* student predictions; iCT says c should scale with ||x−y||.
   Our residuals (student at s vs teacher's x0 at k>s) are tens of latent units, so the loss is
   sqrt(||Δ||²+c²)−c ≈ ||Δ||: unit-norm gradient per sample in x0-space, equal for every noise level. ECT
   (2406.14548, App. A) shows the pseudo-Huber is exactly "squared-L2 × adaptive weight 1/sqrt(||Δ||²+c²)", and
   that "controlling gradient variance and balancing gradients across noise levels are fundamental". Our
   always-active clip then feeds Adam a unit-norm total gradient every step, which is why constant LR
   random-walked (paper Sec. 3.6).
4. **Teacher-side best-of-N is a one-off shift of the target distribution; the field moved selection
   on-policy.** Every 2026 result where "which sample we train on" still matters at convergence selects among
   the *student's* samples with a reward (DrPO 2606.02521 K=24; RTDMD 2605.26108 groups of 24;
   Diffusion-Sharpening 2502.12146 n=3; BRTS 2605.09725 N=4 with correctness-first, student-nearest-second
   selection). Our own factorial (README) already shows the exact RGB reward on the student's predictions is
   the one addition that beats argmax, and the converged schedule shrank teacher-side selection to +0.002.

## 1. Ranked list of concrete changes to OUR recipe

Costs are relative to the current converged 3k run (3,000 updates × 16 captions, 4 GPUs). "Additive with
selection" = whether the change leaves the DINO-photo selection branch intact and orthogonal.

### 1. Put the student on its deployment σ-grid (free evaluation first, then a nested teacher grid)

- **Mechanism.** Train/test mismatch in σ at every inference step (Sec. 0, point 2). The student's velocity at
  σ=1.0 is never supervised under K={4..7} (only teacher-initialised weights carry it), and σ=0.857/0.601 are
  interpolated between trained inputs. CFG-Zero* (2503.18886) shows the first step's velocity is the least
  reliable one in flow models, and Decoupled DMD (2511.22677) / Phased DMD (2510.27684) / CrossDistill
  (2609.14725) all find that high-noise steps decide layout and counting, i.e. exactly the CompBench/GenEval2
  categories where our student is weakest.
- **Evidence.** SD3.5-Flash (2509.21318, Sec. 4.2, Fig. 7): computing the objective on the student's own
  trajectory timesteps ("timestep sharing") instead of random re-noised timesteps "improves image composition
  and generation quality"; removing it gives "poor texture, colour and composition". CDM (2605.06376, Table 2b):
  fixed inference-schedule anchors vs their dynamic continuous schedule on SD3-Medium 4-step, HPSv3 9.482 vs
  9.561 and DPG 83.84 vs 85.26. SenseFlow (2506.00523, App. B.1) uses the deployment sigmas as coarse anchors
  ({0.512, 0.759, 0.904, 1.0} for FLUX).
- **Implementation (three levels).**
  (a) *Zero training cost:* evaluate existing checkpoints on a 4-step grid that IS a subset of the 8-step
  training grid, e.g. states {0,2,4,6} → σ = [1.0, 0.882, 0.692, 0.333, 0] passed as `sigmas=` to
  `scheduler.set_timesteps` in `common/sampling.py`. This changes only inference. (b) *Re-cache on a nesting
  teacher grid:* K∈{7,10,13,16} nests the 4-step grid exactly (report line 1617); K=10 costs +25% per
  candidate rollout and keeps the supervised window at the same σ range. (c) *Continuous σ coverage without
  re-caching:* on the 8-step chord the state at any σ between z_k and z_{k+1} is z_k + (σ−σ_k)·v̄_k (Euler
  segments are straight), so the student can be placed exactly at the deployment σ's plus jitter
  (`--student_sigma_jitter`), including σ=1.0.
- **Expected effect / risk.** Unknown size; this is the one structural defect the report itself documents and
  never tested. Risk nil for (a); (b)/(c) low. Sub-grid (a) has a coarse last step (0.333→0) which may cost
  fidelity; that is what (b)/(c) fix.
- **Additive with selection:** yes (selection is per caption, independent of grid).

### 2. Add a reverse-divergence "sharpening" term on the student's own rollouts, keeping our trajectory loss as the anchor (rCM / CrossDistill / Decoupled-DMD pattern)

- **Mechanism.** Our loss is forward-divergence and mode-covering; the reverse-KL term (teacher score minus
  fake score, or the cheaper CFG-augmentation term) is mode-seeking and is what lets 4-step students exceed
  their teachers on fidelity. Decoupled DMD (2511.22677) proves the engine of DMD in T2I is the term
  (α−1)(s_cond − s_uncond) applied to the student's output re-noised to τ — which needs **no fake network**,
  only two frozen-teacher forwards — and that distribution matching / trajectory losses are the regulariser
  that stops it collapsing. CrossDistill (2609.14725) shows the right way to combine: trajectory loss on the
  high-noise interval, distribution matching on the low-noise interval; reversing the order or mixing
  loss-level at every σ is worse (VBench total 83.77 vs 82.04 reversed; diversity 86% of teacher vs 57% for
  pure DMD). rCM (2510.08431) reaches DMD2 quality from a consistency base with λ=0.01 on the DMD term.
- **Evidence (same teacher family).** SD3.5-Flash on SD3.5-Medium, 4 steps: GenEval 0.70 vs 0.64 for the
  50-step teacher (Table 2), with 2k TG iterations + 800 DMD+GAN iterations. Flash-DMD (2511.20549, Table 2) on
  SD3-Medium with LoRA, TTUR 2, 4k iters × batch 32: ImageReward 1.019 vs teacher 1.017 (28 NFE, CFG 7). CDM
  (2605.06376) on SD3-Medium 4-step: DPG 85.26 vs teacher 85.04, HPSv3 9.56 vs 8.19. RTDMD (2605.26108):
  SD3-M 4-step PickScore 22.86 vs DMD2 22.14.
- **Implementation.** Stage 2 after the current run (or joint with weight λ): backward-simulate 1-4 student
  steps from noise (stop-grad except the last, DMD2/rCM style), take the clean estimate x̂0, re-noise to τ with
  τ drawn from the *high-noise* range for CA (Decoupled DMD: τ_CA > t; Phased DMD: DMD at fixed τ=0.357 fails,
  τ=0.882 works), compute Δ = (w−1)(v_T(z_τ,c) − v_T(z_τ,∅)) and regress x̂0 onto sg[x̂0 + λΔ/‖Δ‖-normalised].
  Cost: +2 teacher forwards per update, no extra 2B model in memory. Add the fake-score DM term only if CA-alone
  shows the documented over-saturation drift (Decoupled DMD Fig. 2: with CA alone, over-saturation and
  high-frequency noise build up over a few thousand iterations and training eventually collapses). Weight: rCM λ=0.01 relative to the consistency loss; TDM/Flash-DMD use the iCT pseudo-Huber
  form on this term.
- **Expected effect / risk.** This is the lever with the largest reported effect on absolute quality
  (FID/CMMD and GenEval both). Risk: mode-seeking reduces diversity/recall (rCM Fig. 7, Phased DMD Table 3);
  keep our trajectory loss on the high-noise states as CrossDistill prescribes. Departs from the paper's
  "consistency-only" framing.
- **Additive with selection:** orthogonal in mechanism (selection lives in the trajectory branch). Empirically
  expect selection's marginal value to shrink further, as it did when optimisation improved.

### 3. Supervise the trajectory loss in a perceptual feature space (decoded x̂0 vs decoded teacher x̃0 through DINOv2/ConvNeXt) — our exact-reward machinery pointed at the teacher target

- **Mechanism.** PFM (2607.03524) shows that x0-regression in VAE-latent L2 is mean-seeking (blurry at high
  noise) while regression in a pretrained perceptual space is mode-seeking, and that this alone turns a
  flow-matching model into a 4-8 step generator; their off-manifold penalty R_φ is 1.59 for the SD3 latent
  space vs 1.94 (DINOv2) / 2.03 (ConvNeXt). Our own result that the *exact* decode+DINO reward beats the latent
  projector (+0.011 CompBench, CMMD 0.80→0.69, README) is the same effect. Our high-σ Tweedie targets are
  themselves blurry one-step estimates; a perceptual distance penalises the student's blur less than the
  target's misplacement.
- **Evidence.** PFM Table 6 (SD3-Medium, 8-step, COCO-2014): ConvNeXt CLIP 31.53 / HPSv3 10.17, DINOv2
  29.76 / 7.46, VGG 29.50 / 5.06, random ViT 20.94 / −9.49 (the pretrained space is what matters); middle
  layers averaged beat deepest layers. Hyper-SD (2404.13686) adds a perceptual/instance-segmentation feedback
  term for the same reason.
- **Implementation.** Cache the decoded teacher targets' features once per (caption, k) (3k × 4 states ≈ 12k
  decodes at 61 ms each, memory note "decode is cheap"); per update decode the student's x̂0 at the supervised
  states (the `--reward_mode rgb` path already does the differentiable decode+DINOv2 pass for 2 states at
  14-16% wall-clock; 4 states ≈ 30%). Loss = pseudo-Huber on the latent + λ_p · (1 − cos) or L2 on mid-layer
  ConvNeXt/DINOv2 features. Start with λ_p set so the two gradient norms match (the 40-update probe used for
  the reward).
- **Expected effect / risk.** Moderate-to-large on CMMD/FID (PFM), likely positive on colour/texture/shape as
  the exact reward was; risk of DINO-specific texture bias (PFM Fig. 7: each backbone leaves a signature —
  prefer ConvNeXt or a VGG+DINO mix). 
- **Additive with selection:** yes, and it reuses the reference-photo DINO embedding pipeline.

### 4. Fix the loss geometry: per-state adaptive weighting, a residual-scaled c, and a clip that only fires on spikes

- **Mechanism.** (i) The always-active clip makes every update a unit-norm step (Sec. 0, point 3): raise the
  threshold to the ~90th percentile of the logged pre-clip norm (100-300) so it only catches spikes, and let the
  cosine schedule do the annealing; with Adam this changes the *relative* weighting across steps, not the step
  size. (ii) Per-noise-level balancing: sCM (2410.11081) and EDM2 (2312.02696) learn a weight per noise level
  by uncertainty, L = Σ_k e^{−u_k}·ℓ_k + u_k — five scalars for our five states; iCT weights low noise more
  (1/(σ_{i+1}−σ_i)); ECT reports no universal timestep weighting but large gains from any scheme that equalises
  gradient variance (ImageNet-64 1-step FID 14.34 → 9.28 → 5.51 across schedule/dropout/weighting changes).
  (iii) c: iCT/ECT say c should be of the order of the typical residual so that small residuals (low noise)
  get L2 gradients and large ones L1; sweep c ∈ {0.14, 5, 20} in latent units. TDM (2503.06674) found the
  pseudo-Huber surrogate worth +0.46 HPS over L2 in a distribution-matching context.
- **Evidence.** sCM Fig. 5a-b: tangent normalisation (or clipping of the *tangent*, not the parameter
  gradient) and adaptive weighting each visibly lower 1- and 2-step FID at ImageNet-512 across training; EDM2: optimal EMA length ∝ 1/(α_ref²·t_ref) and the loss magnitude per noise level drifts during
  training, which is why fixed weights are wrong late in training. No paper directly studies "clip always
  active"; Defazio (2506.02285) shows gradient-norm growth from weight-decay/normalisation interactions (we
  use wd=0, so not our case) and NFNets (2102.06171) replace global clipping by per-unit adaptive clipping.
- **Implementation.** `--clip 30` (from the norm histogram) + `--lr 5e-6`; `--state_weight learned`;
  `--huber_c` sweep. All free.
- **Expected effect / risk.** Small-to-moderate (+0.00-0.01 CompBench); the main value is removing a
  degeneracy that makes LR the only knob. Risk: none beyond a few short runs.
- **Additive with selection:** yes.

### 5. Scale the effective batch and update count, not the caption pool; and use all five COCO captions per photo

- **Mechanism.** Every distillation of a 2B+ rectified-flow teacher in the literature uses batches of 128-1120
  and 2k-30k updates: SD3.5-Flash TG stage batch ≈1120 × 2k iters (≈2.2M samples; ours is 48k), rCM 2B T2I
  batch 512 × 30k, SANA-Sprint 512 × 20k, sCM "same batch as the teacher", CDM 128 × 4k. ECT scales LR by
  sqrt(n) with batch. Our own batch 4→16 gave +0.005 to both arms. On data: i1 (2606.11289, Fig. 20, Table 7)
  shows subsampling from 13.7M to 0.4M images (18.7 passes) costs almost nothing while 5 captions per image
  is "a stronger boost under limited image data"; Qwen-Image-Flash (2606.03746, Table 1) shows a coherent
  20k-prompt set beats a 60k mixture; the distillation scaling law (2502.08606) has student loss a power law in
  distillation tokens for a fixed teacher. Our "16 passes over 3k beat 2 passes over 114k" is the same
  finding: passes/updates, not distinct captions, are the binding constraint.
- **Implementation.** Batch 64-128 (ACCUM or GPUs), lr 2e-5 to 3e-5 (sqrt scaling from 1e-5@16), 6k-12k
  updates cosine; and build the cache for the 5 captions of each of the 3k photos (5× cache cost, same
  reference photos, same DINO embeddings) so the 15k-caption pool keeps the photo pairing that selection needs.
- **Expected effect / risk.** This is the most likely route to "data matters again": at 48k samples the
  student is optimisation-limited; once it saturates 3k×5 captions, the 118k pool should start to pay. Risk:
  compute (4-8× current).
- **Additive with selection:** yes.

### 6. Better teacher trajectories at the same 8 NFE: CFG-Zero*, guidance interval / APG, or autoguidance

- **Mechanism.** The student inherits the teacher's trajectory bias; w=7 is above SD3.5-M's default (4.5) and
  the 8-step Euler teacher is itself only 0.454 CompBench. Better guidance at equal cost lowers the teacher's
  CMMD without touching NFE.
- **Evidence.** CFG-Zero* (2503.18886, Table 4, 10 images/prompt) on SD3.5: colour 0.76→0.78, shape
  0.59→0.60, texture 0.70→0.71, spatial 0.27→0.28, aesthetic 6.96→7.10 — zero-init of the *first* step
  only (more hurts SD3.5) plus a projected uncond scale. Guidance interval (2404.07724) and APG (2410.02416)
  reduce over-saturation at high w. AYF (2506.14603, Table 5): distilling from an autoguided teacher gives
  4-step FID 1.70 vs 2.32 from a CFG teacher (ImageNet-512). sCM and SANA-Sprint randomise the teacher
  guidance scale and condition the student on it (SANA-Sprint: CFG embedding +0.94 CLIP).
- **Implementation.** Only `common/sampling.py` and the cache change (zero the first Euler step's velocity;
  APG/interval on steps 2-8). Optionally add a guidance-scale embedding to the student and train on w∈{4.5, 7}.
- **Expected effect / risk.** CMMD/FID down, CompBench colour/shape +0.01-0.02 (CFG-Zero* numbers). Caveat
  from our own ablation: a stronger teacher halves the within-caption score spread and removes the selection
  gain, so this trades the paper's contrast for absolute quality.
- **Additive with selection:** mechanically yes; empirically it competes with it.

### 7. A short post-distillation preference stage on the 4-step student with an evaluator-disjoint reward (this is where "selection" comes back)

- **Mechanism.** Selection among the *student's own* samples by a reward, converted into an update
  (DrPO's feature-space drift, RTDMD's SubGRPO, PSO's pairwise margin, Diffusion-Sharpening's SFT/DPO), is the
  form of best-of-N that still moves a converged student. Hyper-SD's human-feedback stage exists to "recover
  the loss incurred by distillation".
- **Evidence.** With GenEval itself as reward (circular for GenEval, informative for headroom): SD3.5-M
  4-step GenEval 0.61 (TDM) → 0.92 (TDM-R1 2603.07700) → 0.94 (RTDMD 2605.26108); Flow-OPD (2605.08063) 0.63
  → 0.93 with OCR 0.59 → 0.93 and CompBench++ colour 0.799 → 0.830, shape 0.567 → 0.629, numeracy 0.593 →
  0.684. With disjoint rewards: DrPO on SD-Turbo (2606.02521, Table 3) count 33.8 → 42.5, two-object 46.5 →
  55.6, position 8 → 13; online PSO on SDXL-DMD2 (2410.03190) PickScore 22.35 → 22.73, ImageReward 0.936 →
  0.977 with 4k prompts; Flash-DMD (2511.20549) shows running the RL loss 5:1 against the continuing
  distillation loss prevents the "oil-painting" reward hacking that post-hoc RL on distilled models shows.
- **Implementation.** Keep the distillation loss on (Flash-DMD's regulariser), sample K=8-24 student rollouts
  per caption, score with the DINO-photo score (already built) and/or VQAScore/HPS, use DrPO-style drift
  (no reward gradient needed, so any scorer works) or SubGRPO. Evaluate on CompBench/GenEval2 only with
  rewards that do not share the evaluators' models (BLIP-VQA/UniDet/CLIP for CompBench, Qwen3-VL for GenEval2).
- **Expected effect / risk.** +0.01-0.03 CompBench on categories the reward sees; fidelity/diversity loss
  (ReNFT 2609.00061 documents mode collapse under reward post-training) — monitor recall/CMMD.
- **Additive with selection:** it *replaces* teacher-side selection with on-policy selection; the two can
  coexist (BRTS keeps both branches).

### 8. Split-timestep (two-branch) fine-tuning with EMA and weight interpolation — the cheap composition boost SD3.5-Flash reports

- **Mechanism.** Duplicate the student, fine-tune one branch on high-noise states and one on low-noise states
  (each with EMA 0.99 anchoring to the start point), then interpolate 3:7. Related: Hyper-SD's segment
  curricula; Phased DMD's per-phase experts; our own finding that averaging checkpoints is worth +0.015-0.02.
- **Evidence.** SD3.5-Flash (2509.21318, Sec. 4.2): "a distinct jump in model performance" on GenEval for
  the 4-step model, 400 iterations, 4 h on 8 H100. Merge ratio selected on GenEval (so treat their gain as an
  upper bound).
- **Implementation.** Two 500-update continuations of the final checkpoint with `WINDOW` split at σ≈0.55,
  `train/average_checkpoints.py` with weights.
- **Expected effect / risk.** Unknown for our loss; cheap; the interpolation ratio must be chosen on a
  held-out split, not on CompBench.
- **Additive with selection:** yes.

### 9. Reference-photo-conditioned VFM discriminator (SenseFlow-style LADD) as the reverse-divergence term

- **Mechanism.** SenseFlow's discriminator D(x, c, r) takes DINOv2+CLIP features of the generated image, the
  text, and a *real reference image* r; our per-caption photograph is exactly that r, so selection's scorer
  and the discriminator share one DINOv2 pass. Adversarial terms are the second standard way to add
  mode-seeking pressure (SANA-Sprint: sCM 8.93 FID → 8.11 with LADD, LADD alone 12.20; PCM +0.03 HPS at 1-2
  steps from λ=0.1 adversarial).
- **Evidence.** SenseFlow (2506.00523): vanilla DMD2 collapses on SD3.5-Large, the VFM discriminator is what
  makes training produce images at all; SD3.5-L 4-step GenEval 0.7098 vs teacher 0.7140. NitroFusion /
  SD3.5-Flash re-initialise heads (p=0.005 per step) against discriminator overfitting.
- **Implementation.** Hinge GAN on the decoded x̂0 at the low-noise states, weight (1−σ)² (SenseFlow Eq. 15),
  small λ, head re-init.
- **Expected effect / risk.** Fidelity gains comparable to item 2 but with GAN-tuning risk (Decoupled DMD:
  GAN-as-regulariser collapsed after 4k iterations). Rank below item 2; try only if the CA/DM term fails.
- **Additive with selection:** yes, and reuses the photo.

### 10. Replace the objective: sCM/rCM continuous-time consistency (Phase M) — expensive, and only worth it with the score regulariser

- **Mechanism.** Continuous-time consistency removes discretisation error and the σ-grid problem entirely;
  sCD scales like the teacher (sCM Fig. 6).
- **Evidence.** rCM (2510.08431): pure sCM at 2-14B T2I has fine-detail/blur problems and needs λ=0.01 DMD;
  lr 1e-6, batch 512, 30k iterations, semi-continuous time derivative (finite difference Δt=1e-4) for 2B T2I;
  FlashAttention-JVP kernel. The Practical Guide (2512.13006) on FLUX.1-lite 8B: sCM saturates at 3k
  iterations at GenEval 52.3 (4-step) vs teacher 53.6, MeanFlow needs 25k and collapses below 4 steps; both
  needed timestep-input rescaling and CFG mixing in the target. SANA-Sprint needed QK-norm and dense time
  embeddings to stabilise.
- **Assessment.** As a *replacement* it is 10× our compute and the published 4-step numbers are at or slightly
  below the teacher; as an *addition* its useful part is the score regulariser, which item 2 gives without JVP.
  Keep Phase M as a baseline, not as the route to a better student.
- **Additive with selection:** selection does not transfer (sCM trains on fresh noise, not cached
  trajectories).

## 2. Do-not-bother list

| Idea | Why not |
|---|---|
| Softer/Boltzmann weighting or more (N=8) teacher candidates | Our nulls (README); BRTS 2605.09725 shows teacher rollouts are strongly correlated so the catch rate saturates well below 1−(1−p)^N; the field's gains come from correctness-filtering *plus* student-alignment, not from soft weights. |
| Regress onto the reference photograph in latent space | Our −0.03 CompBench; PFM 2607.03524 explains why (latent L2 to a real sample is mean-seeking); D-OPSD 2605.05204 shows SFT on data states destroys few-step ability (Table 4: SFT on Z-Image-Turbo GenEval 0.16 vs 0.72 for on-policy self-distillation on the same data). |
| EMA-of-student teacher | Our collapse at 0.999 / inert at 0.9999; D-OPSD only makes a self-teacher work with privileged context (the target image through a VLM encoder) — SD3.5's CLIP/T5 encoders cannot take images. |
| Bigger caption pool at the same update budget | Our 118k null; i1 2606.11289 and Qwen-Image-Flash 2606.03746 both find distinct-image count is the least important axis; more coherent data can even hurt. |
| 2-step chord / endpoint targets in our parameterisation | Our ablation (report Sec. "Target horizon": 0.4646 / 0.4565 vs 0.4738 for the 1-step target); endpoint-type targets only work with a flow-map/CTM parameterisation (AYF 2506.14603, ArcFlow 2602.09014), which is item 10 territory. |
| Segment-velocity (progressive-distillation) objective | Our −0.05 CompBench; SD3.5-Flash notes PD "cannot learn extreme low-step inference". |
| Prompt-aware ranking heads / shaped projector rewards | `docs/rank/README.md`; DrPO's own ablation (raw latents PickScore 20.5 vs 23.6 with a pretrained feature map) says the feature space must encode the attribute — text-shaped DINO did not. |
| Latent projector surrogate rewards | Our null; the exact decoded reward works (README) and PFM shows why (pretrained pixel-space features, not a 10M latent projector). |
| GAN without a distribution-matching or trajectory anchor | Decoupled DMD 2511.22677 (collapse after 4k iters), SANA-Sprint (LADD-alone FID 12.2 vs 8.1 combined). |
| Pure sCM at T2I scale without a score regulariser | rCM 2510.08431 Sec. 3.3. |
| GenEval-as-reward then reporting GenEval2 | Evaluator circularity; Flow-OPD/RTDMD/TDM-R1 numbers are with the evaluator in the loop. GenEval2's Soft-TIFA uses Qwen3-VL-8B, so a Qwen-VL judge reward is equally circular for our GenEval2 numbers. |
| Guidance at student inference | Our "halves its scores"; PCM 2405.18407 proves guided distillation makes CFG at inference apply w·w′. |
| A post-trained/RL teacher as the *only* teacher for distillation | Qwen-Image-Flash 2606.03746 Sec. 4.3: destabilises DMD; use it only at the last student step (their λ schedule). |

## 3. Answers to the five questions

### Q1. State of the art in few-step distillation of rectified-flow / DiT models

Numbers are as reported by each paper under its own protocol; they are not comparable across rows.

| Family / paper (arXiv) | Teacher → student | Reported | CFG handling |
|---|---|---|---|
| SD3.5-Flash 2509.21318 (DMD + latent multi-head GAN on the fake model's features, TG pre-training, timestep sharing, split-timestep FT) | SD3.5-M 50 → 4 / 2 steps | GenEval 0.70 / 0.70 vs teacher 0.64; IR 1.10 vs 0.91; FID 29.8 vs 20.1 (COCO-30k); SD3.5M-Turbo (TensorArt checkpoint) 0.54 | teacher score with CFG in DMD; synthetic data from SD3.5-L at CFG 4 |
| SenseFlow 2506.00523 (DMD2 + IDA + ISG + VFM discriminator) | SD3.5-L 80 → 4; FLUX.1-dev | GenEval 0.7098 vs 0.7140 (teacher) vs 0.6877 (SD3.5-L-Turbo); FLUX 0.647 vs 0.670 | CFG in real score |
| Flash-DMD 2511.20549 (timestep-aware DM high-noise + pixel-GAN low-noise + joint latent RL) | SD3-M 28 → 4 (LoRA) | ImageReward 1.019 vs 1.017 teacher; 4k iters × batch 32 | CFG 7 teacher |
| CDM 2605.06376 (continuous-time DMD, dynamic schedule, off-trajectory matching) | SD3-M 100 → 4 | DPG 85.26 vs 85.04; HPSv3 9.56 vs 8.19; D-DMD 9.18; DMD2 8.42 | CA term (CFG 7) + DM |
| Decoupled DMD 2511.22677 | Lumina-Image-2.0, SDXL | CA is the engine, DM the regulariser; decoupled τ schedules: DPG 83.9 → 85.9, HPSv2.1 30.6 → 32.3 | CA = (α−1)(s_c − s_∅) |
| RTDMD 2605.26108 (AC-DMD + SubGRPO) | SD3-M, SD3.5-M, FLUX.2-4B → 4 | SD3-M: PickScore 22.86 vs DMD2 22.14; SD3.5-M GenEval 0.94 (GenEval reward) | no CFG at inference |
| TDM 2503.06674 (trajectory distribution matching, data-free) | PixArt-α, SDXL, SD1.5 | PixArt 4-step beats teacher on HPS/user study after 500 iters (2 A800 h); pseudo-Huber surrogate +0.46 HPS | CFG in real score |
| Phased DMD 2510.27684 (per-phase experts, subinterval score matching) | Qwen-Image-20B, Wan | diversity DINOv3-sim 0.782 vs DMD2 0.826 (Wan T2I); SGTS collapses to 1-step | CFG 4 teacher |
| rCM 2510.08431 (sCM + λ=0.01 DMD) | Cosmos 2B/14B, Wan 1.3B/14B | matches DMD2 on GenEval/VBench, better diversity; 1-4 steps | CFG 4.5-5 in teacher |
| SANA-Sprint 2503.09641 (sCM + LADD, CFG embedding) | SANA 1.6B 20 → 1-4 | GenEval 0.74 / FID 7.59 at 1 step (FLUX-schnell 0.71 / 7.94); LADD +0.8 FID | CFG ∈{4,4.5,5} embedded |
| Practical Guide 2512.13006 | FLUX.1-lite 8B 28 → 1-4 | sCM 52.3 (4-step) / 52.8 (2) / 43.3 (1) vs 53.6; MeanFlow 51.4 at 4, collapses at 1-2 | CFG velocity as target; MeanFlow needs improved-CFG mix |
| AYF 2506.14603 (continuous flow maps, autoguidance, adv. finetune) | EDM2 ImageNet; FLUX.1 LoRA | 4-step FID 1.70 (autoguided) vs 2.32 (CFG teacher); sCD degrades beyond 4 steps | autoguidance |
| ArcFlow 2602.09014 (momentum-mixture non-linear flow, LoRA) | FLUX.1-dev, Qwen-Image → 2 | FLUX GenEval 0.65 vs 0.66, DPG 84.3 vs 84.2 | teacher velocity (CFG) matching on mixed teacher/student integration |
| TwinFlow 2512.05150 / π-Flow 2510.14974 | Qwen-Image-20B, FLUX → 1-4 | GenEval 0.83 at 1 NFE (TwinFlow); π-Flow better diversity than DMD at 4 NFE | self-adversarial / policy imitation |
| PCM 2405.18407, Hyper-SD 2404.13686, TCD 2402.19159, LCM 2310.04378 (discrete CD) | SD1.5/SDXL | PCM 4-step SDXL FID-SD 6.6 vs LCM 11.2; Hyper-SDXL 4-step IR 0.93 vs LCM 0.48 (with human feedback) | CFG-augmented solver (limits inference CFG) |
| OPD family: Flow-OPD 2605.08063, DiffusionOPD 2605.15055, DanceOPD 2606.27377, STEP-OPD 2608.04887, D-OPSD 2605.05204 | SD3.5-M (multi-step, LoRA) | GenEval 0.63 → 0.93-0.96 with GRPO teachers; STEP-OPD +0.034 over DiffusionOPD from output extrapolation + representation-change alignment | teacher CFG velocity along the student's trajectory |

What is additive with trajectory selection: items 2-9 above all leave the selection branch intact. What
would *replace* our x0-consistency loss: sCM/rCM (item 10), MeanFlow distillation (2606.11155 shows MFD 4-step
beating DMD2/SenseFlow on SANA on FID-DINOv2 33.3 vs 36.1 with LoRA in 3k steps), pure DMD/TDM, and flow-map
(AYF/ArcFlow) parameterisations. Every industrial recipe that started from a trajectory loss (SD3.5-Flash,
Hyper-SD, Flash-DMD, AnyFlow 2605.13724) kept it only as a warm start or as the high-noise objective.

### Q2. Optimisation

- **Gradient clipping always active.** No paper studies this directly. The consistency-model literature
  controls gradient scale at the *target* instead: tangent normalisation g/(‖g‖+0.1) and learned per-noise
  weights (sCM 2410.11081), pseudo-Huber as adaptive per-sample scaling (ECT 2406.14548 App. A; iCT
  2310.14189 shows it halves the variance of Adam update norms). With Adam an always-on global clip is a
  per-step normalisation: it does not change the step size (Adam is scale-invariant) but erases the
  information in the gradient magnitude, so the loss cannot modulate the step and the schedule must — which is
  what Sec. 3.6 found. Recommendation in item 4.
- **LR / batch.** Distillation lrs on 2B+ flow models cluster at 1e-6 (SD3.5-Flash TG stage, rCM, SenseFlow,
  Qwen-Image-Flash, Hyper-SD) to 1e-5 (CDM full FT) with batches 64-1120; sCM uses 0.01× the teacher's lr
  and the teacher's batch; ECT scales lr by sqrt(batch). Our 1e-5 at batch 16 is on the aggressive side.
- **EMA.** Every sCM/EDM2 recipe keeps a student EMA (σ_rel 0.05); EDM2 shows the optimal EMA length grows
  with training length and scales as 1/(α_ref²·t_ref); our avg-last-5 is the discrete analogue and raw≈avg
  under cosine, so this is settled for us.
- **Loss weighting across noise levels.** iCT: 1/(σ_{i+1}−σ_i) (low noise emphasised, "errors at low noise
  propagate"); sCM/EDM2: learned uncertainty weights; min-SNR 2303.09556 for diffusion training; ECT: no
  universal answer, but variance balancing matters. Our L1-norm gives equal *gradient* norm per state while the
  noisiest target carries a third of the *loss* — a learned per-state weight resolves the ambiguity.
- **Noise-level window.** Decoupled DMD (τ_CA > t), Phased DMD (DMD fails at τ=0.357 alone, works at 0.882),
  CrossDistill (trajectory loss on [0.94,1], DM below), Flash-DMD (DM at high noise, GAN at low noise) agree:
  high-noise states need the structural (trajectory) signal, low-noise states the sharpening signal. Our
  K={4..7} window drops the highest-noise *target*; item 1(c) restores σ=1 as an *input*.
- **Target horizon.** 1-step teacher chords are the lowest-variance target (our ablation agrees); multi-step
  integrals are used only as warm starts (SD3.5-Flash TG) or with non-linear parameterisations (ArcFlow's
  analytic momentum integration; AYF/MeanFlow with JVP). sCM removes the horizon question entirely but needs
  the score regulariser.
- **Teacher discretisation error and beating the teacher.** Students exceed their teachers routinely in
  2025-26 (SD3.5-Flash GenEval 0.70 vs 0.64; CDM/Flash-DMD on SD3-M; Qwen-Image-Flash; TDM PixArt) but only
  when a reverse-divergence/CA term is present; trajectory-only students land at or below the teacher
  (Practical Guide sCM 52.3 vs 53.6; ArcFlow 0.65 vs 0.66). Our 0.486 > 0.454 (8-step Euler teacher) comes
  from guidance absorption plus x0-prediction sampling; the 0.505 (28-step) ceiling is the trajectory the
  teacher never showed us — items 2/3/6 are the ways to exceed it.

### Q3. Data: does curating which teacher samples to distil help, and how does it interact with size/passes?

- **Direct evidence for best-of-N *teacher* curation is thin and small.** Diffusion-Sharpening (2502.12146,
  Table 1, SDXL): SFT on the best of n=3 three-step continuations (ImageReward/compositional reward) gives
  CompBench colour 0.658 vs 0.644 (standard fine-tune) vs 0.637 (base), shape 0.569 vs 0.577 vs 0.541,
  texture 0.573 vs 0.569 vs 0.564, spatial 0.212 vs 0.208 vs 0.203; n=3 is optimal (n=8 no better,
  Table 2); the DPO variant is larger (colour 0.684, texture 0.640, complex 0.450). BRTS (2605.09725, LLM
  reasoning): best-of-4 teacher rollouts +0.8 to +1.6 points mean accuracy over single-rollout OPD, gains
  come from correctness filtering and from picking the rollout closest to the student; teacher rollouts are
  correlated, so diversity (prompt perturbation) raises the catch rate more than N. TDM (2503.06674): SFT-ing
  the teacher on a compact high-quality set before distilling beats a GAN on real data (HPS 31.31 vs 29.80).
  Our own +0.014-0.017 CompBench at the constant-LR schedule is, as far as I can find, the largest reported
  effect of teacher-side selection — and it went to +0.002 at convergence.
- **Composition beats size.** Qwen-Image-Flash (2606.03746, Table 1): 20k portrait prompts (avg 3.42) >
  40k landscape+portrait (3.40) > 60k mixed (3.02) > 20k text-centric (2.63, worst even in-domain). i1
  (2606.11289): 0.4M images repeated ≈ 13.7M; 5 captions/image helps most when images are few; equal
  weighting across curated sets beats any up-weighting.
- **Passes vs distinct prompts.** Distillation scaling law (2502.08606): student loss is a power law in
  distillation tokens for a fixed teacher, with a capacity-gap regime where a better teacher hurts; the classic
  KD result (2106.05237) is that consistent, patient (long-schedule) distillation matters more than data
  volume. Our 16-passes-over-3k > 2-passes-over-114k is consistent with being far below the sample budgets
  used elsewhere (Sec. 1 item 5).
- **VLM-filtered synthetic data** is used for *base-model* fine-tuning (SELMA-style skill experts, i1's
  captioner choice: Qwen3-VL-30B captions beat Qwen2-VL-2B by 2-4 DPG points) rather than for distillation;
  in distillation the synthetic set is usually the teacher's own outputs at a higher step count and moderate
  CFG (SD3.5-Flash: SD3.5-L, 32 steps, CFG 4).

### Q4. Multi-candidate / multi-trajectory objectives

- **Soft weighting over teacher candidates:** no positive report; our Boltzmann null stands. BOND
  (2407.14622, LLM) distils the best-of-N *distribution* via a Jeffreys divergence rather than a hard pick —
  it needs likelihoods, which few-step generators lack.
- **Preference/contrastive objectives over candidates for few-step students:** PSO 2410.03190 (pairwise
  margin, online +0.4 PickScore), DrPO 2606.02521 (dipole drift from ranked candidates, no reward gradient,
  GenEval count +8.7 / position +5 on SD-Turbo), RTDMD/TDM-R1 (GRPO on the stochastic steps + pathwise on the
  deterministic last step), Diffusion-Sharpening (DPO between best/worst trajectories), region-aware bimodal
  DPO for composition (2605.28615). All operate on the *student's* candidates.
- **"Learning to select inside distillation":** BRTS (correctness-first, student-nearest-second selection of
  the teacher rollout inside the OPD loop, +λ teacher-context loss); STEP-OPD (extrapolate the target beyond
  the teacher along the base→teacher direction, α=0.01 for GenEval); Golden-noise / noise-hypernetwork lines
  (2411.09502, 2412.03895, 2508.09968) learn to pick or map the *initial noise* — which is what our seed
  selection does implicitly; a noise-mapping head trained on our selected seeds would make the selection
  survive as an inference-time module.

### Q5. Evaluation sanity

- **CompBench.** Official protocol is 10 images per prompt with per-prompt means (T2I-CompBench++
  2307.06350; CFG-Zero* and Diffusion-Sharpening both use 10 / 2 images). Our one-image sweep:
  per-prompt paired-difference sd 0.114 between equivalent runs → mean-based MDE 0.0065 for one run vs one run
  (2,398 prompts, α=0.05, power 0.8); training seed moves the mean by 0.0001-0.0018 under the converged
  schedule; the sign test is ~2× more sensitive (win-rate excess 0.031 detectable); 10 images/prompt would
  bring the MDE to ~0.002 (`docs/rank/debug/per_prompt_analysis.md`). Every converged-schedule arm contrast
  (+0.002 to +0.007) is therefore below the one-seed, one-image floor; claims need ≥3 seeds and 10 images.
- **GenEval2** (2512.16853). 800 prompts, 3-10 atoms, Soft-TIFA with Qwen3-VL-8B; report GM (prompt-level,
  the official number) and AM (atom-level). The paper does not fix images per prompt (its human study labels
  one image per prompt-model pair); our per-prompt sd is 0.175 → MDE 1.7 points for one run vs one run. The
  GM is heavy-tailed (36-44% of our prompts "collapse"), so seed sd of 1-1.5 points is expected. Reference
  points: SD3.5-L 22.8, SD3-M 21.3, FLUX.1-dev 21.1, Qwen-Image 33.8 (GM ×100); our students at ~23 are at
  SD3.5-L level by this judge, and the teacher's 17.6 is the geometric-mean artefact already noted.
- **What the field treats as meaningful for 4-step SD3-class students:** ~0.01 on a CompBench category mean
  (CFG-Zero* reports 0.01-0.02 per category), 1-2 GenEval points (STEP-OPD's +3.4 is a large effect; SD3.5-Flash
  vs teacher +6 is a headline), and RL-scale jumps (+30 GenEval) only with the evaluator as reward. Almost no
  distillation paper reports seeds; SenseFlow/CDM/Flash-DMD are single runs.
- **Evaluator families.** CompBench: BLIP-VQA / UniDet / CLIP; GenEval2: Qwen3-VL; VQAScore: CLIP-FlanT5.
  A DINO-photo or ImageReward/HPS reward is disjoint from all three; a GenEval-style detector reward or a
  Qwen-VL judge is not disjoint from GenEval2.

## 4. References (arXiv ids verified 2026-09-17)

Few-step distillation of flow/DiT models
- 2509.21318 SD3.5-Flash: Distribution-Guided Distillation of Generative Flows (Stability AI)
- 2606.03746 Qwen-Image-Flash: Rethinking the Training Recipe for Few-Step Distillation
- 2506.00523 SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation
- 2511.20549 Flash-DMD: Efficient Distillation and Joint Reinforcement Learning
- 2605.06376 Continuous-Time Distribution Matching for Few-Step Diffusion Distillation (CDM)
- 2511.22677 Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
- 2510.27684 Phased DMD: Few-step DMD via Score Matching within Subintervals
- 2503.06674 Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM)
- 2510.08431 Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency (rCM)
- 2503.09641 SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation
- 2512.13006 Few-Step Distillation for Text-to-Image Generation: A Practical Guide
- 2609.14725 CrossDistill: Balancing Quality and Diversity via Trajectory-Level Hybrid Few-Step Distillation
- 2506.14603 Align Your Flow: Scaling Continuous-Time Flow Map Distillation
- 2602.09014 ArcFlow: 2-Step Text-to-Image Generation via High-Precision Non-Linear Flow Distillation
- 2606.11155 Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models
- 2512.05150 TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows
- 2510.14974 pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation
- 2605.13724 AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation
- 2607.03524 Perceptual Flow Matching for Few-Step Generative Modeling (PFM)
- 2405.14867 Improved Distribution Matching Distillation (DMD2); 2311.18828 DMD
- 2403.12015 Latent Adversarial Diffusion Distillation (LADD); 2311.17042 ADD
- 2404.04057 Score identity Distillation (SiD); 2406.01561 Guided SiD (SiD-LSG); 2305.18455 Diff-Instruct;
  2405.16852 EM Distillation
- 2405.18407 Phased Consistency Models; 2404.13686 Hyper-SD; 2402.19159 TCD; 2310.04378 LCM; 2311.05556
  LCM-LoRA; 2403.06807 Multistep Consistency Models; 2310.02279 CTM; 2402.13929 SDXL-Lightning;
  2406.02347 Flash Diffusion; 2503.16397 Scale-wise Distillation; 2412.02030 NitroFusion;
  2202.00512 Progressive Distillation; 2210.03142 On Distillation of Guided Diffusion Models
- 2505.13447 MeanFlow; 2410.12557 Shortcut Models; 2406.07507 Flow Map Matching; 2605.17834 Stabilizing,
  Scaling & Enhancing MeanFlow for Large-scale Diffusion Distillation
- 2602.07345 Adaptive Matching Distillation; 2602.03139 Diversity-Preserved DMD

Consistency-model optimisation
- 2410.11081 Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models (sCM)
- 2406.14548 Consistency Models Made Easy (ECT)
- 2310.14189 Improved Techniques for Training Consistency Models (iCT)
- 2312.02696 Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2, post-hoc EMA)
- 2303.09556 Min-SNR weighting; 2506.02285 Why Gradients Rapidly Increase Near the End of Training;
  2102.06171 Adaptive gradient clipping (NFNets)

Guidance for the teacher
- 2503.18886 CFG-Zero*; 2404.07724 Guidance in a Limited Interval; 2406.02507 Autoguidance;
  2410.02416 APG; 2501.15420 Visual Generation Without Guidance (GFT)

On-policy distillation, preference and RL for (few-step) students
- 2605.08063 Flow-OPD; 2605.15055 DiffusionOPD; 2606.27377 DanceOPD; 2608.04887 STEP-OPD;
  2605.05204 D-OPSD; 2607.24731 Rethinking CFG in On-Policy Diffusion Distillation
- 2605.26108 Reward-Tilted Distribution Matching (RTDMD); 2603.07700 TDM-R1; 2511.13649 DMDR
- 2606.02521 Drifting Preference Optimization for One-Step Generative Models (DrPO)
- 2410.03190 Pairwise Sample Optimization (PSO); 2311.12908 Diffusion-DPO; 2605.28615 Region-aware
  bimodal DPO for compositional T2I
- 2502.12146 Diffusion-Sharpening; 2605.09725 On-Policy Distillation with Best-of-N Teacher Rollout
  Selection (BRTS); 2407.14622 BOND
- 2505.05470 Flow-GRPO; 2505.07818 DanceGRPO; 2509.16117 DiffusionNFT; 2609.00061 ReNFT
- 2403.11027 Reward Guided Latent Consistency Distillation
- 2411.09502 Golden Noise for Diffusion Models (note: 2411.09656 is an unrelated lattice-QCD paper);
  2412.03895 A Noise is Worth Diffusion Guidance; 2508.09968 Noise Hypernetworks

Data and scaling
- 2606.11289 i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models
- 2502.08606 Distillation Scaling Laws; 2106.05237 Knowledge distillation: a good teacher is patient and
  consistent

Evaluation
- 2512.16853 GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation
- 2307.06350 T2I-CompBench(++); 2310.11513 GenEval
- 2607.03256 A Decomposable Probe for Few-Step Diffusion Models (trajectory-rollout students show a
  low-strength score-layer spike; ADD students the lowest score selectivity)
