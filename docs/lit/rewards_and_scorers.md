# Rewards and scorers for scored consistency distillation: what the literature says we should do next

Scope. Literature scout (2023-2026, arXiv ids verified against the arXiv API on 2026-09-17) written
against the failure record of the prompt-aware ranking campaign (`docs/rank/README.md`,
`rank/debug/{vqa_ceiling,reward_state_probe,per_prompt_analysis}.md`). Our setting: SD3.5-Medium
teacher (8 Euler steps, CFG 7) distilled into a 4-step guidance-free student by x0 consistency
distillation on one of four teacher trajectories per COCO caption, chosen by DINOv2-B patch-mean
cosine to the caption's COCO photo, plus a reward `-lambda * cos(P(x0_hat), DINO(photo))` through a
refreshed latent-to-DINO projector P. On the converged schedule the reward family is worth +0.002
CompBench over naive CD (0.486 -> 0.488), and every attempt to make the reward prompt-aware with a
text ranking over structured negatives failed for four measured reasons:

* (a) the DINO-to-photo cosine is blind to the attributes the negatives change (per-negative
  correlation with a VQAScore judge r = 0.00 colour, 0.07 spatial, 0.10 count; only
  texture/verb/shape ~0.3);
* (b) the student cannot realise relation edits (spatial negatives are not contradictions even to
  VQAScore: 0.54 contradiction rate, chance 0.50);
* (c) back-propagating a ranking into the student games the scorer (CompBench 0.401 vs 0.488);
* (d) the ranking-shaped scorer used for selection is worse than raw DINO against the VQAScore
  oracle in the cache (agreement 0.31 vs 0.36).

Numbers below are quoted from the cited papers; "ours" numbers come from the campaign records.
Where a paper's table was read directly the numbers are exact; where only an abstract was available
the claim is marked (abstract-level).

---------------------------------------------------------------------------------------------------

## 0. One-paragraph verdict

Every reward or scorer that has moved colour binding, counting or spatial relations in the
literature reads the prompt (a VQA model, a captioner's likelihood, a detector with the noun list,
or the diffusion model's own text-conditioned denoising loss). No image-to-image similarity to a
reference photo has ever been reported to do so, and the probing literature explains why: colour is
linearly decodable from DINOv2 patch features but sits in a low-variance subspace that a
mean-pooled cosine ignores, and neither DINOv2 nor CLIP composes object-attribute bindings in a way
a global embedding distance can read (Section A). The fixes that transferred to T2I-CompBench /
GenEval are (i) a prompt-conditioned differentiable reward on decoded predictions with an explicit
fidelity anchor (CoMat, PromptEcho), (ii) a latent surrogate of such a judge, refreshed online, in
place of the DINO projector (LaSRO, RG-LCD's LRM, TDM-R1, LPO, DiT-Reward), (iii) bimodal DPO on
minimally-different image/caption pairs (BiDPO, D-Fusion, Di3PO), (iv) the model's own
text-conditioned denoising likelihood as a contrastive alignment signal with no external encoder
(SoftREPA, AGSM), and (v) verifiable detector rewards with a KL anchor for the few-step student
(Flow-GRPO, TDM-R1, RTDMD, REST). One framing fact matters for all of them: on SD3.5-Medium colour
is nearly saturated (CompBench colour 0.80 base, 0.81 for our student, +0.04 even under detector
RL), while 2D-spatial (0.21 ours vs 0.29 base vs 0.54 under RL) and numeracy (0.55 vs 0.59 vs
0.68) hold the headroom (Section C).

---------------------------------------------------------------------------------------------------

## 1. Ranked proposals for our pipeline

Ranking criterion: expected CompBench/GenEval2 gain per unit risk and implementation effort, given
what has already failed here. Cost is per training update at batch 16 relative to the current
~1 s update (the exact RGB reward path already costs 14-16% of an update, VAE decode 61 ms).

### P1. Replace the DINO-photo cosine in `--reward_mode rgb` by a prompt-conditioned differentiable reward, keep everything else

Mechanism. The exact-reward path (`--reward_mode rgb`: decode x0_hat, score, back-propagate) is
the one channel in this project that demonstrably works (+0.011 over argmax, +0.025 over random,
CMMD 0.80 -> 0.69). Its limitation is the target, not the path. Swap the scorer for one that reads
the caption:

* CoMat [2404.03653]: reward = log p_BLIP(caption | decoded image) (the captioner's
  token-level NLL), back-propagated through 5 of 50 steps, plus an "attribute concentration"
  cross-attention/segmentation term and a UNet discriminator that separates fine-tuned from
  original-model latents (fidelity anchor). SDXL T2I-CompBench colour 0.588 -> 0.783 (+0.195),
  shape +0.064, texture +0.117, spatial +0.030; concept-activation term alone gives colour +0.158.
  BLIP beat GIT and LLaVA as the captioner. Without the fidelity module FID rose 16.7 -> 19.0
  (reward hacking); with generated-latent discriminator + mixed latents it fell to 15.4.
* PromptEcho [2604.12652]: reward = negative cross-entropy of the prompt under a frozen VLM given
  the image ("does the image echo the prompt"), one forward pass, deterministic, continuous.
  Z-Image GenEval overall 0.75 -> 0.82: attribute binding 0.59 -> 0.73, position 0.41 -> 0.52,
  counting 0.75 -> 0.85, colours unchanged (0.86). Same VLM asked to emit a score
  ("InferScore") gives nothing (+0.01) because scores cluster and tie; the likelihood does not.
  An 8B VLM retains most of the gain (GenEval 0.77 vs 0.82 with 32B).
* Cycle Consistency as Reward [2506.02095] (abstract-level): a reward from re-captioning
  cycle consistency, no human preferences, same family of idea.

Why it addresses (a),(c),(d). (a) The NLL of the caption is by construction sensitive to every
token, including colour words and counts; PromptEcho moves attribute binding +0.14 where our
DINO channel moves colour +0.003. (c) Our own CD loss to teacher trajectories is a stronger
anchor than CoMat's discriminator (they had none by default and needed one); keep lambda at the
20% gradient share that was calibrated for the DINO reward and check CMMD every 500 updates.
(d) Selection stays raw DINO (or VQAScore, see P2); nothing here touches the selector.

Implementation in our terms. In `train/distill.py` `--reward_mode rgb`, replace the DINOv2
scorer with BLIP-base captioning NLL (CoMat recipe: `Salesforce/blip-image-captioning-base`,
384 px, teacher-forced caption) or Qwen3-VL-8B prompt cross-entropy (PromptEcho recipe;
forward-only if used as a weight, forward+backward if used as a gradient). Apply at the same two
least-noisy supervised states. Add `--reward_fidelity` = CoMat's discriminator on latents (real =
teacher x0_k, fake = student x0_hat) only if CMMD/FID move; our factorial says the DINO reward
improved fidelity, so start without.

Cost. BLIP-base forward+backward on 16 decoded 384-px images ~0.1-0.15 s on an A100 (vs DINOv2-B
~0.05 s); the decode is already paid. Estimated +5-10% per update. Qwen3-VL-8B forward-only at
512 px ~0.3 s per image; use it only as a per-sample weight or at refresh (see P2).

Expected effect and risk. Literature effect sizes are on SD1.5/SDXL, where colour has
0.15-0.20 headroom; on SD3.5-M colour headroom is ~0.03-0.04 (Section C), so expect the gain in
complex (+0.01-0.02), shape/texture (+0.01) and GenEval2 attribute atoms rather than in colour.
Risk: circularity with CompBench's BLIP-VQA evaluator (both BLIP family): report GenEval2
(Qwen3-VL judge) and VQAScore alongside, as the project's evaluator-family table already does.
Second risk: RG-LCD [2403.11027] found that direct optimisation of ImageReward produced
high-frequency noise invisible to resized reward inputs; watch precision/recall and CMMD, and use
P2 (a latent proxy) if it appears.

### P2. Re-target the projector P: a latent surrogate of a compositional judge, trained from the cache's VQAScore labels, refreshed online

Mechanism. Keep the cheap channel (`--reward_mode proj`, 2% of an update) but change what P
predicts. Today P regresses DINO(photo)-space; instead train a latent reward head to rank
candidates the way a compositional judge does, and refresh it on decoded recent predictions
exactly as now (16 AdamW steps every 100 updates).

* LaSRO [2411.15247]: learns a Bradley-Terry surrogate in SDXL latent space (UNet encoder
  backbone + conv head) from win/lose pairs labelled by an arbitrary, even non-differentiable,
  reward (ImageReward, an "attribute-binding score" on T2I-CompBench prompts, a captioning-based
  text-alignment score), alternates 1-10 generator steps with 1-50 surrogate-adaptation steps,
  normalises/clips the surrogate output, and regularises with the original LCM distillation loss.
  It fine-tunes 1-2-step LCMs where DDPO/Diffusion-DPO/PSO fail, and reports gains on the
  CompBench attribute-binding score over GORS-LCM (figure-only in the paper; exact numbers not
  tabulated). A pretrained latent-DM encoder beat CLIP and BLIP backbones for the surrogate.
* RG-LCD [2403.11027]: the latent proxy RM (a CLIP-style latent encoder) is fitted by KL to the
  expert RM's pairwise preferences over (real latent, student prediction, EMA target), which
  removes the high-frequency-noise hacking of direct ImageReward optimisation and gives both
  better FID (42.7 -> 17.2 at 4 steps for the ImageReward run) and better HPSv2.1 than the LCM.
* TDM-R1 [2603.07700]: for a 4-step TDM student of SD3.5-M, a diffusion-parameterised surrogate
  trained by group preference optimisation from a non-differentiable GenEval detector reward, plus
  a marginal (distribution-level) reverse KL to the teacher: GenEval 0.61 -> 0.92 at 4 NFE
  (counting 0.49 -> 0.88, position 0.23 -> 0.93, attribute binding 0.44 -> 0.91), without
  losing DrawBench quality; direct RL losses on the few-step model gave blurry outputs.
* LPO [2502.01051]: the diffusion model itself as a noise-aware latent reward model (UNet
  down/mid features + text features, CFG-style "visual feature enhancement", BT loss on
  Pick-a-Pic pairs filtered for agreement with VQAScore/CLIP/aesthetic gaps). Scores noisy
  latents at any t without decode. SDXL T2I-CompBench++: colour 0.583 -> 0.735, 2D-spatial
  0.194 -> 0.241, numeracy 0.487 -> 0.549, complex 0.333 -> 0.380; GenEval 49.4 -> 59.3;
  10-28x cheaper than Diffusion-DPO.
* DiT-Reward [2606.23626]: the SD3.5 MMDiT's own image-token states (layers 15/23/28/37,
  mean-pooled, near-clean latent, tau = 0.005) + MLP head, BT loss: a frozen-backbone head
  already reaches 72.3% on HPDv3 (HPSv3 76.9%); latent-interface scoring 54 ms/image vs 89 ms
  for HPSv3. Reward information peaks in middle-to-late layers.

Why it addresses (a),(c),(d). (a) The training target is a judge that sees colour/count/spatial
(VQAScore agreement with the CompBench oracle is what the cache already stores; the
`build_candidates.py --vqa` field `endpoint_vqa` gives ~470k scored teacher images at 118k
captions, and the 3k replay set has photos too). (c) The proxy path plus online refresh on the
student's own predictions is exactly the anti-hacking design of RG-LCD and LaSRO, and our
refresh loop already exists. (d) The same head, trained on true DINO/MMDiT features of teacher
candidates, is a candidate for the selection scorer; the acceptance test is offline agreement
with the VQAScore oracle above raw DINO's 0.356 (the ranking head got 0.311).

Implementation in our terms. Two variants, both reusing `train/latent_scorer.py`:
(i) P_vqa: same 10.7M projector, but the loss is a listwise Plackett-Luce over the 4 candidates'
`endpoint_vqa` (cardinal gaps kept; LAIR [2605.26491] shows cardinal > ordinal > top-1) on
noised latents at the reward's sigma range (0.34-0.55), refreshed on decoded recent predictions
re-scored by VQAScore (32 images per refresh: ~15-30 s with CLIP-FlanT5-XXL, negligible at every
100 updates). (ii) P_mmdit: a DiT-Reward head on the student's own mid-block tokens (free
features, adds only an MLP). Reward term unchanged: `-lambda * P(x0_hat)` with lambda re-tuned to
the 20% gradient share. Drop the DINO(photo) anchor entirely.

Cost. Variant (ii) ~0; variant (i) same as today (~2%). Offline: one PL fit on the cache
(minutes to an hour on one GPU).

Expected effect and risk. LPO's +0.15 colour on SDXL translates to +0.02-0.04 total CompBench
on SD3.5-M given headroom; the 4-step TDM-R1 result says the few-step regime is not the
obstacle. Risk: the surrogate must rank the student's one-step estimates, not teacher endpoints;
our reward_state_probe showed a head trained on rollouts collapses on x0_hat inputs. Train on
x0_hat-like inputs (noised teacher x0_k at sigma 0.34-0.55) from the start, as LaSRO trains on
1st- and 2nd-step LCM outputs, and monitor `reward_monitor.py` for the proxy-vs-true divergence
signature.

### P3. Bimodal DPO on our same-noise structured-negative pairs (BiDPO), replacing the ranking head

Mechanism. BiDPO [2605.28615] extends Diffusion-DPO to text preferences: for a positive image
x+ with caption y+ and an edited caption y-, the loser is (x+, y-) and the winner (x+, y+)
("TextDPO"); two such terms on an edited image x- yield implicit image preference. Pairs are
built by caption editing + image editing (Qwen-Image-Edit) and VQA-filtered; a region mask from
the edited noun's box weights the loss (region-level guidance). SDXL T2I-CompBench overall 43.6
-> 54.4 (+0.108): colour 58.9 -> 79.4, shape 46.9 -> 60.5, texture 53.1 -> 71.4, numeracy
50.1 -> 59.3, spatial 21.2 -> 23.4; GenEval 0.53 -> 0.62; DPG +5.5; HPSv2 +2.65. Ablation:
SFT on the pairs does nothing (43.3), ImageDPO alone +2.0 (and numeracy -10.7), TextDPO alone
collapses to 13.5, the bimodal combination gives +9.5 and region guidance +1.3 more. SD3-Medium
also improves on GenEval2. D-Fusion [2505.22002] and Di3PO [2602.06355] show independently that
DPO pairs must be visually consistent (same noise, same background) or the gradient is spent on
confounds; Di3PO's diptych pairs beat background-varying DPO pairs on OCR by 0.35 vs 0.25
substring match at best-of-4.

Why it addresses (b),(c),(a). (b) Only use the families the VQA ceiling certified as
contradictions realised by the student (vqa_ceiling "both": colour 0.93, texture 0.81, verb
0.68, shape 0.52; drop spatial/3d_spatial, keep count with delta >= 2). (c) Our arm A pushed
both sides of a ranking in DINO space with no reference model, which is the configuration BiDPO
found destructive (TextDPO-only 13.5); DPO's implicit reward is bounded by the reference and
the same-noise pairs cancel background gradients. (a) No DINO anywhere.

Implementation in our terms. The pairs exist: `negatives_3k_v3.json` plus the same-noise
student rollouts used for `eval/vqa_ceiling.py` (generate them with the teacher instead, so the
"winner" is a teacher trajectory and the "loser" its same-seed negative-caption trajectory).
Loss: Diffusion-DPO with our CD distance in place of epsilon-MSE,
`s(x, c) = d(x0_hat_theta(z_k-d, c), x0_k) - d(x0_hat_ref(z_k-d, c), x0_k)` with the frozen
initial student as reference, two TextDPO terms per pair, beta swept in {50, 100, 200} at the CD
scale, region mask from the teacher's cross-attention on the edited noun (free) or Grounded-SAM
(offline). Filter pairs by `S[neg][pos] < S[pos][pos]` from the VQA ceiling.

Cost. Two extra student forwards per pair (positive latent with negative caption; negative
latent with its caption) at the two least-noisy states: ~+60-100% per update if every sample
carries a pair, ~+20% if one pair in four. Offline: teacher rollouts of the negatives (already
budgeted for the VQA ceiling), one VQAScore pass.

Expected effect and risk. BiDPO's +0.11 overall on SDXL will compress on SD3.5-M; +0.01-0.02
CompBench with gains in texture/shape/complex and GenEval2 attribute atoms is the realistic
target. Risk: our negatives are rule-based and detectable by a text-only classifier
(Section B); BiDPO's captions were rewritten by a VLM from the image. Use LLM-rewritten,
adversarially filtered negatives (SugarCrepe recipe) or the VLM-recaptioned pairs of P7.

### P4. Contrastive denoising-likelihood alignment with in-batch caption negatives (SoftREPA / AGSM), no scorer at all

Mechanism. SoftREPA [2503.08250]: the logit for (image, caption) is
`exp(-||v_theta(x_t, c) - (eps - x0)||^2 / tau(t))`, a contrastive cross-entropy over the batch's
other captions, with <1M trainable soft tokens prepended to the text stream in the first 2-5
MMDiT layers. SD3 GenEval colours 0.85 -> 0.92, colour attribution 0.55 -> 0.68, position
0.27 -> 0.34, two-objects 0.86 -> 0.95, but counting 0.56 -> 0.29 (soft tokens over-emphasise
text and duplicate objects; restricting to layers 1-2 or adding a YOLO count loss recovers it).
AGSM [2605.30038] reformulates the same signal as a bounded Plackett-Luce alignment guidance
inside score matching with separate positive/negative soft tokens and EMA reward tokens: matches
SoftREPA on alignment, fixes the counting failure (+35% counting on GenEval), 3 in-batch
negatives suffice (Table 13: 1:3 beats 1:7), PL beats pairwise BT, and gamma- must be small on
SD3 (0.1). It is plug-and-play with Diffusion-DPO/DDPO.

Why it addresses (a),(d),(c). (a),(d) The likelihood of the student's own denoiser under a
wrong caption is prompt-aware by construction; no encoder, no photo. (c) AGSM's bounded form
was designed against exactly the unbounded negative push that broke SoftREPA late in training
and that our arm A exhibits.

Implementation in our terms. In the CD loss, for each sample at the two least-noisy states
compute `d_j = ||x0_hat(z_k-d, c_j) - x0_k||` for the true caption and 1-3 negative captions
(in-batch captions, or the v3 structured negatives for a harder variant), and add the AGSM
target: pull toward x0_k under c, push (with gamma- = 0.1) under c_j, weights w_j from the PL
softmax over `-d_j/tau`. Train either the full student (our setting) or soft tokens only (their
setting; 0.9M params, would also give a cheap ablation).

Cost. 1-3 extra student forwards on the negative captions at 2 states: +25-75% per update
(forward only for the EMA reward terms, forward+backward for the trained ones). Restrict to
d = 1 to halve it.

Expected effect and risk. SoftREPA's SD3 gains are the closest backbone match we have
(MMDiT, rectified flow): GenEval attribute/position gains of +0.07-0.13 at the model level,
which on our student would be worth a few GenEval2 points. Risk: counting regression (use
AGSM's bounded form and monitor the numeracy category from update 500), and in-batch COCO
negatives are easy; our B1fshuf null (other captions' negatives) was in DINO-head space, so it
does not predict this.

### P5. Signed advantage weighting over the four cached candidates with a compositional judge (AMD / NFT / Diffusion-Sharpening), replacing softmax Boltzmann weighting

Mechanism. REST [2608.09226] attaches a few-step student to an RL teacher and distils
segment-wise from the teacher's reward-scored rollouts with Advantage-Modulated Distillation:
the per-trajectory distillation loss is multiplied by `lambda * (A + b)`, A the clipped
group-normalised advantage, b = 0.5, so the worst trajectories get a negative (repulsive) weight;
an EMA-student KL (coefficient 0.2) stabilises it. 8-step CFG-free SD3.5-M reaches GenEval 0.94
(teacher 0.95 at 40 steps), OCR 0.96, PickScore 23.96, at <25% extra cost over the RL run;
uniform imitation is worse. Diffusion-Sharpening [2502.12146] samples 3 branches over 3 steps,
scores the sub-trajectories with IterComp's compositional reward, and trains best-vs-worst with
a reward-modulated DPO loss: SDXL colour 0.637 -> 0.684, texture +0.076, complex +0.041,
whereas SFT on the best branch only gives colour +0.021. LAIR [2605.26491] (offline, implicit
reward): using all candidates with cardinal reward gaps beats top-1 and rank-only weighting
(ImageReward 0.81 vs 0.78 vs 0.58 on SD1.5), GenEval SD1.5 42.4 -> 51.4 (colour 74.5 -> 84.3,
counting 36.6 -> 45.6, attribution 5.3 -> 13.3).

Why it addresses (d) and our Boltzmann nulls. Our `boltzmann`/`boltzmann_sample` arms used
positive softmax weights of DINO scores; every positive result above uses (i) a
prompt-reading judge and (ii) a negative term for the worst candidate. Both were absent.

Implementation in our terms. `--selector amd --score_field endpoint_vqa`: per caption, weight
the CD loss of the argmax candidate by `(A_max + b)` and of the argmin candidate by
`(A_min + b)` (negative when A_min < -b), A = clipped group-normalised VQAScore (or the P2
surrogate to keep the "no text model at train time" story), b = 0.5, plus an EMA-student
regulariser `beta * ||x0_hat - x0_hat_ema||^2`, beta = 0.2 of the CD scale.

Cost. Two student passes per caption instead of one (~1.9x per update, the `boltzmann` arm
measured 2.3x for four).

Expected effect and risk. Sharpening's +0.02-0.05 per category on SDXL and LAIR's +9 GenEval
on SD1.5 bound the optimistic case; on SD3.5-M expect +0.005-0.015 CompBench. Risk:
negative-weighted regression diverges (REST needed b = 0.5 and the EMA-KL; NFT bounds it by an
implicit negative policy) and it uses the VQAScore cache field, which changes the paper's
"scorer needs no text model" framing unless the P2 surrogate is used.

### P6. Object-localised, colour-aware selection score instead of mean-pooled DINO cosine

Mechanism. Three independent results say the information is in DINO's patches but not in the
pooled cosine. (i) Canonical colour [2609.09124]: a linear probe on DINOv2-B patch-mean
features decodes a 10-way object colour at 77.0% from RGB (48.6% grayscale), peaking in middle
layers and degrading in the last ones; CLIP/SigLIP are no better. (ii) iREPA [2512.10794]: the
mean of patch tokens carries a large global component that suppresses spatial contrast;
subtracting gamma * mean and normalising restores it and improves REPA across 27 encoders.
(iii) GORS-unbiased in T2I-CompBench++ [2307.06350]: a Grounded-SAM reward (IoU between the
attribute mask and the noun mask, plus grounding confidence) selected fine-tuning samples as
well as BLIP-VQA did (colour 0.6414 vs 0.6603, spatial 0.1725 vs 0.1815 on SD2) - a
detector/segmenter with the caption's nouns is a valid attribute-binding scorer. CoMat's
attribute-concentration term and BiDPO's region guidance use the same masks.

Why it addresses (a) and (d). The candidate score becomes "for each noun in the caption, does
the object's patch region carry the caption's colour/texture word (linear probe) and does it
exist (detector)", which is colour- and count-sensitive by construction; raw DINO-to-photo
stays as a tie-breaker so the selector still needs no VLM.

Implementation in our terms. Offline in `data/build_candidates.py`: Grounded-DINO + SAM2 on
the four decoded candidates (or the teacher's cross-attention maps for the noun tokens, free
from the rollout), DINOv2 patch features masked per object, a 10-colour linear probe trained on
COCO panoptic masks (hours), score = mean over nouns of [detector confidence x probe
probability of the caption's colour] + spatially-normalised patch cosine to the photo (iREPA
normalisation). Acceptance test before any training: agreement with the VQAScore oracle on the
3k cache above 0.36 (raw DINO) and mean VQAScore of the pick above 0.882.

Cost. Offline only: ~0.3 s per image for Grounded-SAM on 470k images (~40 GPU-h), or ~0 with
attention masks; training cost unchanged.

Expected effect and risk. Selection contributes +0.010-0.014 today with a colour-blind score;
a colour-aware score at most doubles the offline headroom captured (project record: VQAScore
selection captured +0.0585 of a +0.1407 oracle). Risk: detector failure on SD-style
renderings; the CompBench 2D-spatial evaluator's exact-string noun gate (project record) caps
what any selector can do for spatial.

### P7. Dense re-captioning of the COCO photos so caption attributes are grounded in the reference image

Mechanism. ELLA [2403.05135] attributes its colour/texture gains (SDXL colour 0.637 -> 0.726,
texture 0.564 -> 0.669) to MLLM captions that are "highly sensitive to colour and texture"
(CogVLM captions: 8.1 adjectives per caption vs 0.7 in LAION alt-text) but "unreliable for
shape and spatial relationships". SPRIGHT [2404.01197]: COCO captions contain left/right in
0.16%/0.47% of captions; LLaVA spatial re-captions raise every spatial phrase to 20-60%;
fine-tuning SD2.1 on 444 re-captioned images with >18 objects raises T2I-CompBench spatial
0.151 -> 0.213 (+41%) and colour 0.507 -> 0.625, with a 50:50 mix of spatial and original
captions optimal (100% spatial captions hurt). PromptEcho trains only on VLM-generated dense
captions and transfers to GenEval.

Why it addresses (a). Our selector compares candidates to the photo; if the caption names the
colours and layout the photo actually has, "closest to the photo" and "compositionally
correct" coincide far more often, and the teacher's candidates are conditioned on those words.
It also raises the density of colour/count/spatial words the student ever sees during
distillation (COCO captions average 11 tokens).

Implementation in our terms. `data/build_pool.py --recaption`: LLaVA-1.5-13B or Qwen2.5-VL-7B
with the SPRIGHT prompt (spatial + relative sizes) merged with a dense description; keep the
original caption with p = 0.5; SD3.5's T5 stream takes 256 tokens, the CLIP streams truncate at
77 (SPRIGHT shows longer captions help SD2.1 even with truncation). Rebuild the 3k cache first
(4 x 3k rollouts).

Cost. ~1 s/image for captioning (3k: minutes; 118k: ~33 GPU-h) plus the cache rebuild, which
is the dominant cost of any arm anyway.

Expected effect and risk. SPRIGHT's spatial +0.06 on SD2.1 and ELLA's colour gains are
base-model fine-tuning effects; for a distillation student the effect is indirect and unproven.
Risk: distribution shift between dense training captions and the short CompBench/GenEval2
prompts (PromptEcho reports transfer; ELLA reports it for CompBench).

### P8. Verifiable detector reward on the 4-step student itself, restricted to spatial and numeracy prompts

Mechanism. Flow-GRPO [2505.05470] on SD3.5-M with GenEval rule rewards (Mask2Former detections
+ colour classifier, partial credit for count/position/colour), KL beta = 0.04, group 24,
10-step training rollouts: GenEval 0.63 -> 0.95 and, trained only on GenEval-style prompts,
T2I-CompBench++ colour 0.799 -> 0.838, shape 0.567 -> 0.613, texture 0.734 -> 0.724, 2D-spatial
0.285 -> 0.545, 3D-spatial 0.374 -> 0.447, numeracy 0.593 -> 0.675, non-spatial +0.005; without
the KL the DrawBench quality metrics collapse. For few-step students: TDM-R1 [2603.07700]
(surrogate + marginal KL, 4 NFE, GenEval 0.92), RTDMD [2605.26108] (reward-tilted DMD: GRPO on
the stochastic intermediate steps with shared noise outside a step subset, plus direct reward
backprop through the deterministic last step; 4-step SD3.5-M GenEval 0.94; note "SubGRPO"
shared-noise groups halve the variance), Flow-Map GRPO [2607.00535] (anchored stochastic
composition for consistency/MeanFlow generators), REST [2608.09226] (8-step 0.94). Regularisers
that kept these stable: KL to the base at 0.01-0.04 (Flow-GRPO), marginal KL to the teacher via
the fake score (TDM-R1/RTDMD), gated KL on the ~10% highest-uncertainty samples + periodic
reference reset + DINOv3-diversity advantage shaping (GARDO [2512.24138], GenEval 0.95 at 400
steps without unseen-reward loss), win-rate rewards from a pairwise VLM judge instead of
pointwise scores (Pref-GRPO [2508.20751]: pointwise scores cluster within a group, the
normalised advantage amplifies noise, "illusory advantage"), and removing the std
normalisation (GARDO).

Why it addresses (b) and the headroom. Spatial negatives are not contradictions because the
student cannot draw the relation; only a reward that pays for drawing it changes that. Our
student's 2D-spatial 0.21-0.24 vs 0.545 and numeracy 0.55 vs 0.675 under RL are the two
largest gaps in the benchmark.

Implementation in our terms. A post-distillation stage on the 4-step student: prompts from
the CompBench spatial/numeracy train splits plus GenEval templates, groups of 8 (cheap: 4 NFE),
UniDet/Mask2Former reward with the CompBench spatial rule, P2's surrogate for the gradient
(TDM-R1) or RTDMD's hybrid estimator, marginal KL to the 8-step teacher through our existing CD
loss as the anchor (replace their fake score with the teacher trajectory distance), CMMD/FID
and DINO-patch diversity monitored every 200 updates.

Cost. 8 rollouts x 4 NFE + 8 detector calls (~0.1 s each) per prompt: ~10x our current update,
i.e. a separate 1-2 GPU-day stage, not an add-on to distillation.

Expected effect and risk. The largest available gain (+0.05-0.10 CompBench overall if the
Flow-GRPO transfer holds at 4 steps), but it changes the paper's story from "selection" to
"RL", and the CompBench 2D-spatial vocabulary gate (31.8% of spatial prompts score 0 for any
image, project record) caps the measured spatial gain. Texture dropped 0.01 under Flow-GRPO.

### P9. A noise hypernetwork on the frozen student for the compositional reward (no weight change to the generator)

Mechanism. HyperNoise [2508.09968]: a LoRA copy of the distilled generator predicts a noise
delta, trained to maximise a reward with KL approximated by 0.5 * ||delta||^2 (exact in noise
space by a Jacobian bound); the generator stays frozen, so there is no reward hacking of its
weights. GenEval: SD-Turbo 0.49 -> 0.57, SANA-Sprint 0.70 -> 0.75 (counting 0.64 -> 0.71,
position 0.41 -> 0.51, attribution 0.51 -> 0.55), FLUX-schnell 0.68 -> 0.72, recovering about
half of ReNO's test-time gain at 1.3x inference cost. CARINOX [2509.17458] (abstract-level)
adds category-aware rewards for initial-noise optimisation.

Why it fits us. Our selection is literally a choice among four initial noises; a hypernoise
network learns the map from noise to good noise instead of enumerating four, and it is the one
reward channel whose regulariser has a proof of boundedness. Addresses (c).

Implementation. A short stage after distillation: rank-32 LoRA on the student as f_phi,
reward = P1's BLIP-NLL or BLIP-VQA disentangled probability (differentiable), 4-step student
forward with gradient, L2 on the delta; ~5k updates.

Cost. Reward backprop through 4 student steps + decode + scorer per sample: ~3-5 s per batch
of 16, a 1-GPU-day stage; inference +1 LoRA forward.

Expected effect and risk. +0.03-0.05 GenEval on SANA-Sprint-class models; on our student a
few GenEval2 points. Risk: needs a differentiable compositional reward (VQAScore's 11B
CLIP-FlanT5 is too heavy for the backward pass; BLIP-VQA and BLIP captioning are not).

### P10. Adopt the anti-hacking bookkeeping regardless of which channel is chosen

Not a proposal on its own but three cheap additions that the literature agrees on:
(i) uncertainty gate (GARDO): apply the reward gradient only to samples where the proxy and a
second scorer disagree least, or scale by their agreement (we already decode 32 images per
refresh, so the true-vs-proxy gap is available); (ii) win-rate rewards over the 4 candidates
instead of raw cosines wherever a scalar enters an advantage (Pref-GRPO; our projector-cosine
gaps of ~0.001 are the "illusory advantage" regime); (iii) an unseen-reward panel (HPSv3,
ImageReward, CMMD, DINOv3 group diversity) logged every 500 updates, since every hacking report
above was invisible to the optimised reward.

---------------------------------------------------------------------------------------------------

## A. Why DINO/DINOv2 features are insensitive to colour (and to which colour belongs to which object)

**Recipe.** DINO [2104.14294] trains the student to match the teacher's CLS distribution across
views produced with the BYOL augmentation family: random resized crops, horizontal flip, colour
jittering (p = 0.8), random grayscale (p = 0.2), Gaussian blur and solarisation. DINOv2
[2304.07193] keeps the DINO/iBOT augmentations ("similar to DINO"; tabulated in [2401.00463],
Table 1, App. A.2) and adds the iBOT patch loss and KoLeo. The objective therefore rewards
representations that are invariant to hue/saturation shifts and to grayscale conversion of the
whole view. The classical evidence that this discards colour: LooC [2008.05659] shows
colour-invariant contrastive features hurt colour-dependent recognition (Flowers) and fixes it
with an augmentation-specific embedding head; Purushwalkam and Gupta [2007.13916] measure the
invariances MoCo/PIRL acquire from their augmentations. CoMat [2404.03653] also found a DINO
discriminator worse than a UNet one (FID 23.9 vs 16.7) for judging generated latents.

**But colour is not erased; it is low-variance and not bound.** Canonical Color as a Lens
[2609.09124] linearly probes mean-pooled patch features of DINOv2-B, CLIP-B/32, SigLIP-B,
ViT-MAE and Swin-V2 for a 10-way object colour: from RGB inputs DINOv2 reaches 77.0% (S 71.1,
L 79.4, g 80.6), from grayscale 48.6%, and 43.8% after histogram equalisation, against a 14%
majority baseline; CLIP 41.5% and SigLIP 42.4% under the same equalised grayscale, i.e. DINOv2 is not worse than
the language-supervised encoders at exposing colour. Accuracy rises from early to middle layers
and drops in the last layers for DINOv2, SigLIP and especially MAE ("specialisation for the
pretraining objective"). Data or Language Supervision [2510.11835] trains CLIP and DINO on the
same 10M images/architecture and finds DINO *more* responsive to low-level colour schemes and
styles in its embedding geometry (image pairs with high DINO but low CLIP similarity share a
palette), while CLIP separates object identity and text (Cars +20.6, CUB +9.3 linear probe; OCR
+7.5 in a VLM). Cambrian-1 [2406.16860] and MMVP [2401.06209] use DINOv2 distance (< 0.6 cosine)
to find image pairs that CLIP conflates, including "colour and appearance" and "quantity and
count" pairs; scaling CLIP fixes only colour/appearance and state, not orientation, count or
position.

So the global DINOv2 embedding does respond to a scene's palette. What it does not do is encode
*which object carries which colour* in a way a distance can read. How can embedding models bind
concepts? [2605.31503] shows that CLIP and DINOv2 scene embeddings decompose additively over
objects (a two-object image embedding is close to the sum of the single-object embeddings), that
concept presence (red, cube) is linearly recoverable, but that object identity (red cube vs blue
cube) requires a high-complexity, combination-specific map: a multiplicative binding probe that
generalises for from-scratch models leaves object recognition near zero for both CLIP-B/32 and
DINOv2-B/16, even though a trained probe recovers binding uni-modally (CLIP image 0.96). Does
CLIP Bind Concepts? [2212.10537] reports the same failure for CLIP on two-object attribute
scenes. In DINOv2's geometry "objects with different shapes are far apart, whereas objects with
similar colours remain close" ([2605.31503], App. D.5), consistent with a palette signal that is
not object-resolved.

**Why our cosine gaps were r = 0.00 for colour.** Three compounding reasons, each measurable:
(1) mean-pooling over 256 patches averages the two objects' colour evidence into one vector and
removes the binding; (2) iREPA [2512.10794] shows the patch mean carries a large global
component that dominates pairwise structure, and that generation benefits track *spatial
structure* (pairwise patch similarity, Pearson |r| > 0.85 with FID) not global semantics
(|r| = 0.26); spatial normalisation (subtract gamma * mean, gamma in [0.6, 0.8], and divide by
the spatial std) restores contrast and helps every encoder, DINOv2-B included (FID 19.1 -> 17.0
at 100k); (3) a "red -> blue" edit changes a few tens of patches by a direction that a 10-way
linear probe can read but that is a small fraction of the 768-d variance dominated by shape,
layout and category; the VQA ceiling shows the same edits carry large VQAScore gaps (+0.47 to
+0.75). Our measurement that DINO's alignment signal is between-prompt (pooled r = +0.394 vs
VQAScore) and collapses within-prompt (+0.062) is the same statement.

**Layer and pooling.** Canonical-colour probes peak in middle layers (roughly layers 6-9 of 12
for DINOv2-B) and decline in the final two; Analyzing Local Representations [2401.00463] shows
DINOv2's last two layers re-specialise (k-NN patch classification of hard categories jumps at
layers 11-12 while easy/low-level categories are already saturated by layer 8), and Deep ViT
Features [2112.05814] shows early-layer keys carry position and low-level appearance, deeper
layers semantics. Practical consequence: if colour must be read from DINO, read it from
mid-layer patch tokens restricted to the object's mask with a linear probe (P6), or spatially
normalise before pooling; do not expect it from a CLS/final-layer cosine. A larger DINO is not
the answer: colour decodability rises only 46 -> 52% from S to g [2609.09124], and iREPA finds
larger DINOv2 no better as an alignment target.

---------------------------------------------------------------------------------------------------

## B. Ranking and listwise losses for image-text reward or selection models, and the "edited-prompt-ness" shortcut

**How T2I reward models are trained.** ImageReward [2304.05977]: annotators rank k in [4, 9]
images per prompt; every ordered pair (up to C(k,2)) enters a pairwise Bradley-Terry loss
`-log sigma(f(x_i) - f(x_j))` on a BLIP backbone with cross-attention fusion and an MLP head;
137k comparisons; 70% of transformer layers frozen to stop rapid overfitting; preference
accuracy 65.1% vs CLIP 54.8, Aesthetic 57.4, BLIP 57.8. PickScore [2305.01569] fits CLIP-H with
an in-batch preference objective on Pick-a-Pic; HPSv2 [2306.09341] fine-tunes CLIP on HPDv2 with
a KL/pairwise form that HPSv3 shows is equivalent to BT. HPSv3 [2508.03789]: Qwen2-VL-7B
backbone, an uncertainty-aware ranking loss where each score is a Gaussian N(mu, sigma) and the
preference probability integrates the sigmoid over both, +0.1 to +2.2 points over plain RankNet
on four test sets (2.9% relative on PickScore; 85.4% HPDv2, 76.9% HPDv3); 1.17M pairs with 9-19 annotators each and only
> 95%-agreement pairs used. IterComp [2410.07171]: three composition-aware BT reward models
(attribute binding, spatial, non-spatial) on BLIP, trained on 52.5k human rankings of a
six-model gallery (15 pairs per prompt from 6 images), then ReFL-style feedback learning with
iterative expansion of the ranked lists with the tuned model's own samples (ranks inserted by
the current reward model). Multi-dimensional variants: MPS [2405.14705], VisionReward
[2412.21059], RAHF [2312.10240]; VLM judges: UnifiedReward [2503.05236] (pairwise and pointwise),
HPSv3++ [2606.14657]. Listwise objectives: RankDPO [2410.18013] (ranking-based DPO on synthetic
ranked lists; SD3 GenEval 0.68 -> 0.74 per SoftREPA's table), Diffusion-LPO [2510.01540]
(Plackett-Luce over ranked image lists), LAIR [2605.26491] (softmax-centred advantage weights
over all candidates on the implicit reward with a quadratic penalty; closed-form bounded optimum
`s_i* = N_c w_i / (2 lambda)`; cardinal > ordinal > top-1), AGSM [2605.30038] (PL over in-batch
captions beats pairwise BT on every alignment metric). Surrogates trained by ranking: LaSRO
[2411.15247] (BT in latent space from win/lose pairs of the target reward; W/L = max/min of
N_s = 6 samples so the gap is maximal), RG-LCD [2403.11027] (KL between the latent proxy's and
the expert's softmax preferences over three latents, with a temperature so that hard binary
experts are admissible, and the real image latent as a privileged positive), TDM-R1
[2603.07700] (group preference optimisation of a diffusion-parameterised surrogate with an EMA
reference), Pref-GRPO [2508.20751] (win-rate over all pairs of a group from a pairwise VLM
judge as the reward).

**What stopped shortcut learning.** SugarCrepe [2306.14610] is the key negative result for our
negatives: on ARO, CREPE and VL-Checklist the rule-generated hard negatives (word swaps,
replacements, shuffles, negations) are "not plausible" and "non-fluent", so a *text-only*
plausibility model (Vera) or a grammar model beats every CLIP on 9 of 10 tasks without seeing
the image; NegCLIP's improvements were "hugely overestimated" (near-human 94% on matched
REPLACE negatives collapses on SugarCrepe). The fix has two parts: LLM-written fluent negatives
(commonsense score 37 -> 50, grammar 77 -> 89) and adversarial refinement that subsamples pairs
until the text-only score-gap distributions are symmetric around zero, after which the blind
models rank last. SugarCrepe++ [2406.11171] adds lexical-vs-semantic controls. On the training
side, FSC-CLIP [2410.05210] shows the global hard-negative loss (NegCLIP, DAC, CE-CLIP)
damages zero-shot and retrieval because the negative caption embeds almost on top of the
positive; a *local* token-patch hard-negative loss, focal weighting (gamma = 2) so confident
pairs contribute little, and label smoothing (beta = 0.02) that grants the negative a small
positive margin keep compositionality (Comp 53.5 vs NegCLIP 54.1) while retaining ZS (55.9 vs
55.9 pre-trained) and I2T retrieval (58.2 vs 53.8). TripletCLIP [2411.02545] and CounterCurate
[2402.13254] generate negatives on the image side too (synthetic/edited images), and SPEC
[2312.00081] synthesises minimal-change images for size/position/count; VisMin [2407.16772]
provides human-verified minimal-change pairs. BiDPO [2605.28615] is the generative analogue:
text-only DPO on edited captions collapses (CompBench 13.5), and only when each edited caption is
paired with an edited image realising it does the objective work (54.4).

**Reading our campaign through this.** The head's held-out ordering (0.58-0.60 shaped vs
0.50-0.53 against a shuffled photo) and the "edited-prompt-ness" mechanism are the SugarCrepe
artefact: rule-based v2/v3 negatives are separable from their originals by a text-only model, so
a head with a text side learns that rather than the image content. Arm A is the unbounded push
that both SoftREPA's authors [2605.30038] and BiDPO [2605.28615] report as destructive when the
negative side has no bounded reference. The remedies, in order of cost: (1) audit the negatives
with a text-only classifier (DistilBERT-CoLA / Vera as in SugarCrepe) and keep only pairs the
blind model cannot separate, or rewrite them with an LLM under fluency/plausibility constraints;
(2) demand an image-side realisation and a VQA contradiction for every kept pair (the VQA
ceiling's "both" set, 66.5%); (3) if a head is kept, make it local (patch-token similarity) with
focal weighting and label smoothing rather than a global cosine ranking; (4) put the comparison
on a bounded implicit reward (DPO/DSPO/LAIR/AGSM) rather than a free ranking margin; (5) prefer
listwise PL with cardinal VQAScore gaps over the four cached candidates plus negatives (LAIR,
AGSM) to pairwise margins; (6) train the scorer on the inputs it will score (one-step estimates
at sigma 0.34-0.55), the LaSRO/LPO lesson our reward_state_probe re-discovered.

---------------------------------------------------------------------------------------------------

## C. Per-category evidence on colour binding, counting and spatial relations

T2I-CompBench (BLIP-VQA for attributes, UniDet for spatial/numeracy) unless marked GenEval
(detector-based). Deltas are absolute score differences from the paper's own baseline.

| Intervention (base) | colour | shape | texture | 2D-spatial | numeracy / count | other | source |
|---|---|---|---|---|---|---|---|
| Attend-and-Excite, training-free attention (SD2) | 0.507 -> 0.640 (+0.134) | +0.030 | +0.104 | +0.011 | +0.019 | | [2307.06350] |
| GORS, BLIP-VQA/UniDet-selected reward-weighted SFT (SD2) | +0.154 | +0.056 | +0.137 | +0.047 | +0.025 | 3D +0.034 | [2307.06350] |
| GORS-unbiased, Grounded-SAM/GLIP selection (SD2) | +0.135 | +0.033 | +0.110 | +0.038 | +0.027 | | [2307.06350] |
| ELLA, LLM text encoder + MLLM captions (SDXL) | 0.637 -> 0.726 (+0.089) | +0.023 | +0.105 | +0.018 | | non-spatial -0.004 | [2403.05135] |
| CoMat, BLIP captioning NLL reward (SDXL) | 0.588 -> 0.783 (+0.195); CA alone +0.158 | +0.064 | +0.117 | +0.030 | | complex +0.044 | [2404.03653] |
| IterComp, composition BT rewards + iterative ReFL (SDXL) | 0.637 -> 0.798 (+0.161) | +0.081 | +0.205 | +0.116 | | complex +0.078 | [2410.07171] |
| Diffusion-Sharpening RLHF, best/worst trajectory DPO (SDXL) | +0.047 | +0.027 | +0.076 | +0.010 | | complex +0.041; SFT variant colour +0.021 | [2502.12146] |
| LPO, diffusion-native latent reward + step-level DPO (SDXL) | 0.583 -> 0.735 (+0.152) | +0.068 | +0.140 | +0.048 | +0.062 | 3D +0.076, complex +0.047 | [2502.01051] |
| BiDPO, bimodal DPO on edited pairs + region mask (SDXL) | 0.589 -> 0.794 (+0.205) | +0.136 | +0.182 | +0.022 | +0.093 | ImageDPO alone: colour +0.085, numeracy -0.107; TextDPO alone: 0.135 total | [2605.28615] |
| SPRIGHT, 444 spatial re-captioned images >18 objects (SD2.1) | 0.507 -> 0.625 (+0.119) | +0.043 | +0.100 | 0.151 -> 0.213 (+0.063) | GenEval count 0.44 -> 0.49 | | [2404.01197] |
| Flow-GRPO, GenEval detector reward + KL 0.04 (SD3.5-M, 40 NFE) | 0.799 -> 0.838 (+0.039) | +0.046 | -0.010 | 0.285 -> 0.545 (+0.260) | 0.593 -> 0.675 (+0.083) | 3D +0.073, non-spatial +0.005; GenEval colours 0.81 -> 0.92, counting 0.50 -> 0.95, position 0.24 -> 0.99, attr 0.52 -> 0.86 | [2505.05470] |
| TDM-R1, surrogate + marginal KL (SD3.5-M, 4 NFE) | GenEval colours 0.79 -> 0.85 | | | GenEval position 0.23 -> 0.93 | GenEval counting 0.49 -> 0.88 | attr 0.44 -> 0.91; overall 0.61 -> 0.92 | [2603.07700] |
| RTDMD, reward-tilted DMD (SD3.5-M, 4 NFE) | | | | | | GenEval 0.94 with non-differentiable GenEval reward | [2605.26108] |
| REST, advantage-modulated distillation from RL rollouts (SD3.5-M, 8 NFE, CFG-free) | | | | | | GenEval 0.94 (teacher 0.95); PickScore 23.96 | [2608.09226] |
| SoftREPA, soft tokens + contrastive denoising (SD3) | GenEval colours 0.85 -> 0.92 | | | GenEval position 0.27 -> 0.34 | GenEval counting 0.56 -> 0.29 | attr 0.55 -> 0.68, two-obj 0.86 -> 0.95 | [2503.08250] |
| AGSM, bounded PL alignment guidance (SD1.5/SDXL/SD3) | matches SoftREPA | | | | counting +35% relative vs SoftREPA | | [2605.30038] |
| LAIR, listwise reward-aware implicit reward (SD1.5 / SDXL) | GenEval 74.5 -> 84.3 / 88.3 -> 91.0 | | | 3.5 -> 6.3 / 11.0 -> 13.5 | 36.6 -> 45.6 / 42.8 -> 39.7 | attr 5.3 -> 13.3 / 21.0 -> 28.3 | [2605.26491] |
| PromptEcho, VLM prompt cross-entropy reward (Z-Image) | GenEval colours 0.86 -> 0.86 | | | position 0.41 -> 0.52 | counting 0.75 -> 0.85 | attr 0.59 -> 0.73 | [2604.12652] |
| HyperNoise, learned initial noise on frozen distilled model (SANA-Sprint) | GenEval colours 0.86 -> 0.85 | | | position 0.41 -> 0.51 | counting 0.64 -> 0.71 | attr 0.51 -> 0.55 | [2508.09968] |
| DreamSync, VQA-filtered best-of-8 self-training (SDXL) | human DSG colour 0.82 -> 0.84 | | | spatial 0.73 -> 0.78 | count 0.72 -> 0.77 | TIFA +1.7, DSG +2.9 | [2311.17946] |
| Ours, DINO selection + refreshed projector reward vs naive CD (SD3.5-M student, 4 NFE, converged 3k) | +0.003 | +0.005 | +0.006 | -0.012 (noise) | +0.008 | complex +0.0075 (t p 0.03); total +0.0024 | `per_prompt_analysis.md` |

Reading. (1) Every colour gain above +0.05 comes from a signal that reads the prompt: a VQA/BT
reward, a captioning likelihood, an LLM text encoder with attribute-dense captions, or
attribute-word masks; the two training-free attention methods and the segmentation-based
GORS-unbiased reward show it is object-attribute *localisation*, not a better global embedding,
that moves colour. (2) The gains scale with base-model headroom: SD2/SDXL sit at 0.51-0.64
colour and move +0.09 to +0.21; SD3.5-M sits at 0.80 and moves +0.04 under the strongest reward
(detector RL). Our student is at 0.807-0.817 colour, i.e. already at the SD3.5-M base; the
teacher at 28 steps scores 0.505 overall, so colour is not where a distillation student can gain
more than ~0.03. (3) Counting and 2D-spatial are where SD3.5-M moves most (numeracy +0.08,
spatial +0.26 under Flow-GRPO; +0.10 counting and +0.11 position for PromptEcho on Z-Image) and
where our student is furthest below the base (2D-spatial 0.21-0.24 vs 0.285; numeracy 0.55 vs
0.59). Both require the generator to learn to *draw* the relation, which is why offline
selection and cosine rewards cannot reach them (our (b)); in the literature only RL/DPO with a
detector or VLM judge, or dense spatial captions, moved them. (4) Two documented regressions to
guard against: SoftREPA's counting collapse (over-emphasised text duplicates objects) and
Flow-GRPO's texture -0.01; ImageDPO-only numeracy -0.11 in BiDPO.

---------------------------------------------------------------------------------------------------

## D. Answers to the five questions

**Q1. Which scorers are sensitive to colour/count/spatial, and which are cheap enough?**
Sensitivity (human-correlation evidence, T2I-CompBench++ [2307.06350], Kendall tau): CLIPScore
0.19 colour / 0.27 spatial / 0.16 numeracy; disentangled BLIP-VQA 0.63 colour, 0.52 texture;
UniDet 0.48 spatial, 0.31 3D, 0.43 numeracy; GPT-4V 0.52 colour, 0.35 spatial, 0.48 non-spatial
and 0.51 complex (best there). VQAScore [2404.01291] (CLIP-FlanT5-XXL, P("Yes" | image, "Does
this figure show <text>?")) is the strongest single alignment score on Winoground/GenAI-Bench and
is the oracle our cache stores; DSG [2310.18235] and TIFA decompose the prompt into atomic
questions (GenEval2 [2512.16853] uses the same soft-TIFA idea with Qwen3-VL); GenEval
[2310.11513] uses Mask2Former detection + a CLIP colour classifier and is the reward behind the
+0.3 GenEval results. Preference models (ImageReward, PickScore, HPSv2/v3, UnifiedReward) are
trained on holistic preferences and cluster within a prompt (Pref-GRPO), though LPO shows that
VQAScore-filtered preference pairs still teach colour. Costs measured in the cited papers:
HPSv3 (Qwen2-VL-7B) 89 ms per 512-px image, DiT-Reward on the latent 54 ms, PickScore/CLIP
~10 ms, BLIP-base captioning/VQA tens of ms, Mask2Former ~0.1 s, VQAScore-XXL ~0.5-1 s (11B,
forward only), Qwen3-VL-8B/32B prompt-CE one forward pass (PromptEcho; 32B on 32 H20s for 100 h
at group 8). Distillable into a proxy: yes, and that is the LaSRO/RG-LCD/TDM-R1/LPO/DiT-Reward
line (P2); RG-LCD's proxy even learns from non-differentiable judges. DINO/CLIP colour
invariances: Section A.

**Q2. Rewards on few-step models without hacking.** Diffusion-DPO [2311.12908] and D3PO
[2311.13231] were derived for many-step models; LaSRO [2411.15247] documents why policy-gradient
and RWR/DPO objectives fail at <= 2 steps (non-smooth map, no stochasticity, denoising-loss
surrogates blur) and why a latent surrogate with off-policy noise exploration works; RLCM
[2404.03673] (DDPO on consistency models) is beaten by it. RG-LCD [2403.11027] adds the reward
to the LCD loss on the one-step estimate with beta 1-5 and needs the latent proxy to avoid
ImageReward's high-frequency noise. SDPO [2411.11727] (dense reward differences for few-step
models), Hyper-SD [2404.13686] (human-feedback stage after trajectory-segmented CD), DMD2
[2405.14867] (GAN term), and the 2026 wave that reaches GenEval 0.92-0.95 at 4-8 NFE on
SD3.5-M: TDM-R1 [2603.07700] (surrogate + marginal reverse KL to the teacher, beta_g set so
reward:KL gradient = 2:1), RTDMD [2605.26108] (KL to a reward-tilted teacher = DMD + reward;
SubGRPO shared-noise step subsets; direct backprop through the deterministic last step adds
+0.1 PickScore), Flow-Map GRPO [2607.00535] (for consistency/flow-map generators), REST
[2608.09226] (advantage-modulated distillation with EMA-KL 0.2), DiffusionOPSD [2608.24646]
(reward gradients converted into bounded clean-output targets at low-noise queries, fitted as
detached supervision with an EMA behaviour policy: best of 20 reward-matched settings on SD3.5-M
and step-distilled Z-Image-Turbo (19 of 20), 40-63% fewer GPU-hours than DiffusionNFT; query noise 0.90
hurts, radii 0.08-0.40 fine, which brackets our reward states at sigma 0.34-0.55), AdvDMD
[2604.28126], Reward-aware trajectory shaping [2604.14910], DMDR [2511.13649], GDMD [2604.19009],
ReDiF [2512.22802], DGPO [2510.08425] (GenEval 0.97 on the full model). Regularisers that kept
them stable: KL to the base policy (Flow-GRPO beta 0.04 for GenEval/OCR, 0.01 for PickScore;
without it Aesthetic 5.39 -> 4.93 and ImageReward 0.87 -> 0.44 while GenEval still reads 0.95),
marginal/distribution-level KL through a fake score or the teacher (TDM-R1, RTDMD), the
original distillation loss as a regulariser (LaSRO c = 500, RG-LCD), an EMA copy of the student
(REST, DiffusionOPSD), a discriminator against the original model's latents (CoMat), gated KL on
uncertain samples + reference reset + diversity shaping (GARDO), win-rate rewards (Pref-GRPO),
and surrogate-trajectory design for reward backprop (FlowBP [2606.11075]: evaluate the reward on
the actual rollout endpoint rather than a one-step Tweedie estimate (+0.12 PickScore on
ReFL/DRTune), keep at most one Jacobian factor, compact active sets; FLUX.1-dev GenEval 63.3 ->
69.9 with HPSv2.1 as the only reward). Reward share: TDM-R1 2:1 reward:KL, RG-LCD beta 1-5,
RTDMD beta = 1, ReFL lambda = 1e-3 with the pretraining loss, ours 20% of the CD gradient.

**Q3. Hard-negative and preference-pair construction.** Text side: NegCLIP/ARO [2210.01936],
Winoground [2204.03162], SugarCrepe(+) [2306.14610, 2406.11171], FSC-CLIP [2410.05210],
TripletCLIP [2411.02545], CounterCurate [2402.13254], SPEC [2312.00081], VisMin [2407.16772]
(Section B: rule negatives are text-only hackable; LLM + adversarial filtering; local losses).
Image side for generation: BiDPO [2605.28615] (edited images + edited captions, VQA-filtered,
region masks; the only one with per-category CompBench evidence), D-Fusion [2505.22002]
(same-noise base image + self-attention fusion from a high-preference reference under a
cross-attention mask, so the pair differs only in the prompt-related region; improves DPO/DDPO/
DPOK on attribute and positional templates by CLIPScore), Di3PO [2602.06355] (diptych prompting
gives pairs with identical backgrounds; beats background-varying DPO), Diffusion-Sharpening
[2502.12146] (best/worst sub-trajectories among 3 branches), IterComp [2410.07171] (multi-model
gallery rankings + iterative self-insertion), DreamSync [2311.17946] (VQA >= 0.9 + aesthetic
>= 0.6 filter over 8 samples, 25-30% of prompts pass; +1.7 TIFA, +2.9 DSG on SDXL), GORS
[2307.06350]. The signals that transferred to CompBench/GenEval are VQA- or detector-judged
pairs with minimal visual difference; text-only negatives never did.

**Q4. Representation alignment.** REPA [2410.06940] (DINOv2 target, layer 8, cosine), REPA-E
[2504.10483] (end-to-end VAE), VA-VAE [2501.01423], U-REPA [2503.18414], REG [2507.01467], DDT
[2504.05741], SRA [2505.02831] (self-alignment, no external encoder), Diffuse-and-Disperse
[2506.09027], RAE [2510.11690]: all report FID/IS/convergence on ImageNet or COCO-FID, none
reports T2I-CompBench or GenEval. What matters in the target: spatial structure not global
semantics (iREPA [2512.10794]: SAM2 with 24% ImageNet accuracy beats PE-Core-G with 82.8%; SIFT
and HOG work; conv projector + spatial normalisation improve every encoder including CLIP-L and
DINOv2/3 on ImageNet and on MMDiT text-to-image FID). When: HASTE [2505.16792] measures the
cosine between the REPA and denoising gradients: positive early, orthogonal by ~400k
iterations, negative later and negative from the start at t <= 0.1 (low noise), so alignment
should be applied at mid noise and terminated (tau = 250k of 4M); this matches our null at
sigma 0.009 (B2) and harm at sigma 0.86 (B2m) with only 3k updates of a converged student.
Text-conditioned / VLM targets: the only REPA-family method with GenEval gains is SoftREPA
[2503.08250], which aligns text and image *inside* the diffusion model through the conditional
denoising loss with in-batch caption negatives (SD3 GenEval 0.68 -> 0.70 with attribute
+0.13, but counting -0.27), refined by AGSM [2605.30038]; VoT [2609.07815] (abstract-level)
proposes a unified multimodal representation alignment for T2I. There is no published evidence
that a DINO-target alignment loss moves compositional metrics, in either direction.

**Q5. Selecting/curating teacher trajectories.** GORS [2307.06350] (reward-thresholded,
reward-weighted SFT on self-generated samples; SD2 colour +0.15, robust to the reward model
used), DreamSync [2311.17946] (VQA + aesthetic best-of-8, diminishing after 3 iterations),
Emu [2309.15807] (quality tuning on a few thousand curated images), Diffusion-Sharpening
[2502.12146] (path-integral selection of the best sub-trajectory during fine-tuning; n = 3
branches x m = 3 steps optimal; DPO best-vs-worst >> SFT-on-best), REST [2608.09226] (the
teacher's reward-scored rollouts *are* the distillation data; signed advantage weights),
Inference-time scaling [2501.09732] and ReNO/HyperNoise [2508.09968] (search or learn the
initial noise under a verifier; best-of-N with a compositional verifier gives SANA-Sprint
GenEval 0.70 -> 0.79 and the amortised version 0.75), CARINOX [2509.17458] (category-aware
verifier for noise search). Effect sizes for pure selection are modest and judge-dependent:
SFT-on-best +0.02 colour (Sharpening), DreamSync +1.7 TIFA; the same pairs used contrastively
(best vs worst) give 2-4x more. That is consistent with our factorial (argmax +0.010-0.014,
reward +0.005) and argues for P5 over more candidates.

---------------------------------------------------------------------------------------------------

## E. Do-not-bother list

* DINO-target representation alignment (REPA-style) at any layer/sigma for compositional
  metrics: no CompBench/GenEval evidence exists; HASTE's gradient-conflict analysis predicts
  harm at low noise on a trained model; our B2/B2m reproduce it.
* Text-ranking heads on DINO-anchored features with rule-based negatives: SugarCrepe hackability
  plus FSC-CLIP's global-loss damage; our head learned edited-prompt-ness; three rounds of nulls.
* Positive-only softmax (Boltzmann) weighting over candidates with a colour-blind score: our
  three nulls; every positive weighting result in the literature has a prompt-reading judge and a
  negative term (REST, Sharpening, LAIR, NFT).
* Pointwise cosine gaps of ~0.001 as a training signal: the Pref-GRPO "illusory advantage"
  regime; normalising them amplifies noise (our 7-8x Jacobian gain at the switch).
* Spatial / 3D-spatial structured negatives for any ranking or DPO objective: not contradictions
  to the judge (0.54 / 0.68 contradiction rate), the student cannot draw them, and the CompBench
  2D evaluator's noun gate zeros 31.8% of prompts regardless.
* Pure best-of-N self-training (SFT on the student's own VQA-filtered samples) at 3k captions:
  +0.02 colour class of effect, 20-30% of prompts pass the filter, and our student already sits
  at the SD3.5-M colour level.
* CLIPScore as reward or selector: tau 0.19 for colour; over-optimising it writes the prompt into
  the image (RG-LCD).
* Larger or newer DINO (DINOv2-L/g, DINOv3) as the selector: colour decodability 46 -> 52%,
  iREPA finds no target-quality gain from size, and the binding failure is encoder-family-wide.
* More candidates with the same scorer (8 vs 4 was +0.001 here); the scorer, not N, binds.
* Multi-reward sums as the training signal (GARDO: slower and conflicting) unless one term is an
  explicit anchor (CD loss, KL) and the others are gated.
* Reward-hacking detection by the optimised reward alone: every hacking report above needed an
  unseen-reward panel or a human study.

---------------------------------------------------------------------------------------------------

## F. References (arXiv ids resolved and titles matched on 2026-09-17)

Scorers, judges, reward models
* 2404.01291 Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)
* 2310.18235 Davidsonian Scene Graph (DSG)
* 2310.11513 GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment
* 2512.16853 GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation
* 2307.06350 T2I-CompBench / T2I-CompBench++ (v3 incl. GORS, GORS-unbiased, metric-human correlations)
* 2304.05977 ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation
* 2305.01569 Pick-a-Pic (PickScore)
* 2306.09341 Human Preference Score v2
* 2508.03789 HPSv3: Towards Wide-Spectrum Human Preference Score
* 2606.14657 HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities
* 2503.05236 Unified Reward Model for Multimodal Understanding and Generation
* 2405.14705 Learning Multi-dimensional Human Preference for Text-to-Image Generation (MPS)
* 2412.21059 VisionReward
* 2312.10240 Rich Human Feedback for Text-to-Image Generation (RAHF)
* 2606.23626 DiT-Reward: Generative Representations for Text-to-Image Reward Modeling
* 2502.01051 Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization (LRM/LPO)
* 2604.12652 PromptEcho: Annotation-Free Reward from Vision-Language Models for Text-to-Image RL
* 2607.11886 Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation
* 2506.02095 Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
* 2603.22228 SpatialReward: Verifiable Spatial Reward Modeling for Fine-Grained Spatial Consistency
* 2602.24233 Enhancing Spatial Understanding in Image Generation via Reward Modeling
* 2605.17602 AutoRubric-T2I: Robust Rule-Based Reward Model for Text-to-Image Alignment

Encoders, colour, binding
* 2104.14294 Emerging Properties in Self-Supervised Vision Transformers (DINO)
* 2304.07193 DINOv2: Learning Robust Visual Features without Supervision
* 2008.05659 What Should Not Be Contrastive in Contrastive Learning (LooC)
* 2007.13916 Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases
* 2112.05814 Deep ViT Features as Dense Visual Descriptors
* 2401.00463 Analyzing Local Representations of Self-supervised Vision Transformers
* 2609.09124 Canonical Color as a Lens into Concept Decodability in Vision Encoders and VLMs
* 2510.11835 Data or Language Supervision: What Makes CLIP Better than DINO?
* 2401.06209 Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs (MMVP)
* 2406.16860 Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs
* 2605.31503 How can embedding models bind concepts?
* 2212.10537 Does CLIP Bind Concepts? Probing Compositionality in Large Image Models
* 2512.17178 ABE-CLIP: Training-Free Attribute Binding Enhancement for Compositional Image-Text Matching
* 2502.03566 CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally

Negatives, compositional contrastive training
* 2210.01936 When and why vision-language models behave like bags-of-words (NegCLIP/ARO)
* 2204.03162 Winoground
* 2306.14610 SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality
* 2406.11171 SugarCrepe++
* 2410.05210 Preserving Multi-Modal Capabilities of Pre-trained VLMs (FSC-CLIP)
* 2411.02545 TripletCLIP
* 2402.13254 CounterCurate
* 2312.00081 Synthesize, Diagnose, and Optimize (SPEC)
* 2407.16772 VisMin: Visual Minimal-Change Understanding

Rewards and preferences for diffusion / few-step models
* 2311.12908 Diffusion Model Alignment Using Direct Preference Optimization
* 2311.13231 Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model (D3PO)
* 2406.04314 Step-by-step Preference Optimization (SPO)
* 2406.06382 Diffusion-RPO
* 2305.13301 Training Diffusion Models with Reinforcement Learning (DDPO)
* 2309.17400 Directly Fine-Tuning Diffusion Models on Differentiable Rewards (DRaFT)
* 2310.03739 Aligning Text-to-Image Diffusion Models with Reward Backpropagation (AlignProp)
* 2606.11075 Exploring the Design Space of Reward Backpropagation for Flow Matching (FlowBP)
* 2403.11027 Reward Guided Latent Consistency Distillation (RG-LCD, latent proxy RM)
* 2411.15247 Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward (LaSRO)
* 2404.03673 RL for Consistency Models (RLCM)
* 2411.11727 Aligning Few-Step Diffusion Models with Dense Reward Difference Learning
* 2404.13686 Hyper-SD
* 2405.14867 Improved Distribution Matching Distillation (DMD2)
* 2505.05470 Flow-GRPO: Training Flow Matching Models via Online RL
* 2505.07818 DanceGRPO
* 2507.21802 MixGRPO
* 2508.04324 TempFlow-GRPO
* 2509.16117 DiffusionNFT
* 2509.25050 Advantage Weighted Matching (AWM)
* 2510.08425 Reinforcing Diffusion Models by Direct Group Preference Optimization (DGPO)
* 2508.20751 Pref-GRPO: Pairwise Preference Reward-based GRPO
* 2512.24138 GARDO: Reinforcing Diffusion Models without Reward Hacking
* 2603.07700 TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward
* 2605.26108 Reinforcing Few-step Generators via Reward-Tilted Distribution Matching (RTDMD)
* 2607.00535 Flow-Map GRPO
* 2608.09226 RL-Native Distillation: Exploiting Scored Trajectories for Few-Step Image Generation (REST)
* 2608.24646 On-Policy Self-Distillation in Diffusion Models (DiffusionOPSD)
* 2511.13649 Distribution Matching Distillation Meets Reinforcement Learning (DMDR)
* 2604.19009 Guiding Distribution Matching Distillation with Gradient-Based Reinforcement Learning
* 2604.28126 AdvDMD
* 2604.14910 Reward-Aware Trajectory Shaping for Few-step Visual Generation
* 2512.22802 ReDiF: Reinforced Distillation for Few Step Diffusion
* 2603.14128 Diffusion Reinforcement Learning via Centered Reward Distillation
* 2605.26491 Beyond Pairwise Preferences: Listwise Reward-Aware Alignment for Diffusion Models (LAIR)
* 2510.01540 Towards Better Optimization For Listwise Preference in Diffusion Models
* 2410.18013 Scalable Ranked Preference Optimization for Text-to-Image Generation (RankDPO)
* 2603.18991 CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think
* 2605.28615 Compositional Text-to-Image Generation Via Region-aware Bimodal DPO (BiDPO)
* 2505.22002 D-Fusion: DPO for Aligning Diffusion Models with Visually Consistent Samples
* 2602.06355 Di3PO - Diptych Diffusion DPO
* 2410.07171 IterComp
* 2404.03653 CoMat
* 2312.03626 TokenCompose
* 2301.13826 Attend-and-Excite
* 2306.08877 Linguistic Binding in Diffusion Models (SynGen)
* 2403.05135 ELLA
* 2404.01197 Getting it Right: Improving Spatial Consistency (SPRIGHT)
* 2305.13655 LLM-grounded Diffusion
* 2406.10210 Make It Count
* 2311.17946 DreamSync
* 2403.06952 SELMA
* 2309.15807 Emu
* 2502.12146 Diffusion-Sharpening
* 2501.09732 Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps
* 2508.09968 Noise Hypernetworks: Amortizing Test-Time Compute in Diffusion Models
* 2509.17458 CARINOX: Category-Aware Reward-based Initial Noise Optimization and Exploration
* 2503.13070 Reward-Instruct

Representation alignment
* 2410.06940 Representation Alignment for Generation (REPA)
* 2504.10483 REPA-E
* 2501.01423 Reconstruction vs. Generation (VA-VAE)
* 2512.10794 What matters for Representation Alignment: Global Information or Spatial Structure? (iREPA)
* 2505.16792 REPA Works Until It Doesn't (HASTE)
* 2503.18414 U-REPA
* 2507.01467 Representation Entanglement for Generation (REG)
* 2504.05741 DDT: Decoupled Diffusion Transformer
* 2505.02831 No Other Representation Component Is Needed (SRA)
* 2506.09027 Diffuse and Disperse
* 2510.11690 Diffusion Transformers with Representation Autoencoders (RAE)
* 2503.08250 Aligning Text to Image in Diffusion Models is Easier Than You Think (SoftREPA)
* 2605.30038 Alignment-Guided Score Matching for Text-to-Image Alignment (AGSM)
* 2609.07815 VoT: Vision-of-Thought for Unified Multimodal Representation Alignment
