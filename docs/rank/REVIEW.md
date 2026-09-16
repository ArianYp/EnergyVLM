# Review of the prompt-aware ranking campaign (2026-09-16)

Six independent review lenses (code / design fidelity / negatives / runs+stats / eval+probe / metrics) plus
adversarial verification of the first issues; the workflow was stopped before synthesis to save quota. Raw
findings and verdicts: `review_raw.json`. This file is the recovered synthesis.

## Verdict
The plumbing is correct (every lens validated the gradient paths, DDP parity, RNG isolation, scheduler
restore, checkpoint/hook interplay, eval chain settings and the caption->negative join). The problems are
in what the round-1 contrasts can conclude, and round 1's seed-0 results are consistent with the review:
B1 s0 = ours on CompBench (0.4866 vs 0.4877) and GenEval2 (23.4 vs 23.0); its head reaches 0.72-0.74
pairwise accuracy on training captions but 0.585 held-out in the projector space and adds nothing on
true DINO (0.607 vs 0.612 raw). Arm A's consistency loss rises and its true-DINO reward score falls
after step 1500 while its shaped reward climbs to 0.71: the generator is gaming the head.
The hypothesis is falsified if, with a frozen head and norm-matched weights, (a) the head's held-out
ordering does not beat the shuffled-anchor and shuffled-negatives controls, or (b) the student's
CompBench colour/spatial/numeracy categories do not move while the ranking metrics do.

## Confirmed issues (verified) and fixes
1. B2's alignment hooked the LAST rollout forward (input sigma 0.009): a self-alignment regulariser on a
   finished image, not the note's transfer at a noisy timestep. Fix: `--align_step 1` (sigma 0.86). Round-1 B2
   results test the weakest Variant B.
2. The head's Jacobian inflates the shaped reward 2-4x and rotates it against the consistency gradient, so
   B1/A-vs-ours confound geometry with reward strength; in A the ranking+reward share exceeded the intended 20%.
   Fix: staged recipe (`--rank_head_freeze_step 1000`): plain ours + head training, then head frozen and the
   switch calibrated once (`--rank_shaped_match_norm`, `--rank_share 0.2`, `--align_share 0.2`).
3. The refreshed projector was never saved, so trained heads were probed against the pretrained projector.
   Fix: head files now carry `proj`; the probe uses it. Round-1 latent_g numbers keep this caveat.
4. The held-out probe leaked 70/300 captions from the projector's pretraining manifest and 9 replay photos.
   Fix: excluded (probe now 600 captions); only the latent columns were affected.
5. Arm A pushes the negatives' images away from the photo (degrade-the-loser channel, unmonitored). Fix:
   `--rank_theta_side positive` for the next A.
6. Negatives: the surface-noun skip after "on top of" only caught caption-final nouns (regex, fixed);
   41% of three-negative captions spent all three edits on one word (now distinct spans first; the
   remaining single-span captions keep repeats). Dominant families (colour 32%, verb, count) are where a
   patch-mean DINO cosine has the least signal (probe: colour 0.56-0.59 even for true DINO).
7. Statistics: with 3 seeds the minimal detectable effect is ~0.01-0.02 CompBench, above the effects in play;
   the 2x8 controls are not seed-paired with the 4x4 arms. Treat CompBench as a screen; the 600-caption
   probe with the shuffled-anchor control is the primary mechanistic endpoint.

## Checked and fine (refuted or immaterial)
- Probe-subset claim (i % 4 skips attention weights): an artefact of indexing the state_dict; parameters() is correct.
- Routing the shaped reward through detached head weights: correct and immaterial.
- The pretraining-manifest overlap of the probe is expected (same 118k pool) and only biases the latent columns.

## Recommended additions (information per GPU-hour)
1. Shuffled-negatives null arm (other captions' negatives): run now as `B1fshuf`.
2. Shuffled-anchor accuracy in the probe (done): a head that still ranks the positive first against another
   photo learned "typical positive rollout", not agreement with the caption's photo.
3. VQAScore on positive vs negative generations per family: the O1 ceiling (do the images differ at all?).
   One t2v_metrics pass over the probe's decoded rollouts, ~1 GPU-hour.
4. Frozen-head arms (done: B1f, Af, B2m). 5. Per-family CompBench mapping (colour->color, count->numeracy,
   spatial->spatial/3d_spatial, texture->texture, shape->shape) in the final table. 6. Larger probe (1000
   captions) once the arms exist. 7. Train the head on true DINO features (decode is cheap) if the projector
   space stays at chance.

## Reading round 2 (seed 0 of B1f / Af / B2m / B1fshuf)
Success = held-out latent_g (with the refreshed P) clearly above latent AND above B1fshuf's, rgb_g >= rgb,
shuffled-anchor accuracy near chance, and CompBench colour/spatial/numeracy up vs ours with GenEval2 not
down. Null = probe gains without benchmark movement. Shortcut = B1fshuf matches B1f, or shuffled-anchor
accuracy tracks the real one.
