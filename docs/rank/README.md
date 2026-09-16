# Prompt-aware ranking (2026-09-16 campaign): what was tested and what was found

Design note: `DESIGN_NOTE.tex` (verbatim). Hypothesis: a fixed text ranking over a caption and its
structured negatives (same objects, one attribute / relation / count / verb changed) bends the
DINO-anchored reward space into one where "closeness to the real photograph" tracks compositional
correctness; aligning the student to that space transfers the structure into generation. Tested on top
of the paper's arm (scored trajectory selection + the refreshed projector reward, `scripts/train.lsf`
with the converged 3k schedule: cosine, batch 16, lr 1e-5, window K={4..7}, 3,000 updates).

## Code

| piece | file |
|---|---|
| negatives (rule-based, audited) | `data/build_negatives.py`, `data/audit_negatives.py`; the campaign's files in `negatives/` (v2 = all families, v3 = colour/texture/verb/shape/count with count delta >= 2) |
| head, rollouts, ranking loss, alignment hook | `train/rank_utils.py` |
| trainer flags | `train/distill.py --rank_*`, `--align_*` (see `--help`); the head file `rank_head_*.pt` carries the head, the alignment projector and the REFRESHED projector |
| held-out probe | `eval/rank_probe.py`, `scripts/rank_probe.lsf` (600 unseen captions; spaces rgb / latent / latent_g / rgb_g; shuffled-anchor control) |
| VQAScore ceiling of the negatives | `eval/vqa_ceiling.py` |
| prompt-aware selection | `data/build_rank_pairs.py` -> `train/rank_head.py` -> `data/rescore_cache.py` |

Arms (pass as `EXTRA` to `scripts/train.lsf` on the ours recipe: `SELECTOR=dino_patch REWARD_MODE=proj REWARD_LAMBDA=80 REWARD_REFRESH=100 REWARD_REFRESH_STEPS=16 ACCUM=4 LR=1e-5 WARMUP=150 LR_SCHEDULE=cosine WINDOW=0.5:0.9 EPOCHS=16 CACHE=cache/train_3k`):

    RANK="--rank_negatives cache/negatives_3k.json --rank_monitor_every 100"
    STG="--rank_head_freeze_step 1000 --rank_shaped_match_norm"
    B1      $RANK --rank_mode head --rank_shaped_reward                                        # head trained throughout, shaped reward from update 0
    B2      $RANK --rank_mode head --align_lambda 60 --align_layer 8                           # REPA-style alignment, last rollout step (sigma 0.009)
    A       $RANK --rank_mode backprop --rank_lambda 1.3 --rank_shaped_reward                  # ranking backprop through the rollouts, both sides
    B1f     $RANK --rank_mode head --rank_shaped_reward $STG                                   # staged: head frozen at 1000, shaped reward norm-matched
    B2m     $RANK --rank_mode head --align_share 0.2 --align_layer 8 --align_step 1 $STG       # alignment at the sigma-0.86 step, calibrated weight
    Af      $RANK --rank_mode backprop --rank_share 0.2 --rank_theta_side positive --rank_shaped_reward $STG
    B1fshuf $RANK --rank_mode head --rank_shaped_reward --rank_negatives_shuffle $STG          # null: other captions' negatives
    B1x/Ax/B1xshuf: the same with --rank_input xhat (ranking on the reward's own inputs) and the v3 negatives

## Results (seed 0 unless stated; converged-schedule references: naive 0.4860 CompBench / 23.0 GenEval2, ours 0.4877 / 23.0)

| arm | CompBench (raw / averaged) | GenEval2 | prompt win rate vs ours | verdict |
|---|---|---|---|---|
| B1 | 0.4866 / 0.4867 | 23.4 / 23.3 | 0.50 | = ours |
| B2 (seeds 1, 2, averaged) | 0.4884, 0.4866 | 23.7, 23.4 | – | = ours |
| A | 0.4014 / – | 16.1 | 0.43 (p 1e-80) | destructive (two-sided push: degrades every category) |
| B1f, seeds 0 / 1 / 2 | 0.4926, 0.4884, 0.4892 / 0.4882, 0.4868, 0.4879 | 23.9, 24.3, 23.3 | 0.506, –, 0.498 raw; 0.494, –, 0.498 averaged | = ours: averaged 0.4876 vs ours 0.4880 (seeds 0, 2); the raw finals sit +0.002 higher on a late-checkpoint rise (seed 0: 0.4873 / 0.4895 / 0.4926 at steps 2000 / 2500 / 3000) that the shuffled null shows too and that averaging removes |
| B2m | 0.4802 / – | 21.1 | 0.47 (p 0.007) | harmful |
| Af | 0.4579 / – | 20.1 | 0.43 (p 2e-10) | harmful (colour -0.08) |
| B1fshuf (null) | 0.4870 / 0.4852 | 23.6 | 0.498 | = ours |
| B1x (round 3: ranking on the reward's inputs, v3 negatives) | 0.4874 / 0.4866 | 22.7 | 0.502 | = ours |
| Ax (round 3, positive-only push, calibrated) | 0.4859 / 0.4861 | 23.5 | 0.489 (p 0.3) | = ours or slightly below |
| B1xshuf (round-3 null) | 0.4878 / 0.4845 | 23.3 | 0.497 | = ours |

Round-3 note: on the one-step inputs the positive and negative estimates differ by only ~0.001 in
projector cosine, so the head's Jacobian gain at the switch was 7-8x (norm-matched scale 0.125-0.149),
and the heads trained there do not carry over to rollouts (held-out latent_g 0.52-0.55 vs latent 0.54).

Held-out probe (600 unseen captions, pairwise accuracy positive > negative): raw DINO on decoded images
0.55-0.59 for every student (base 0.547, naive 0.586, ours 0.581); the latent projector 0.50-0.54 (chance);
trained heads 0.58-0.60 in projector space (+5 to +9 points, anchor-dependent: 0.50-0.53 against a
shuffled photo) and +1.5 to +3 points on true DINO. `probe/*.json` holds every record.

Prompt-aware SELECTION (`rescore_summary.json`, `head_dino_metrics.json`): a head trained on true DINO
features of teacher images generalises (0.580 -> 0.651 held-out, but 0.616 against a shuffled anchor)
and makes candidate selection WORSE against the VQAScore oracle stored in the cache: agreement 0.311
vs raw DINO's 0.356, mean VQAScore of the pick 0.867 vs 0.882 (random 0.843, oracle 0.933).

Debug records (`debug/`): `per_prompt_analysis.md` (noise floor: sd 0.12 per prompt between equivalent
runs, minimal detectable effect 0.0067 per run; the reward family beats naive only by a sign test,
+0.0024, win rate 0.533), `reward_state_probe.md` (a head trained on 4-step rollouts does not transfer
to the reward's one-step inputs: on the ours student it lowers the ordering from 0.70-0.80 raw to
0.52-0.58), `vqa_ceiling.md` (79% of v2 negatives are real contradictions; colour 0.96, texture 0.91,
shape 0.85, verb 0.78, count 0.63, 3d_spatial 0.68, spatial 0.54). Review of the round-1 code and design
by six independent lenses: `REVIEW.md`, raw findings `review_raw.json`.

## Reading

The representation half of the hypothesis holds modestly: a text ranking makes a DINO-anchored space
prompt-aware on unseen captions and the effect depends on the caption's own photo. The generation half
has not held: every transfer path (shaped reward, over-driven or calibrated; alignment at a clean or a
noisy step; ranking pushed into the student on both sides or the positive only; the shaped scorer used
for candidate selection) matches or hurts the paper's arm. Mechanism, as measured: the head learns
"edited-prompt-ness", separable from the photo but not compositional correctness; the anchor scorer
cannot see the relations that have benchmark headroom (raw DINO at chance on colour, count and left /
right), the student cannot realise relation edits (VQAScore), and the reward channel itself is nearly
inert on the converged schedule (ours vs naive +0.002).
