#!/usr/bin/env bash
# S4j: component and hyperparameter ablations of the exact-reward recipe, ONE factor per arm against the
# full recipe (argmax + rgb reward, lambda 15.504, two least-noisy states, antialiased bicubic resize,
# fp32 DINO = the factorial's CD_dinop_hard-rewXi arm), three seeds each, RNG-restored trainer
# phaseW/train_pilot_frozen_s4j.py (= s4i + the ablation flags; defaults verified against s4i).
#   lambda        7.75 | 31 | 62                 (15.5 is the recipe; 62 probes over-optimisation)
#   states R      1 | 5                          (2 is the recipe; 5 = every supervised state)
#   which states  the two NOISIEST               (recipe: the two least-noisy)
#   resize        plain bilinear                 (recipe: antialiased bicubic)
#   DINO dtype    bf16                           (recipe: fp32)
#   projector reward, is the refresh under-driven: 16 refresh steps | refresh every 25 updates
set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM
LEDGER=phaseW/s4j_jobs.txt
TR=phaseW/train_pilot_frozen_s4j.py
RES_TRAIN=${RES_TRAIN-"-U iclr_2026"}     # RES_TRAIN="" sends the trainings to the general pool (H100/A100/H200 hosts)
RES_EVAL=${RES_EVAL-"-U iclr_2026"}
RGB="--reward_mode rgb --reward_grad_probe_every 500"
PRJ="--reward_proj phaseW/latent_scorer/projector/projector.pt --reward_lambda 80 --reward_refresh_every 100"
submit () {  # variant seed extra tagx
  local V=$1 S=$2 EX=$3 TX=$4 TAG="$1$4"
  local T=$(bsub $RES_TRAIN -env "all,VARIANT=$V,COUP=fresh,PSIG=0,SEED=$S,TRAINER=$TR,EXTRA=$EX,TAGX=$TX" < ablations/phaseW_train_s4.lsf | grep -oE "[0-9]{6}")
  local RUN="checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T}"
  local E1=$(bsub $RES_EVAL -w "done($T)" -env "all,EVAL_LABEL=S4_${TAG}_s${S},EVAL_CKPT=$RUN/checkpoint_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  local AV=$(bsub $RES_TRAIN -w "done($T)" -env "all,RUN=$RUN,STEPS=2000:4000:final" < phaseW/average_ckpts.lsf | grep -oE "[0-9]{6}")
  local E2=$(bsub $RES_EVAL -w "done($AV)" -env "all,EVAL_LABEL=S4_${TAG}-avglast3_s${S},EVAL_CKPT=$RUN/checkpoint_avg_last5.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  echo "$TAG $S $T $E1 $AV $E2" | tee -a $LEDGER
}
for S in 0 1 2; do
  submit CD_dinop_hard $S "$RGB --reward_lambda 7.752"                                  -rewXi-l7.75
  submit CD_dinop_hard $S "$RGB --reward_lambda 31.008"                                 -rewXi-l31
  submit CD_dinop_hard $S "$RGB --reward_lambda 62.016"                                 -rewXi-l62
  submit CD_dinop_hard $S "$RGB --reward_lambda 15.504 --reward_states 1"               -rewXi-R1
  submit CD_dinop_hard $S "$RGB --reward_lambda 15.504 --reward_states 5"               -rewXi-R5
  submit CD_dinop_hard $S "$RGB --reward_lambda 15.504 --reward_states_pick noisiest"   -rewXi-noisiest
  submit CD_dinop_hard $S "$RGB --reward_lambda 15.504 --reward_resize bilinear"        -rewXi-bilinear
  submit CD_dinop_hard $S "$RGB --reward_lambda 15.504 --reward_dino_dtype bf16"        -rewXi-bf16
  submit CD_dinop_hard $S "$PRJ --reward_refresh_steps 16"                              -rewRi-s16
  submit CD_dinop_hard $S "$PRJ --reward_refresh_every 25"                              -rewRi-e25
done
echo "submitted $(wc -l < $LEDGER) chains -> $LEDGER"
