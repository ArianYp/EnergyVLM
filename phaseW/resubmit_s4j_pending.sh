#!/usr/bin/env bash
# Move the still-PENDING S4j chains to another reservation: kill the four jobs of each pending chain and
# resubmit the chain (same arm, same seed, same flags) with RES_TRAIN / RES_EVAL. Lines whose training
# is already running or done are left alone. Rewrites phaseW/s4j_jobs.txt.
set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM
RES_TRAIN=${RES_TRAIN-"-U lotfollahi-training-normal"}
RES_EVAL=${RES_EVAL-"-U iclr_2026"}
MAXMOVE=${MAXMOVE:-99}
TR=phaseW/train_pilot_frozen_s4j.py
RGB="--reward_mode rgb --reward_grad_probe_every 500"
PRJ="--reward_proj phaseW/latent_scorer/projector/projector.pt --reward_lambda 80 --reward_refresh_every 100"
declare -A EXTRA=(
  [-rewXi-l7.75]="$RGB --reward_lambda 7.752"   [-rewXi-l31]="$RGB --reward_lambda 31.008"   [-rewXi-l62]="$RGB --reward_lambda 62.016"
  [-rewXi-R1]="$RGB --reward_lambda 15.504 --reward_states 1"   [-rewXi-R5]="$RGB --reward_lambda 15.504 --reward_states 5"
  [-rewXi-noisiest]="$RGB --reward_lambda 15.504 --reward_states_pick noisiest"
  [-rewXi-bilinear]="$RGB --reward_lambda 15.504 --reward_resize bilinear"   [-rewXi-bf16]="$RGB --reward_lambda 15.504 --reward_dino_dtype bf16"
  [-rewRi-s16]="$PRJ --reward_refresh_steps 16"   [-rewRi-e25]="$PRJ --reward_refresh_every 25"
)
NEW=phaseW/s4j_jobs.txt.new; : > $NEW; moved=0
while read TAG S T E1 AV E2; do
  st=$(bjobs -noheader -o stat $T 2>/dev/null || true)
  # move chains whose training is still pending, or was killed (EXIT) by an earlier aborted move
  if { [ "$st" != "PEND" ] && [ "$st" != "EXIT" ]; } || [ $moved -ge $MAXMOVE ]; then echo "$TAG $S $T $E1 $AV $E2" >> $NEW; continue; fi
  V=CD_dinop_hard; TX=${TAG#$V}; EX=${EXTRA[$TX]}
  bkill $T $E1 $AV $E2 >/dev/null 2>&1 || true
  T2=$(bsub $RES_TRAIN -env "all,VARIANT=$V,COUP=fresh,PSIG=0,SEED=$S,TRAINER=$TR,EXTRA=$EX,TAGX=$TX" < ablations/phaseW_train_s4.lsf | grep -oE "[0-9]{6}")
  RUN="checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T2}"
  E1b=$(bsub $RES_EVAL -w "done($T2)" -env "all,EVAL_LABEL=S4_${TAG}_s${S},EVAL_CKPT=$RUN/checkpoint_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  AVb=$(bsub $RES_TRAIN -w "done($T2)" -env "all,RUN=$RUN,STEPS=2000:4000:final" < phaseW/average_ckpts.lsf | grep -oE "[0-9]{6}")
  E2b=$(bsub $RES_EVAL -w "done($AVb)" -env "all,EVAL_LABEL=S4_${TAG}-avglast3_s${S},EVAL_CKPT=$RUN/checkpoint_avg_last5.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  echo "$TAG $S $T2 $E1b $AVb $E2b" | tee -a $NEW; moved=$((moved+1))
done < phaseW/s4j_jobs.txt
mv phaseW/s4j_jobs.txt phaseW/s4j_jobs.txt.bak; mv $NEW phaseW/s4j_jobs.txt
echo "moved $moved chains to '$RES_TRAIN'; ledger rewritten (backup phaseW/s4j_jobs.txt.bak)"
