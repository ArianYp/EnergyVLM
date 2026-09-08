#!/usr/bin/env bash
# S4i: the selection x reward factorial on the 3k pool, predeclared budget FIVE seeds per cell.
#   cell (s=0, r=0)  random pick, no reward       B2               seeds 0-2 exist (S4); seeds 3-4 added here
#   cell (s=1, r=0)  argmax, no reward            CD_dinop_hard    seeds 0-2 exist (S4); seeds 3-4 added here
#   cell (s=0, r=1)  random pick + exact reward   B2-rewXi         seeds 0-4 (the missing cell)
#   cell (s=1, r=1)  argmax + exact reward        CD_dinop_hard-rewXi  seeds 0-4 (re-run of S4h under the
#                    RNG-restored trainer so its caption order matches the reward-free arms at the same seed)
# Reward recipe fixed at the S4h values (rgb, lambda 15.504, two states, monitor every 100) plus a gradient
# probe every 500 updates. Each training is chained to its raw eval, checkpoint average and averaged eval.
set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM
LEDGER=phaseW/s4i_jobs.txt
REW="--reward_mode rgb --reward_lambda 15.504 --reward_grad_probe_every 500"
submit () {  # variant seed trainer extra tagx
  local V=$1 S=$2 TR=$3 EX=$4 TX=$5
  local TAG="${V}${TX}"
  local T=$(bsub -U iclr_2026 -env "all,VARIANT=$V,COUP=fresh,PSIG=0,SEED=$S,TRAINER=$TR,EXTRA=$EX,TAGX=$TX" < ablations/phaseW_train_s4.lsf | grep -oE "[0-9]{6}")
  local RUN="checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T}"
  local E1=$(bsub -U iclr_2026 -w "done($T)" -env "all,EVAL_LABEL=S4_${TAG}_s${S},EVAL_CKPT=$RUN/checkpoint_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  local AV=$(bsub -U iclr_2026 -w "done($T)" -env "all,RUN=$RUN,STEPS=2000:4000:final" < phaseW/average_ckpts.lsf | grep -oE "[0-9]{6}")
  local E2=$(bsub -U iclr_2026 -w "done($AV)" -env "all,EVAL_LABEL=S4_${TAG}-avglast3_s${S},EVAL_CKPT=$RUN/checkpoint_avg_last5.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  echo "$TAG $S $T $E1 $AV $E2" | tee -a $LEDGER
}
for S in 0 1 2 3 4; do
  submit B2            $S phaseW/train_pilot_frozen_s4i.py "$REW" -rewXi
  submit CD_dinop_hard $S phaseW/train_pilot_frozen_s4i.py "$REW" -rewXi
done
for S in 3 4; do
  submit B2            $S phaseW/train_pilot_frozen_s4.py "" ""
  submit CD_dinop_hard $S phaseW/train_pilot_frozen_s4.py "" ""
done
echo "submitted $(wc -l < $LEDGER) chains -> $LEDGER"
