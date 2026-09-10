#!/usr/bin/env bash
# S4m: does an ONLINE EMA of the student (checkpoint_ema_final.pt, decay updated every step per the
# diffusers/iCT formula already implemented in the trainer -- see train_pilot_frozen_s4j.py:1350-1359)
# match or beat our current POST-HOC practice of averaging the last 3-5 raw checkpoints
# (checkpoint_avg_last5.pt)? Same recipe as S4i's best cell (CD_dinop_hard-rewXi), same seeds, same
# trainer -- the ONLY change is adding --ema_decay, so all three checkpoints (raw final, avg-last-5,
# ema-final) come out of the SAME training run and are directly comparable at matched compute.
#
# Decay choice: literature review (iCT, arXiv:2310.14189) uses a REPORTING-only EMA (decoupled from
# any in-loop teacher EMA, which does not apply to us -- our teacher is frozen, not a self-EMA) with
# a half-life around 3% of the total run. Scaled to our num_steps=6000:
#   decay = 1 - ln(2) / (0.03 * 6000) = 0.9961   (half-life ~180 steps)
# The trainer's (1+s)/(10+s) warm-up (diffusers EMAModel formula) applies automatically since
# --ema_no_warmup is not passed.
#
# Farm note (2026-09-10): the training job requests gmem=78G, and the code comment at line 636 says
# the EMA shadow copy adds ~+10 GB/GPU for SD3.5-M -- on an 80 GB-physical A100/H100 card that could
# exceed hardware capacity even though LSF's 78G *reservation* would still "fit" on paper. So training
# stays on iclr_2026 (H200, 140 GB physical, real headroom); the idle lotfollahi-training-normal
# A100 pool (13/16 GPUs free when checked) is used for the ckpt-average and eval steps instead, which
# request far less memory (10G / 40G) and carry no such risk.
set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM
LEDGER=phaseW/s4m_ema_jobs.txt
REW="--reward_mode rgb --reward_lambda 15.504 --reward_grad_probe_every 500 --ema_decay 0.9961"
RES_TRAIN=${RES_TRAIN-"-U iclr_2026"}
RES_EVAL=${RES_EVAL-"-q training-normal -U lotfollahi-training-normal"}
for S in 0 1 2; do
  V=CD_dinop_hard; TAG="${V}-rewXi-emaF"
  ENV="all,VARIANT=$V,COUP=fresh,PSIG=0,SEED=$S,TRAINER=phaseW/train_pilot_frozen_s4j.py,EXTRA=$REW,TAGX=-rewXi-emaF"
  T=$(bsub $RES_TRAIN -env "$ENV" < ablations/phaseW_train_s4.lsf | grep -oE "[0-9]{6}")
  RUN="checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T}"
  E1=$(bsub $RES_EVAL -w "done($T)" -env "all,EVAL_LABEL=S4_${TAG}_s${S},EVAL_CKPT=$RUN/checkpoint_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  AV=$(bsub $RES_EVAL -w "done($T)" -env "all,RUN=$RUN,STEPS=2000:4000:final" < phaseW/average_ckpts.lsf | grep -oE "[0-9]{6}")
  E2=$(bsub $RES_EVAL -w "done($AV)" -env "all,EVAL_LABEL=S4_${TAG}-avglast3_s${S},EVAL_CKPT=$RUN/checkpoint_avg_last5.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  E3=$(bsub $RES_EVAL -w "done($T)" -env "all,EVAL_LABEL=S4_${TAG}-emafinal_s${S},EVAL_CKPT=$RUN/checkpoint_ema_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment.lsf | grep -oE "[0-9]{6}")
  echo "$TAG $S $T $E1 $AV $E2 $E3" | tee -a $LEDGER
done
echo "submitted $(wc -l < $LEDGER) chains -> $LEDGER"
