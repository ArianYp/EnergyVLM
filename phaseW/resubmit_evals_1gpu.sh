#!/usr/bin/env bash
# Convert still-PENDING evaluation jobs of a chain ledger (TAG SEED TRAIN EVAL AVG AVGEVAL per line) to
# the single-GPU evaluation launcher, keeping the chain dependencies (raw eval after the training,
# averaged eval after the average). Alternates the reservation between RES_A and RES_B per chain so
# the jobs backfill single free cards on both. Rewrites the ledger in place (backup *.bak1gpu).
#   bash phaseW/resubmit_evals_1gpu.sh phaseW/s4j_jobs.txt
set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM
LEDGER=${1:?ledger}
RES_A=${RES_A-"-U iclr_2026"}
RES_B=${RES_B-""}
NEW=$LEDGER.new; : > $NEW; moved=0; i=0
while read TAG S T E1 AV E2; do
  i=$((i+1)); RES=$RES_A; [ $((i % 2)) -eq 0 ] && RES=$RES_B
  RUN=$(ls -d checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T} 2>/dev/null || echo "checkpoints/phaseS4/phaseS4_${TAG}_s${S}_${T}")
  s1=$(bjobs -noheader -o stat $E1 2>/dev/null || true); s2=$(bjobs -noheader -o stat $E2 2>/dev/null || true)
  if [ "$s1" = "PEND" ]; then
    bkill $E1 >/dev/null 2>&1 || true
    E1=$(bsub $RES -w "done($T)" -env "all,EVAL_LABEL=S4_${TAG}_s${S},EVAL_CKPT=$RUN/checkpoint_final.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment_1gpu.lsf | grep -oE "[0-9]{6}"); moved=$((moved+1))
  fi
  if [ "$s2" = "PEND" ]; then
    bkill $E2 >/dev/null 2>&1 || true
    E2=$(bsub $RES -w "done($AV)" -env "all,EVAL_LABEL=S4_${TAG}-avglast3_s${S},EVAL_CKPT=$RUN/checkpoint_avg_last5.pt,EVAL_CFG=1.0" < ablations/phaseN_eval_alignment_1gpu.lsf | grep -oE "[0-9]{6}"); moved=$((moved+1))
  fi
  echo "$TAG $S $T $E1 $AV $E2" >> $NEW
done < $LEDGER
cp $LEDGER $LEDGER.bak1gpu; mv $NEW $LEDGER
echo "$LEDGER: $moved pending evaluations resubmitted as single-GPU jobs"
