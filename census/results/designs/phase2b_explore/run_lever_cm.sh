#!/bin/zsh
# EXPLORATORY driver for p2b_lever_cm.py (one job at a time, nice 15, one thread, memory gate logged before each job).
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/phase2b_explore
for s in {2953000..2953019}; do
  [[ -f $D/runs_lever/globcm_$s.json ]] && continue
  $D/gate.sh $D/p2b_lever_cm.log
  echo "# run: globcm $s" >> $D/p2b_lever_cm.log
  nice -n 15 .venv/bin/python $D/p2b_lever_cm.py run $s >> $D/p2b_lever_cm.log 2>&1 || { echo "exit $? on $s" >> $D/p2b_lever_cm.log; exit 1; }
done
echo "$(date '+%F %T') globcm queue done" >> $D/p2b_lever_cm.log
