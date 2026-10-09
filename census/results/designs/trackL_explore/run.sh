#!/bin/zsh
# EXPLORATORY driver for L_explore.py (Track L, L2): one job at a time (nice 15, one thread); the memory gate (gate.sh)
# is logged as a separate step before each run; resumable (L_explore.py skips a key already in OUT).
# Usage: run.sh LOG OUT "P ARM SEED [LR] [WIDTH] [CAP]" ...
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/trackL_explore
L=$D/$1; O=$2; shift 2
for args in "$@"; do
  $D/gate.sh $D/memory_gate.log || { echo "$(date '+%F %T') gate STOP (disk)" >> $L; exit 1; }
  echo "# $(date '+%F %T') run: $args" >> $L
  nice -n 15 .venv/bin/python $D/L_explore.py $O ${=args} >> $L 2>&1 || { echo "exit $? on $args" >> $L; exit 1; }
done
echo "# $(date '+%F %T') done" >> $L
