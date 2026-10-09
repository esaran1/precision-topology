#!/bin/zsh
# EXPLORATORY driver for A_explore.py (Track L output-multiplier follow-up, motivated by the Track L null): one job at
# a time (nice 15, one thread); the memory gate (gate.sh) is logged as a separate step before each run; resumable
# (A_explore.py skips a key already in OUT).  Waits while another heavy python job (>1 GB RSS) is running.
# Usage: run.sh LOG OUT "ALPHA MODE ARM SEED [CAP]" ...
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/trackL_alpha_explore
L=$D/$1; O=$2; shift 2
for args in "$@"; do
  while ps -axo rss=,comm= | awk '$2 ~ /[Pp]ython/ && $1 > 1000000 {f=1} END {exit !f}'; do
    echo "$(date '+%F %T') another heavy python job running: wait 60 s" >> $L; sleep 60
  done
  $D/gate.sh $D/memory_gate.log || { echo "$(date '+%F %T') gate STOP (disk)" >> $L; exit 1; }
  echo "# $(date '+%F %T') run: $args" >> $L
  nice -n 15 .venv/bin/python $D/A_explore.py $O ${=args} >> $L 2>&1 || { echo "exit $? on $args" >> $L; exit 1; }
done
echo "# $(date '+%F %T') done" >> $L
