#!/bin/zsh
# EXPLORATORY driver, one job at a time (nice 15, one thread), memory gate logged before each job; resumable (a run
# whose JSON exists is skipped).  Usage: run.sh LOG "COND SEED [BUDGET]" ...
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/phase2b_explore
L=$D/$1; shift
for args in "$@"; do
  a=(${=args})
  [[ -f $D/runs/${a[1]}_${a[2]}.json ]] && continue
  $D/gate.sh $L
  echo "# run: $args" >> $L
  nice -n 15 .venv/bin/python $D/p2b_explore.py run ${=args} >> $L 2>&1
done
