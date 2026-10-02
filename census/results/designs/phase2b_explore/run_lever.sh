#!/bin/zsh
# EXPLORATORY driver for p2b_lever.py: one job at a time (nice 15, one thread), memory gate (gate.sh) logged as a
# separate step before each job; resumable (a run whose JSON exists is skipped). Usage: run_lever.sh LOG "COND SEED" ...
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/phase2b_explore
L=$D/$1; shift
for args in "$@"; do
  a=(${=args})
  [[ -f $D/runs_lever/${a[1]}_${a[2]}.json ]] && continue
  $D/gate.sh $L
  echo "# run: $args" >> $L
  nice -n 15 .venv/bin/python $D/p2b_lever.py run ${=args} >> $L 2>&1 || { echo "exit $? on $args" >> $L; exit 1; }
done
