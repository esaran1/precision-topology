#!/bin/zsh
# Exploration c, one job at a time, memory gate logged before each (nice 15, one thread).
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
L=results/designs/phase2a_explore/p2a_explore_c.log
for args in "$@"; do
  results/designs/phase2a_explore/gate.sh $L
  echo "# args: $args" >> $L
  nice -n 15 .venv/bin/python results/designs/phase2a_explore/p2a_explore_c.py ${=args} >> $L 2>&1
done
