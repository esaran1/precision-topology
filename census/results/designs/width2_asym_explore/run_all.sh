#!/bin/zsh
# EXPLORATORY: reproduces every file in this directory, in order (one process at a time, nice 15, one thread).
# Check memory before each step (memory_pressure -Q free >= 25%, swap free >= 500 MB).
set -e
cd "$(dirname "$0")"
PY=/Users/Evan/precision-topology/census/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
R() { nice -n 15 $PY "$@"; }
R w2_explore_a.py > w2_explore_a.log                                   # equal shares, s0 = 0.2225
R w2_explore_a.py 2.5398 1.0 20000 > w2_explore_a_s2.5398.log          # equal shares, s0 = half of D's switch
R w2_explore_a.py 0.2225397 1.0 20000 0.1 100 > w2_explore_a_s0.2225_u0.1.log
R w2_explore_b.py > w2_explore_b.log
W2A=w2_explore_a_s0.2225_u0.1.json R w2_explore_b.py > w2_explore_b_s0.2225_u0.1.log
R w2_explore_d.py > w2_explore_d.log
R w2_explore_e.py > w2_explore_e.log
R w2_explore_f.py > w2_explore_f.log
{ R w2_explore_c.py 0.3 1.0 200000 0 2          # arm D candidate (duplicate classes D, D')
  R w2_explore_c.py 1.0 1.0 200000 0
  for rho in 1.0 0.1 0.01 0.003 0.001 0.0003; do
    W2A=w2_explore_a_s0.2225_u0.1.json W2B=w2_explore_b_s0.2225_u0.1.json R w2_explore_c.py 0.3 $rho 400000 0 2
  done
  W2A=w2_explore_a_s0.2225_u0.1.json W2B=w2_explore_b_s0.2225_u0.1.json R w2_explore_c.py 0.03 0.001 400000 0   # arm T candidate
} > w2_explore_c.log
python3 w2_explore_g.py > w2_explore_g.log
# added after the author's approval with changes A-D (T′ feasibility; population and used seeds only)
R w2_explore_h.py > w2_explore_h.log
{ for e r in 0.03 0.001 0.03 0.0003 0.3 0.001 0.3 0.003 0.3 0.01; do
    CONT=1 W2A=w2_explore_a_s0.2225_u0.1.json W2B=w2_explore_b_s0.2225_u0.1.json R w2_explore_c.py $e $r 1000000 2
  done; } > w2_explore_c_Tprime.log
python3 w2_explore_i.py > w2_explore_i.log
