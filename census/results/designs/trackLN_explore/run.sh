#!/bin/zsh
# EXPLORATORY detached driver for LN_explore.py (Track LN): ONE python process for all jobs (nice 15, one thread); the
# memory gate is logged as a separate step before the job (gate.sh) and inside LN_explore.py before every run;
# resumable per (seed, arm).  Usage: nohup run.sh LOG OUT SEED:ARM ... &!
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/trackLN_explore
L=$D/$1; O=$2; shift 2
$D/gate.sh $D/memory_gate.log || { echo "$(date '+%F %T') gate STOP (disk)" >> $L; exit 1; }
echo "# $(date '+%F %T') start: $*" >> $L
nice -n 15 .venv/bin/python $D/LN_explore.py $O "$@" >> $L 2>&1
echo "# $(date '+%F %T') exit $?" >> $L
