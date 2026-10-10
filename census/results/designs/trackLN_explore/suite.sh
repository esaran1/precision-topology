#!/bin/zsh
# Gated two-process test suite (sequential, one thread; no outer nice: tests that call os.nice(15) fail with EPERM under nice 15 — see ../trackL_explore/L_suite_main_nice15.log); exit statuses logged.  Usage: suite.sh TAG
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/trackLN_explore
T=${1:-final}
$D/gate.sh $D/memory_gate.log || exit 1
echo "# $(date '+%F %T') suite main start" >> $D/LN_suite_main_$T.log
.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_verify_certificates.py >> $D/LN_suite_main_$T.log 2>&1
echo "exit $?" >> $D/LN_suite_main_$T.log
$D/gate.sh $D/memory_gate.log || exit 1
echo "# $(date '+%F %T') suite certs start" >> $D/LN_suite_certs_$T.log
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_verify_certificates.py >> $D/LN_suite_certs_$T.log 2>&1
echo "exit $?" >> $D/LN_suite_certs_$T.log
.venv/bin/python -m src.verify_ledger > $D/LN_ledger_$T.log 2>&1
echo "exit $?" >> $D/LN_ledger_$T.log
