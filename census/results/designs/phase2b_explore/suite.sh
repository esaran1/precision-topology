#!/bin/zsh
# Gated two-process test suite (sequential, nice 15, one thread); exit statuses logged.
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/phase2b_explore
$D/gate.sh $D/p2b_suite_main.log
nice -n 15 .venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_verify_certificates.py >> $D/p2b_suite_main.log 2>&1
echo "exit $?" >> $D/p2b_suite_main.log
$D/gate.sh $D/p2b_suite_certs.log
nice -n 15 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_verify_certificates.py >> $D/p2b_suite_certs.log 2>&1
echo "exit $?" >> $D/p2b_suite_certs.log
