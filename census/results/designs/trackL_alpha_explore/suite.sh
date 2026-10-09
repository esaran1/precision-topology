#!/bin/zsh
# Gated two-process test suite (sequential, one thread; no outer nice: tests that call os.nice(15) fail with EPERM under nice 15 — see ../trackL_explore/L_suite_main_nice15.log); exit statuses logged.
cd /Users/Evan/precision-topology/census
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=results/designs/trackL_alpha_explore
$D/gate.sh $D/A_suite_main.log
.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_verify_certificates.py >> $D/A_suite_main.log 2>&1
echo "exit $?" >> $D/A_suite_main.log
$D/gate.sh $D/A_suite_certs.log
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_verify_certificates.py >> $D/A_suite_certs.log 2>&1
echo "exit $?" >> $D/A_suite_certs.log
