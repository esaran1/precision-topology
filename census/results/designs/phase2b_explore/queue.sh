#!/bin/zsh
# EXPLORATORY: queue seeds FIRST..LAST (seed-major; adam_r1 first per seed, it fixes the matched step budget).
D=/Users/Evan/precision-topology/census/results/designs/phase2b_explore
C=(adam_r1 adam_r4 adam_r16 adam_r64 gd_rho1 gd_scale_r6 gd_scale_r9 gd_plain_r6 gd_plain_r9 gd_wn_r6 gd_wn_r9 adam_wn_r64)
jobs=()
for s in {$2..$3}; do for c in $C; do jobs+=("$c $s"); done; done
$D/run.sh $1 "${jobs[@]}"
echo "$(date '+%Y-%m-%d %H:%M:%S') queue $2..$3 done" >> $D/$1
