#!/bin/zsh
# Memory gate (separate logged step before every job): free >= 25% and swap free >= 500 MB, else wait; disk >= 20 GB, else STOP.
log=${1:-/dev/stdout}
while true; do
  free=$(memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  swapfree=$(sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/')
  disk=$(df -g / | tail -1 | awk '{print $4}')
  ok=$(awk -v f=$free -v s=$swapfree 'BEGIN{print (f>=25 && s>=500)?"OK":"WAIT"}')
  [[ $disk -lt 20 ]] && ok=STOP
  echo "$(date '+%Y-%m-%d %H:%M:%S') memory gate: free ${free}% swap_free ${swapfree}M disk_free ${disk}G -> $ok" >> $log
  [[ $ok == STOP ]] && exit 1
  [[ $ok == OK ]] && exit 0
  sleep 60
done
