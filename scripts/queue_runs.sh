#!/usr/bin/env bash
# One SERIAL queue for everything waiting on a training slot. Runs each job only when at most
# $MAX_TRAINS other `spacy train` processes are alive, so queued work never piles onto a busy box.
# Two independent waiters would race for the same slot, which is why there is exactly one of these.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX=${MAX_TRAINS:-1}

wait_slot() {
  while :; do
    n=$(pgrep -f "spacy train" | grep -vc "^$$\$" || true)
    n=$(pgrep -f "spacy train" | wc -l | tr -d ' ')
    [ "$n" -le "$MAX" ] && return
    sleep 120
  done
}

run() {                       # run <logfile> <args...>
  local log=$1; shift
  wait_slot
  echo "=== $(date '+%F %T') starting: $*" >> queue_runs.log
  $PY -u -m spacy train "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') finished ($?): $*" >> queue_runs.log
  sleep 30
}

# 1. lzh: the base arm that learns sentence boundaries (see configs/config_lzh_seg.cfg).
run train_lzh_seg.log configs/config_lzh_seg.cfg --output training_lzh_seg/ \
    --code scripts/seg_code.py --system.seed 0

# 2-3. Multi-seed confirmation of the Sanskrit word-order augmentation. Seed 1 gave +1.85 test LAS;
#      one seed is not a result in this project (docs/, NEGATIVE-RESULTS.md both turn on this).
for s in 0 2; do
  run train_sa_mp2_order_s$s.log configs/config_sa_mp2_order.cfg \
      --output training_sa_mp2_order_s$s/ --code scripts/seg_code.py --system.seed $s
done
