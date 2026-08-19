#!/usr/bin/env bash
# Wait for a training slot, then train the lzh sentence-segmenting base arm.
# See configs/config_lzh_seg.cfg for what it changes and why.
set -uo pipefail
cd "$(dirname "$0")/.."
MAX=${MAX_TRAINS:-1}          # launch once this many OTHER spacy trainings remain
while :; do
  n=$(pgrep -f "spacy train" | wc -l | tr -d ' ')
  [ "$n" -le "$MAX" ] && break
  sleep 120
done
exec .venv/bin/python -u -m spacy train configs/config_lzh_seg.cfg \
  --output training_lzh_seg/ --code scripts/seg_code.py --system.seed 0
