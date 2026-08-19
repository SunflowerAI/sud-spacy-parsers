#!/usr/bin/env bash
# Rebuild the lzh stack on the SENTENCE-SEGMENTING base (training_lzh_seg).
#
# `seg` is a BASE recipe, not a stackable layer (CLAUDE.md; zh paid for this), so morph, the
# conditioned tagger and the SUD MISC pipes all have to be retrained against it. Each step sources
# the previous one, so they run IN ORDER — the later configs will not even validate until their
# source directory exists.
#
# ⚠ This waits for scripts/queue_runs.sh to EXIT before doing anything. Two waiters polling the same
# slot would both fire at once, which is the pile-up the queue exists to prevent — and NEVER append
# to a queue script bash is already executing: bash re-seeks by byte offset and can run the new text
# as garbage.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX=${MAX_TRAINS:-1}

while pgrep -f queue_runs.sh > /dev/null; do sleep 120; done

wait_slot() {
  while :; do
    n=$(pgrep -f "spacy train" | wc -l | tr -d ' ')
    [ "$n" -le "$MAX" ] && return
    sleep 120
  done
}

run() {
  local log=$1; shift
  wait_slot
  echo "=== $(date '+%F %T') starting: $*" >> queue_runs.log
  $PY -u -m spacy train "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') finished ($?): $*" >> queue_runs.log
  sleep 30
}

run train_lzh_seg_morph.log configs/config_lzh_seg_morph.cfg --output training_lzh_seg_morph/ \
    --code scripts/seg_code.py --system.seed 0
run train_lzh_seg_xposwarm.log configs/config_lzh_seg_xposwarm.cfg \
    --output training_lzh_seg_xposwarm/ --code scripts/seg_code.py --system.seed 0
run train_lzh_seg_sud.log configs/config_lzh_seg_sud.cfg --output training_lzh_seg_sud/ \
    --code scripts/seg_code.py --system.seed 0

# The graft: the tagger must sit BEHIND the morphologiser or package_sud.sh refuses the arm.
wait_slot
echo "=== $(date '+%F %T') grafting the xpos tagger" >> queue_runs.log
$PY scripts/graft_xpos_tagger.py training_lzh_seg_sud/model-best \
    training_lzh_seg_xposwarm/model-best training_lzh_seg_sud_xw \
    --corpus corpus_lzh_trad/lzh_kyoto-sud-test.relabeled_ext.udep_ruled.punct.rulemerged.spacy \
    >> queue_runs.log 2>&1
echo "=== $(date '+%F %T') graft finished ($?)" >> queue_runs.log
