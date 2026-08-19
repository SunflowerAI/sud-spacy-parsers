#!/usr/bin/env bash
# Retrain the whole lzh chain on the SUBTYPE-PRESERVING relabel (mod@tmod / comp:obl@lmod), on the
# sentence-segmenting base. Same order as queue_lzh_stack.sh: base -> morph -> tagger -> SUD -> graft.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX=${MAX_TRAINS:-1}
wait_slot() { while :; do n=$(pgrep -f "spacy train" | wc -l | tr -d ' '); [ "$n" -le "$MAX" ] && return; sleep 120; done; }
run() { local log=$1; shift; wait_slot
  echo "=== $(date '+%F %T') starting: $*" >> queue_runs.log
  $PY -u -m spacy train "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') finished ($?): $*" >> queue_runs.log; sleep 30; }

run train_lzh_seg.log configs/config_lzh_seg.cfg --output training_lzh_seg/ \
    --code scripts/seg_code.py --system.seed 0
run train_lzh_seg_morph.log configs/config_lzh_seg_morph.cfg --output training_lzh_seg_morph/ \
    --code scripts/seg_code.py --system.seed 0
run train_lzh_seg_xposwarm.log configs/config_lzh_seg_xposwarm.cfg \
    --output training_lzh_seg_xposwarm/ --code scripts/seg_code.py --system.seed 0
run train_lzh_seg_sud.log configs/config_lzh_seg_sud.cfg --output training_lzh_seg_sud/ \
    --code scripts/seg_code.py --system.seed 0
wait_slot
echo "=== $(date '+%F %T') grafting" >> queue_runs.log
$PY scripts/graft_xpos_tagger.py training_lzh_seg_sud/model-best \
    training_lzh_seg_xposwarm/model-best training_lzh_seg_sud_xw \
    --corpus corpus_lzh_trad/lzh_kyoto-sud-test.relabeled_ext.udep_ruled.punct.rulemerged.spacy \
    >> queue_runs.log 2>&1
echo "=== $(date '+%F %T') graft done ($?)" >> queue_runs.log
