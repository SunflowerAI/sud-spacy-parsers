#!/usr/bin/env bash
# Rebuild the yue chain on the BEST-OF-FOUR seg base (seed 1, dev LAS 64.14; seeds ran 61.58 /
# 62.81 / 63.17 / 64.14 — a 2.56-point spread on an 11 158-token treebank, which is why one seed is
# not a measurement here). Then the xpos graft the released arm ships.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
E=corpus_yue_ext/yue_hk-sud
S=corpus_yue_sud
wait_slot() { while :; do n=$(pgrep -f "spacy train" | wc -l | tr -d ' '); [ "$n" -le "${MAX_TRAINS:-1}" ] && return; sleep 60; done; }
run() { local log=$1; shift; wait_slot
  echo "=== $(date '+%F %T') starting: $*" >> queue_runs.log
  $PY -u -m spacy train "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') finished ($?): $*" >> queue_runs.log; sleep 20; }

run train_yue_morph.log  configs/config_yue_morph.cfg  --output training_yue_morph/  --code scripts/seg_code.py \
    --paths.train $E-train.relabeled_ext.spacy --paths.dev $E-dev.relabeled_ext.spacy
run train_yue_lemma.log  configs/config_yue_lemma.cfg  --output training_yue_lemma/  --code scripts/seg_code.py \
    --paths.train $E-train.relabeled_ext.spacy --paths.dev $E-dev.relabeled_ext.spacy
run train_yue_xposwarm.log configs/config_yue_xposwarm.cfg --output training_yue_xposwarm/ --code scripts/seg_code.py \
    --paths.train $E-train.relabeled_ext.spacy --paths.dev $E-dev.relabeled_ext.spacy
run train_yue_sud.log    configs/config_yue_sud.cfg    --output training_yue_sud/    --code scripts/seg_code.py \
    --paths.train $S/train.spacy --paths.dev $S/dev.spacy
wait_slot
echo "=== $(date '+%F %T') yue graft" >> queue_runs.log
$PY scripts/graft_xpos_tagger.py training_yue_sud/model-best training_yue_xposwarm/model-best \
    training_yue_sud_xpos --corpus $E-test.relabeled_ext.spacy >> queue_runs.log 2>&1
echo "=== $(date '+%F %T') yue graft done ($?)" >> queue_runs.log
