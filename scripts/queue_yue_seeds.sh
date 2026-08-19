#!/usr/bin/env bash
# yue's rebuilt base landed 2.5 LAS below the released arm. yue is a TEST-ONLY treebank cut 80/10/10
# (11 158 train tokens), so seed variance is the first hypothesis to rule out before concluding the
# relabel change caused it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
E=corpus_yue_ext/yue_hk-sud
wait_slot() { while :; do n=$(pgrep -f "spacy train" | wc -l | tr -d ' '); [ "$n" -le "${MAX_TRAINS:-1}" ] && return; sleep 60; done; }
for s in 1 2 3; do
  wait_slot
  echo "=== $(date '+%F %T') yue seg seed $s" >> queue_runs.log
  $PY -u -m spacy train configs/config_yue_seg.cfg --output training_yue_seg_s$s/ \
      --code scripts/seg_code.py --system.seed $s \
      --paths.train $E-train.relabeled_ext.spacy --paths.dev $E-dev.relabeled_ext.spacy \
      > train_yue_seg_s$s.log 2>&1
done
echo "=== $(date '+%F %T') yue seed sweep done" >> queue_runs.log
