#!/usr/bin/env bash
# The sa morph-first arm retrained on the SUBTYPE-PRESERVING relabel (8 249 tokens: mod -> mod@instr
# / comp:obl -> comp:obl@goal etc.), THREE SEEDS.
#
# ⚠ THREE SEEDS BECAUSE ONE IS NOT A MEASUREMENT. The yue rebuild in this same batch landed 2.5 LAS
# below its released arm on a single seed and looked like a regression; four seeds spanned 6.89 LAS
# (57.25 - 64.14) and the "regression" was the draw, not the change.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
N=corpus_sa_mwt_rl2_norm_new
wait_slot() { while :; do n=$(pgrep -f "spacy train" | wc -l | tr -d ' '); [ "$n" -le "${MAX_TRAINS:-1}" ] && return; sleep 60; done; }
for s in 1 0 2; do
  wait_slot
  echo "=== $(date '+%F %T') sa mp2 subtype seed $s" >> queue_runs.log
  $PY -u -m spacy train configs/config_sa_morphparse2.cfg --output training_sa_mp2_sub_s$s/ \
      --code scripts/seg_code.py --system.seed $s \
      --paths.train $N/train.csl_mwt.spacy \
      --paths.dev $N/sa_vedic-sud-dev.relabeled_ext.csl_mwt.spacy \
      > train_sa_mp2_sub_s$s.log 2>&1
done
echo "=== $(date '+%F %T') sa subtype sweep done" >> queue_runs.log
