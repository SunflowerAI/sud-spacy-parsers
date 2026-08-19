#!/usr/bin/env bash
# Phase B: the jackknifed segmenter and its matched control.
#
# The control is retrained here rather than reusing models/lzh_seg_char, so that the only difference
# between the two arms is the held-out types -- not a seed, a width or a version of the trainer.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ARGS="--width 64 --depth 6 --epochs 30"
for arm in ctl jk; do
  case $arm in
    ctl) D=data_seg_lzh    ;;
    jk)  D=data_seg_lzh_jk ;;
  esac
  out=models/lzh_seg_char_$arm
  [ -d "$out" ] && { echo "=== $arm: exists, skipping"; continue; }
  echo "=== training $arm from $D ==="
  $PY -u scripts/train_samhita.py "$D/train.jsonl" "$D/dev.jsonl" "$out" $ARGS \
      > "train_lzh_seg_$arm.log" 2>&1 || { echo "  FAILED"; tail -5 "train_lzh_seg_$arm.log"; continue; }
  tail -3 "train_lzh_seg_$arm.log"
done
echo "SEG JACKKNIFE TRAINING DONE"
