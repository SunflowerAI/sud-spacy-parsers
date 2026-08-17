#!/usr/bin/env bash
# Phase C: does a sub-character phonological/graphic channel help the segmenter generalise?
#
# THE COMPARISON. Both arms train on the JACKKNIFED data (data_seg_lzh_jk), where 158 multi-char
# types are split apart in train+dev and left merged in the untouched test split. So the question is
# measured on 611 held-out multi-char tokens rather than the 16 the hand gold offers, and the
# retained 1,093 act as a memorisation control that should barely move.
#
# THREE SEEDS EACH, because this repo has been burned by single-seed reads more than once (the
# kanripo vectors read +0.46 on seed 0 and +0.04 over three).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=data_seg_lzh_jk
ARGS="--width 64 --depth 6 --epochs 30"

# Wait for the seed-0 control to FINISH, not to appear. ⚠ train_samhita.py writes the output
# directory at epoch 0 and rewrites it on every improvement, so `[ -d ... ]` is true almost
# immediately -- an earlier version of this guard copied a 2-epoch model as the control, which
# would have handicapped the control and flattered the treatment with nothing in the logs to say
# so. Wait for the driver's completion line instead.
for _ in $(seq 1 480); do grep -q "SEG JACKKNIFE TRAINING DONE" seg_jk_sweep.log 2>/dev/null && break; sleep 15; done
grep -q "SEG JACKKNIFE TRAINING DONE" seg_jk_sweep.log 2>/dev/null \
  || { echo "jk control never finished; aborting"; exit 1; }

for seed in 0 1 2; do
  for arm in ctl phon; do
    case $arm in
      ctl)  AUX="" ;;
      phon) AUX="--aux radical,qieyun" ;;
    esac
    out=models/lzh_seg_jk_${arm}_s${seed}
    # seed 0 of the control is already trained under another name; reuse it rather than repeat it
    if [ "$arm" = ctl ] && [ "$seed" = 0 ] && [ -d models/lzh_seg_char_jk ]; then
      [ -d "$out" ] || cp -R models/lzh_seg_char_jk "$out"
      echo "=== jk_${arm}_s${seed}: reused from models/lzh_seg_char_jk"; continue
    fi
    [ -d "$out" ] && { echo "=== jk_${arm}_s${seed}: exists, skipping"; continue; }
    echo "=== training jk_${arm}_s${seed} ==="
    $PY -u scripts/train_samhita.py "$D/train.jsonl" "$D/dev.jsonl" "$out" $ARGS \
        --seed "$seed" $AUX > "train_lzh_seg_jk_${arm}_s${seed}.log" 2>&1 \
      || { echo "  FAILED"; tail -6 "train_lzh_seg_jk_${arm}_s${seed}.log"; continue; }
    tail -2 "train_lzh_seg_jk_${arm}_s${seed}.log"
  done
done
echo "SEG PHON SWEEP DONE"
