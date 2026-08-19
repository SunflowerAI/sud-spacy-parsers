#!/usr/bin/env bash
# Does a WIDER RECEPTIVE FIELD help the segmenter generalise? A proxy for "would an RNN help?"
#
# The shipped segmenter is a depth-6 CNN over characters, so it sees +/-6 characters. The only thing
# a recurrent model offers over it is unbounded context. Rather than build one, widen the window:
# depth 12 sees +/-12. If that does not move HELD-OUT recall on the jackknife, context length is not
# the binding constraint and an RNN will not fix it either -- the failure is memorisation
# (84.3% -> 4.6% when types are held out), not a short window.
#
# Controls are the depth-6 jackknife arms already trained by train_seg_phon.sh, so only the new
# depth arms run here. Three seeds, because held-out recall on this slice swings 28/30/45 between
# control seeds on ~30 recovered tokens.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=data_seg_lzh_jk
for seed in 0 1 2; do
  out=models/lzh_seg_jk_deep_s${seed}
  [ -d "$out" ] && { echo "=== deep_s${seed}: exists, skipping"; continue; }
  echo "=== training deep_s${seed} (depth 12) ==="
  $PY -u scripts/train_samhita.py "$D/train.jsonl" "$D/dev.jsonl" "$out" \
      --width 64 --depth 12 --epochs 30 --seed "$seed" \
      > "train_lzh_seg_jk_deep_s${seed}.log" 2>&1 \
    || { echo "  FAILED"; tail -6 "train_lzh_seg_jk_deep_s${seed}.log"; continue; }
  tail -2 "train_lzh_seg_jk_deep_s${seed}.log"
done
echo "SEG DEPTH SWEEP DONE"
