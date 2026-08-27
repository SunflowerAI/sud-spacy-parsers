#!/usr/bin/env bash
# Train the v3 arms, SEQUENTIALLY. One seed per invocation; pass SEEDS to widen.
#
# ⚠ SEQUENTIAL ON PURPOSE. spaCy on CPU already saturates the cores through AppleOps, so two arms at
# once halve each other rather than finishing sooner, and a previous sweep in this repo died twice
# on a fork limit. There may also be an unrelated `spacy train` on this machine; check before
# assuming the wall clock below.
#
# ⚠ NEVER PIPE A TRAINING COMMAND TO `head`. SIGPIPE truncates the run at whatever checkpoint it had
# reached, and the tell is `model-best == model-last`. Output is redirected, and `python -u` is what
# makes that output appear before the process exits.
#
# `model-last`'s mtime is the reliable progress signal -- it is rewritten at EVERY eval, whereas
# `tail -f` on the log shows nothing until spaCy flushes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ARMS=${ARMS:-"g3_base g3_vec g3_vec_ctl g3_vec_shuf"}
SEEDS=${SEEDS:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

for seed in $SEEDS; do
  for arm in $ARMS; do
    out="training_v3_${arm}_s${seed}"
    log="train_v3_${arm}_s${seed}.log"
    if [ "$SKIP_EXISTING" = 1 ] && [ -f "$out/model-best/meta.json" ]; then
      echo "skip  $out (already has model-best)"; continue
    fi
    echo "=== $out  $(date -u +%H:%M:%SZ)"
    $PY -u -m spacy train "configs/config_${arm}.cfg" \
        --output "$out" --code scripts/generic_code_v3.py \
        --system.seed "$seed" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "  !! $arm seed $seed exited $rc -- see $log"; continue; fi
    echo "  done $(date -u +%H:%M:%SZ)  $(grep -cE '^ *[0-9]' "$log" || true) eval rows"
  done
done
echo "ALL DONE $(date -u +%H:%M:%SZ)"
