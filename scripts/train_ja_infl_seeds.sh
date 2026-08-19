#!/usr/bin/env bash
# Extra seeds for the Inflection experiment. Seed 0 is the run train_ja_infl.sh leaves behind.
# "Multi-seed or don't claim it" (NEGATIVE-RESULTS.md) -- a single run produced at least one wrong
# claim in this repo's history, and the margin here is under half a point.
set -u
PY=.venv/bin/python
CORPUS=corpus_ja_infl
for seed in "$@"; do
  for v in infl infl_ctl; do
    arm="training_ja_${v}_s${seed}"
    [ -d "$arm/model-best" ] && { echo "  $v s$seed exists -- skip"; continue; }
    $PY -u -m spacy train "configs/config_ja_$v.cfg" --code scripts/seg_code.py \
        --output "$arm/" --system.seed "$seed" \
        --paths.train "$CORPUS/train.spacy" --paths.dev "$CORPUS/dev.spacy" \
        > "train_ja_${v}_s${seed}.log" 2>&1
    if [ -d "$arm/model-best" ]; then
      $PY scripts/eval_ja_infl.py "$arm/model-best" "$CORPUS/test.spacy" \
          --label "seed $seed  $v" | grep -E 'seed|tag_acc'
    else
      echo "  $v s$seed FAILED"; tail -5 "train_ja_${v}_s${seed}.log"
    fi
  done
done
