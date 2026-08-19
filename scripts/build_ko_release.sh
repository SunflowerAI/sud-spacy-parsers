#!/bin/bash
# The ko release chain for the analyser-channel arm: seg base -> morphologiser -> lemmatiser,
# then the wheel and the checks that a wheel has to pass.
#
# ⚠ WHY THIS IS NOT `train_ko_analyser.sh seeds` WITH A DIFFERENT NAME. That driver trains the
# MEASUREMENT arm, on `config_ko_eojeol.cfg`'s single-sentence reader, which is what makes the
# channel's contribution single-variable. It cannot ship: fed two sentences it returns ONE, with a
# single self-headed root, because it never sees an example with a boundary in it. Its
# `--gold-preproc` SENT F of 99.70 says nothing — every example there is already one sentence
# (CLAUDE.md hazard 4). `seg` is a BASE recipe, not a stackable layer, so the release arm is the
# same one-block change applied to `config_ko_eojeol_seg.cfg` and trained from scratch.
#
# Phases (run all, or name one): base | seeds | pick | stack | wheel | verify | raw
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
export MECAB_PATH="${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}"

P=ko_gsd-sud
TRAIN=corpus_ko_eojeol/$P-train.relabeled_ext.spacy
DEV=corpus_ko_eojeol/$P-dev.relabeled_ext.spacy
TEST=corpus_ko_eojeol/$P-test.relabeled_ext.spacy
SEEDS="${SEEDS:-0 1 2}"
#: set by `pick`, overridable. The arm the stack and the wheel are built on.
PICK_FILE=.ko_release_pick

do_seeds() {
  for s in $SEEDS; do
    out=training_ko_anseg_s$s
    echo "### $out"
    $PY -u -m spacy train configs/config_ko_analyser_seg.cfg $CODE --output "$out/" \
      --paths.train "$TRAIN" --paths.dev "$DEV" --system.seed "$s" \
      > "train_ko_anseg_s$s.log" 2>&1
    [ -d "$out/model-best" ] || { echo "!! FAILED"; tail -20 "train_ko_anseg_s$s.log"; exit 1; }
    grep -E '^[[:space:]]*[0-9]' "train_ko_anseg_s$s.log" | tail -1
  done
}

do_pick() {
  # ⚠ SELECT ON DEV. The test set is reported, never chosen on; spaCy's own `model-best` is already
  # a dev choice, and picking the seed on test would be a second, hidden one.
  echo "### PICK: best dev SCORE across seeds"
  best=""; bestv=0
  for s in $SEEDS; do
    v=$(grep -E '^[[:space:]]*[0-9]' "train_ko_anseg_s$s.log" | awk '{print $NF}' | sort -g | tail -1)
    las=$(grep -E '^[[:space:]]*[0-9]' "train_ko_anseg_s$s.log" | awk '{print $8}' | sort -g | tail -1)
    printf "   seed %s  best dev SCORE %s  best dev LAS %s\n" "$s" "$v" "$las"
    if awk "BEGIN{exit !($v > $bestv)}"; then bestv=$v; best=$s; fi
  done
  echo "training_ko_anseg_s$best/model-best" > $PICK_FILE
  echo "   -> $(cat $PICK_FILE)"
}

do_stack() {
  base=$(cat $PICK_FILE)
  echo "### STACK: morphologiser then lemmatiser, by the freeze recipe, on $base"
  # The freeze recipe: source the arm's components, FREEZE them, train only the new one with its own
  # small HashEmbedCNN. Frozen components must come out byte-identical, which `verify` asserts.
  $PY scripts/make_ko_stack_configs.py "$base"
  $PY -u -m spacy train configs/config_ko_anseg_morph.cfg $CODE --output training_ko_anseg_morph/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_anseg_morph.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_anseg_morph.log | tail -1
  $PY -u -m spacy train configs/config_ko_anseg_lemma.cfg $CODE --output training_ko_anseg_lemma/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_anseg_lemma.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_anseg_lemma.log | tail -1
}

do_wheel() {
  echo "### WHEEL: package training_ko_anseg_lemma as ko_sud_gsd $VERSION"
  # KO_BASE names the arm; the --code list in package_sud.sh already carries sud_ko_embed.py and
  # ko_analyser.py, and pkg() refuses to build without them.
  KO_BASE=training_ko_anseg_lemma/model-best VERSION="${VERSION:-0.3.0}" \
    bash scripts/package_sud.sh ko
}

do_verify() {
  echo "### VERIFY: the frozen layers, the segmentation, then the INSTALLED wheel"
  bash scripts/verify_ko_release.sh
}

do_raw() {
  echo "### RAW: end-to-end, the model finding its own sentences"
  $PY -m spacy evaluate training_ko_anseg_lemma/model-best "$TEST" $CODE \
      --output metrics_ko_anseg_raw.json | grep -E '^(TOK|TAG|POS|MORPH|LEMMA|UAS|LAS|SENT)'
  echo "--- and with gold sentences, for comparison with everything else in docs/korean.md"
  $PY -m spacy evaluate training_ko_anseg_lemma/model-best "$TEST" --gold-preproc $CODE \
      --output metrics_ko_anseg_gp.json | grep -E '^(TAG|UAS|LAS|SENT)'
}

phase=${1:-all}
case "$phase" in
  seeds)  do_seeds ;;
  pick)   do_pick ;;
  stack)  do_stack ;;
  wheel)  do_wheel ;;
  verify) do_verify ;;
  raw)    do_raw ;;
  all)    do_seeds; do_pick; do_stack; do_raw; do_wheel; do_verify ;;
  *) echo "unknown phase: $phase (seeds|pick|stack|wheel|verify|raw)"; exit 1 ;;
esac
echo "DONE: $phase"
