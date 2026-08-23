#!/bin/bash
# Korean, with the morphemes the eojeol tokenisation hides handed to the parser as an input channel.
#
# THE DEFECT. `ko_sud_gsd` is tokenised by eojeol -- a stem with its particles and endings fused
# into one whitespace-delimited token -- so every stem-and-particle combination is a fresh string.
# 34.5 % of test tokens are eojeol the parser has never seen, and they parse 33.7 LAS below the
# rest (scripts/eval_ko_oov.py). It is not a vocabulary shortage: 72.3 % of those unseen eojeol have
# a FIRST MORPHEME that is in the training data already, sitting there unreachable behind a
# particle. `sud.KoAnalyserEmbed.v1` reaches it, by calling mecab-ko at runtime.
#
# THE ARM TO BEAT is the capacity control, not the plain baseline: `constant = true` builds the same
# columns and the same Maxout width and gives every token the sentinel, so the delta between them is
# the INFORMATION. Half of Latin's lemma-vector gain turned out to be the extra rows alone
# (docs/latin.md), which is why this is a phase and not an afterthought.
#
# SEEDS. ko trains on 56 687 tokens, a tenth of Latin's, and Latin's own seed spread is mean
# absolute 0.272 LAS with a maximum of 0.82 (docs/latin.md). A single-seed Korean delta is
# unreadable, so both arms are trained three times and `eval` reports every seed.
#
# THE ORDER ARM is the same config plus `scripts/ko_order.py`, so the analyser seeds are its
# control. Its phases are separate because its prior is much weaker (docs/korean.md): Korean's
# measured order-sensitivity is −2.7 LAS against Latin's −17.4, so what is left to win is
# regularisation on the smallest treebank in the set rather than robustness.
#
# Phases (run all, or name one): check | seeds | labels | order | eval | oov | scramble
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
export MECAB_PATH="${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}"

P=ko_gsd-sud
TRAIN=corpus_ko_eojeol/$P-train.relabeled_ext.spacy
DEV=corpus_ko_eojeol/$P-dev.relabeled_ext.spacy
TEST=corpus_ko_eojeol/$P-test.relabeled_ext.spacy
SEEDS="${SEEDS:-0 1 2}"
#: the released arm, for reference. Its parser is the one users have.
REL=training_ko_eojeol_lemma/model-best

phase=${1:-all}

do_check() {
  echo "### CHECK: the layer is what it claims to be"
  # Six assertions, including that no-channel is byte-identical to spacy.MultiHashEmbed.v2 and that
  # a backend mismatch REFUSES. A channel that silently reads nothing scores like its own control.
  $PY scripts/check_ko_embed.py
  # A permutation bug in HEAD does not raise: it yields a well-formed Example with a DIFFERENT tree.
  $PY scripts/check_ko_order.py "$TRAIN" --docs 300
}

do_labels() {
  echo "### LABELS: collected over augmented passes, not from the one pass spaCy would use"
  # `max_epochs = -1` makes spaCy initialise from the first hundred examples only, and a label
  # missing from that set does not raise — it teaches label 0 (CLAUDE.md hazard 9).
  [ -f scripts/ko_order_bigrams.json ] || $PY scripts/calibrate_ko_order.py \
      assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu \
      --out scripts/ko_order_bigrams.json
  $PY scripts/init_aug_labels.py configs/config_ko_order.cfg labels_ko_order $CODE --passes 4 \
      --paths.train "$TRAIN" --paths.dev "$DEV"
}

do_order() {
  for s in $SEEDS; do
    out=training_ko_order_s$s
    echo "### $out"
    # BOTH seeds move: leaving the augmenter's own seed at 0 would re-use one sequence of
    # linearisations and measure only the weight initialisation.
    $PY -u -m spacy train configs/config_ko_order.cfg $CODE --output "$out/" \
      --paths.train "$TRAIN" --paths.dev "$DEV" --system.seed "$s" \
      --corpora.train.augmenter.seed "$s" > "train_ko_order_s$s.log" 2>&1
    [ -d "$out/model-best" ] || { echo "!! FAILED"; tail -20 "train_ko_order_s$s.log"; exit 1; }
    grep -E '^[[:space:]]*[0-9]' "train_ko_order_s$s.log" | tail -1
  done
}

do_scramble() {
  echo "### SCRAMBLE: how much each arm rests on the order of pre-head siblings"
  args=""
  for s in $SEEDS; do
    for arm in analyser order; do
      [ -d training_ko_${arm}_s$s/model-best ] \
        && args="$args --model ${arm}_s$s=training_ko_${arm}_s$s/model-best"
    done
  done
  $PY scripts/eval_ko_scramble.py "$TEST" --model released=$REL $args
}

do_seeds() {
  for s in $SEEDS; do
    for arm in analyser analyser_ctl; do
      out=training_ko_${arm}_s$s
      cfg=configs/config_ko_${arm}.cfg
      echo "### $out"
      $PY -u -m spacy train "$cfg" $CODE --output "$out/" \
        --paths.train "$TRAIN" --paths.dev "$DEV" --system.seed "$s" \
        > "train_ko_${arm}_s$s.log" 2>&1
      [ -d "$out/model-best" ] || { echo "!! FAILED"; tail -20 "train_ko_${arm}_s$s.log"; exit 1; }
      grep -E '^[[:space:]]*[0-9]' "train_ko_${arm}_s$s.log" | tail -1
    done
  done
}

do_eval() {
  echo "### EVAL: test, --gold-preproc, every seed of both arms"
  printf "%-26s %7s %7s %7s\n" arm TAG UAS LAS
  for arm in eojeol_lemma $(for s in $SEEDS; do echo analyser_s$s analyser_ctl_s$s order_s$s; done); do
    d=training_ko_$arm/model-best
    [ -d "$d" ] || { printf "%-26s MISSING\n" "$arm"; continue; }
    printf "%-26s " "$arm"
    $PY -m spacy evaluate "$d" "$TEST" --gold-preproc $CODE \
        --output metrics/ko/metrics_ko_${arm}_gp.json 2>/dev/null \
      | grep -E '^(TAG|UAS|LAS) ' | awk '{printf "%7s ", $2}'
    echo
  done
}

do_oov() {
  echo "### OOV: the split the channel is judged on"
  args=""
  for s in $SEEDS; do
    [ -d training_ko_analyser_s$s/model-best ] \
      && args="$args --model analyser_s$s=training_ko_analyser_s$s/model-best"
    [ -d training_ko_analyser_ctl_s$s/model-best ] \
      && args="$args --model ctl_s$s=training_ko_analyser_ctl_s$s/model-best"
    [ -d training_ko_order_s$s/model-best ] \
      && args="$args --model order_s$s=training_ko_order_s$s/model-best"
  done
  $PY scripts/eval_ko_oov.py "$TEST" --model released=$REL $args --keys
}

case "$phase" in
  check)    do_check ;;
  seeds)    do_seeds ;;
  labels)   do_labels ;;
  order)    do_order ;;
  eval)     do_eval ;;
  oov)      do_oov ;;
  scramble) do_scramble ;;
  all)      do_check; do_seeds; do_labels; do_order; do_eval; do_oov; do_scramble ;;
  *) echo "unknown phase: $phase (check|seeds|labels|order|eval|oov|scramble)"; exit 1 ;;
esac
echo "DONE: $phase"
