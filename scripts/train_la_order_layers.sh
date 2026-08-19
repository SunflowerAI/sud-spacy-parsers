#!/bin/bash
# Retrain the Latin morphologiser and lemmatiser THROUGH THE WORD-ORDER AUGMENTER, then rebuild the
# combined arm on top of them.
#
# WHY. `training_la_order_lemvec` reads a frozen morphologiser and lemmatiser that have only ever
# seen natural word order, and measured on the scrambled test set those pipes degrade:
#
#     evaluated on   POS     MORPH   LEMMA
#     identity       92.20   74.70   88.45
#     order          91.11   72.01   87.91
#     order_hyper    90.84   71.43   87.78
#
# MORPH is the channel the parser most depends on and it loses 2.69. That is a candidate
# explanation for why the two changes came out SUB-ADDITIVE by ~1.0 LAS in both regimes:
#
#     identity   aug 71.32   +lemvec 73.13   +order 71.51   both 72.29  (additive would be 73.32)
#     order      aug 53.88   +lemvec 55.38   +order 63.13   both 63.64  (additive would be 64.63)
#
# The deficit being the SAME 1.0 in two very different regimes is what suggests one shared
# bottleneck rather than two independent costs, and a degraded input channel is the obvious one.
#
# The freeze recipe makes this cheap to attribute: each pipe carries its OWN HashEmbedCNN rather
# than a listener, so retraining them cannot touch the parser underneath -- verified with `cmp`.
#
# Phases (run all, or name one): morph | lemma | combined | eval
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
P=la_ittbproiel-sud
CODE="--code scripts/seg_code.py"
TRAIN=corpus_la_ext_macron/$P-train.relabeled_ext.macron.spacy
DEV=corpus_la_ext_union/dev

phase=${1:-all}

# EACH LAYER COLLECTS ITS OWN LABELS IMMEDIATELY BEFORE IT TRAINS, and that ordering is not
# cosmetic. A layer's config SOURCES the arm below it, so resolving config_la_order_lemma.cfg loads
# `training_la_order_morph/model-best` -- which does not exist until the morph phase has run. An
# earlier draft collected both label sets up front and died with E050 fifty-seven seconds in,
# leaving the lemma phase to fail much later on a labels file that was never written.
# train_la_aug.sh already interleaves them for exactly this reason.
init_labels() {  # $1=config  $2=outdir  $3=passes
  $PY scripts/init_aug_labels.py "$1" "$2" $CODE --passes "$3" \
      --paths.train "$TRAIN" --paths.dev "$DEV"
}

train_arm() {  # $1=suffix  $2=config
  echo "### $1"
  $PY -u -m spacy train "$2" $CODE --output training_la_$1/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_la_$1.log 2>&1
  [ -d training_la_$1/model-best ] || { echo "!! FAILED"; tail -20 train_la_$1.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_$1.log | tail -1
}

do_morph() {
  # 2 passes: the morphologiser's labels are FEATS bundles, a property of the TOKENS, which
  # neither augmenter touches.
  init_labels configs/config_la_order_morph.cfg labels_la_order_morph 2
  train_arm order_morph configs/config_la_order_morph.cfg
  cmp training_la_order/model-best/parser/model training_la_order_morph/model-best/parser/model \
    && echo "  parser byte-identical (freeze recipe holds)"
}

do_lemma() {
  # 10 passes: edit trees are a property of the FORMS, so the ORTHOGRAPHY half of the augmenter
  # keeps minting new ones (uītae / vitae / vītæ -> uita are three labels), and a missing one does
  # not raise -- it silently trains that token against label 0.
  init_labels configs/config_la_order_lemma.cfg labels_la_order_lemma 10
  train_arm order_lemma configs/config_la_order_lemma.cfg
  cmp training_la_order_morph/model-best/parser/model \
      training_la_order_lemma/model-best/parser/model \
    && echo "  parser byte-identical (freeze recipe holds)"
}

do_combined() {
  echo "### COMBINED v2: the same arm, reading order-augmented morph and lemma"
  $PY scripts/make_la_lemvec_config.py --base configs/config_la_order.cfg \
      --out configs/config_la_order_lemvec2.cfg --labels-dir labels_la_order \
      --source training_la_order_lemma/model-best
  # Assert the channels actually arrive before spending hours on them: a frozen pipe that is not
  # in `annotating_components` leaves the embed hashing `Case=` on every token and training
  # perfectly happily to its own capacity control's score.
  $PY scripts/check_la_lemvec_inputs.py configs/config_la_order_lemvec2.cfg --train "$TRAIN"
  train_arm order_lemvec2 configs/config_la_order_lemvec2.cfg
}

do_eval() {
  echo "### EVAL: does the retrained input layer recover the missing ~1.0 LAS?"
  echo "-- how much better are the retrained pipes on scrambled order?"
  for arm in aug_lemma order_lemma; do
    d=training_la_$arm/model-best; [ -d "$d" ] || continue
    echo "== $arm"
    for v in identity order order_hyper; do
      printf "   %-12s " "$v"
      $PY -m spacy evaluate "$d" corpus_la_orders/$v.spacy --gold-preproc $CODE 2>/dev/null \
        | grep -E '^(POS|MORPH|LEMMA) ' | tr -s ' \n' ' '; echo
    done
  done
  echo; echo "-- parsing, all four arms"
  for arm in order lemvec order_lemvec order_lemvec2; do
    d=training_la_$arm/model-best; [ -d "$d" ] || continue
    echo "== $arm"
    for v in identity order order_proj order_hyper order_free; do
      printf "   %-12s " "$v"
      $PY -m spacy evaluate "$d" corpus_la_orders/$v.spacy --gold-preproc $CODE 2>/dev/null \
        | grep -E '^(UAS|LAS) ' | tr -s ' \n' ' '; echo
    done
    for sl in all itp perseus; do
      case "$sl" in all) t=corpus_la_ext/$P-test.relabeled_ext.spacy ;;
                    *)   t=corpus_la_eval_slices/$sl-test.relabeled_ext.spacy ;; esac
      printf "   %-12s " "plain:$sl"
      $PY -m spacy evaluate "$d" "$t" --gold-preproc $CODE 2>/dev/null \
        | grep -E '^(UAS|LAS) ' | tr -s ' \n' ' '; echo
    done
  done
}

case "$phase" in
  morph)    do_morph ;;
  lemma)    do_lemma ;;
  combined) do_combined ;;
  eval)     do_eval ;;
  all)      do_morph; do_lemma; do_combined; do_eval ;;
  *) echo "unknown phase: $phase (morph|lemma|combined|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
