#!/bin/bash
# Latin, orthographically AND syntactically augmented: the released `train_la_aug.sh` recipe with a
# fresh word order sampled per sentence on top of the fresh edition style sampled per document.
#
# The released la arm (training_la_aug) resamples the ORTHOGRAPHY every epoch, which took the LAS
# spread across editions from 54.4 to 7.0. It says nothing about ORDER: the three treebanks have
# strong and mutually different positional habits (ITTB scholastic verb-final prose, PROIEL
# narrative, Perseus classical verse), and the parser is free to learn any of them as if it were a
# fact about Latin. scripts/la_order.py re-linearises the tree instead -- same heads, same deprels,
# same lemmas, only the string moves -- honouring the constraints Latin actually has: Wackernagel
# particles stay second in their clause, `-que` stays on its host, prepositions and subordinators
# stay in front of what they govern, and 37.75 % of the sentences keep a crossing arc because that
# is what the corpus has (scripts/calibrate_la_order.py sets the displacement rate against it).
#
# The arm to beat is training_la_aug, NOT training_la_seg: this changes exactly one thing about the
# released recipe.
#
# Phases (run all, or name one): check | variants | labels | base | seeds | eval
#
# SEEDS. `base` is one run. The natural-order deltas this arm is judged on are small (+0.13 LAS
# combined, +1.16 on Perseus) and single-seed spaCy runs move by more than that, so `seeds` trains
# two more and `eval` reports all three. The scrambled-order gain (+9) needs no such help. Both the
# system seed AND the augmenter's own seed are moved: leaving the augmenter at 0 would re-use the
# identical sequence of linearisations and measure only the weight initialisation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
P=la_ittbproiel-sud
A=assets_la
CODE="--code scripts/seg_code.py"

MACRON_TRAIN=corpus_la_ext_macron/$P-train.relabeled_ext.macron.spacy
# Dev stays the UNION, unaugmented, for the same reason train_la_aug.sh gives: checkpoint selection
# has to be the same question the arm being compared against was asked.
DEV=corpus_la_ext_union/dev
TEST_SRC=$A/$P-test.relabeled_ext.macron.conllu

# Word-order test renderings. `identity` is the control (same file, same order) and must score
# exactly what an ordinary test run scores, which is what makes the rest of the column meaningful.
ORDERS="identity order order_proj order_hyper order_free"
#: arms the eval walks, in the order it reports them
ARMS="${ARMS:-aug order order_s1 order_s2}"

phase=${1:-all}

do_check() {
  echo "### CHECK: the augmenter moves the string and nothing else"
  # A permutation bug in HEAD does not raise -- it yields a well-formed Example with a DIFFERENT
  # tree, trains happily, and shows up nowhere in the log. So it is asserted, not assumed.
  $PY scripts/make_la_scrambled_conllu.py "$TEST_SRC" --check
  $PY scripts/check_la_order.py "$MACRON_TRAIN" --docs 500
}

do_variants() {
  echo "### VARIANTS: render the test set in each word order"
  rm -rf $A/orders corpus_la_orders && mkdir -p $A/orders corpus_la_orders
  for v in $ORDERS; do
    $PY scripts/make_la_scrambled_conllu.py "$TEST_SRC" "$A/orders/test.$v.conllu" --style "$v"
    $PY -m spacy convert "$A/orders/test.$v.conllu" corpus_la_orders/ --converter conllu -n 10 \
      >/dev/null 2>&1
    mv corpus_la_orders/test.$v.spacy corpus_la_orders/$v.spacy 2>/dev/null || true
  done
  ls corpus_la_orders/
}

do_labels() {
  [ -f scripts/la_glide_lut.json.gz ] || $PY scripts/build_la_glide_lut.py
  echo "### LABELS: collect label sets over augmented passes"
  $PY scripts/make_la_aug_config.py configs/config_la_seg.cfg \
      --out configs/config_la_order.cfg --labels-dir labels_la_order --order
  # SIX passes, not the two train_la_aug.sh uses for its base. Under orthographic augmentation
  # alone the parser's labels are properties of the TREES and one pass would do; word order makes
  # them properties of the ORDER, because a non-projective gold tree is pseudo-projectivised and
  # picks up a `||` suffix naming the lifted arc. init_aug_labels.py now prints the parser's
  # coverage on a fresh pass for exactly this reason -- read it.
  $PY scripts/init_aug_labels.py configs/config_la_order.cfg labels_la_order $CODE --passes 6 \
      --paths.train "$MACRON_TRAIN" --paths.dev "$DEV"
}

do_base() {
  echo "### BASE: train training_la_order/ (orthography + word order, one copy)"
  $PY -u -m spacy train configs/config_la_order.cfg $CODE --output training_la_order/ \
    --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" > train_la_order.log 2>&1
  [ -d training_la_order/model-best ] || { echo "!! FAILED"; tail -20 train_la_order.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_order.log | tail -1
}

do_seeds() {
  for s in 1 2; do
    echo "### SEED $s: training_la_order_s$s/"
    $PY -u -m spacy train configs/config_la_order.cfg $CODE --output training_la_order_s$s/ \
      --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" \
      --system.seed $s --corpora.train.augmenter.seed $s > train_la_order_s$s.log 2>&1
    [ -d training_la_order_s$s/model-best ] \
      || { echo "!! FAILED"; tail -20 train_la_order_s$s.log; exit 1; }
    grep -E '^[[:space:]]*[0-9]' train_la_order_s$s.log | tail -1
  done
}

do_eval() {
  echo "### EVAL: training_la_aug vs training_la_order, across every word order"
  for arm in $ARMS; do
    d=training_la_$arm/model-best
    [ -d "$d" ] || { echo "== $arm: MISSING ($d) -- skip"; continue; }
    echo "== $arm"
    for v in $ORDERS; do
      t=corpus_la_orders/$v.spacy
      [ -f "$t" ] || { printf "   %-12s MISSING\n" "$v"; continue; }
      printf "   %-12s " "$v"
      $PY -m spacy evaluate "$d" "$t" --gold-preproc $CODE \
          --output metrics/la/metrics_la_${arm}_${v}.json 2>/dev/null \
        | grep -E '^(TAG|UAS|LAS) ' | tr -s ' \n' ' '
      echo
    done
  done
}

case "$phase" in
  check)    do_check ;;
  variants) do_variants ;;
  labels)   do_labels ;;
  base)     do_base ;;
  seeds)    do_seeds ;;
  eval)     do_eval ;;
  all)      do_check; do_variants; do_labels; do_base; do_seeds; do_eval ;;
  *) echo "unknown phase: $phase (check|variants|labels|base|seeds|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
