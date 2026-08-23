#!/bin/bash
# Telugu: the base parser, its morph/lemma layers, and the word-order augmenter.
#
# ⚠ THERE IS NO `lemvec` ARM HERE, AND THAT IS A PROPERTY OF THE TREEBANK, NOT A DECISION.
# SUD_Telugu-MTG carries no lemma column at all and 115 FEATS values in 6 465 tokens (92
# `NumType=Card`, 56 SUD `Shared`); its XPOS is a verbatim copy of UPOS, zero mismatches. The
# Latin/Sanskrit recipe's two parser input channels have nothing to read. `docs/dravidian.md`
# records the library survey — LTRC/anusAraka (2001, C+Perl, hardcoded build paths), `apertium-tel`
# (six noun roots), Indic NLP (unsupervised segmentation, no lemma), Stanza (trained on MTG itself,
# so its lemmatiser returns None) — which found nothing that could fill the columns.
#
# What Telugu DOES get is the augmentation half, and it is the better half for a corpus this small:
# 1 051 training sentences averaging 4.8 tokens, with 23 % OSV among the arguments of a verb, is
# exactly the case where a parser memorises the orders it happened to see.
#
# Phases (run all, or name one): prep | base | layers | order | eval
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
P=te_mtg-sud
TRAIN=corpus_te/$P-train.spacy
DEV=corpus_te/$P-dev.spacy

phase=${1:-all}

train_arm() {  # $1=suffix $2=config
  echo "### TRAIN training_te_${1}/"
  $PY -u -m spacy train "$2" $CODE --output "training_te_${1}/" \
    --paths.train "$TRAIN" --paths.dev "$DEV" > "train_te_${1}.log" 2>&1
  [ -d "training_te_${1}/model-best" ] || {
    echo "!! FAILED"; tail -20 "train_te_${1}.log"; exit 1; }
  grep -E '^[[:space:]]*[0-9]' "train_te_${1}.log" | tail -1
}

do_prep() {
  echo "### PREP: stage MTG, fall the empty lemma column back to IDENTITY, convert"
  $PY scripts/prep_te.py
  rm -rf corpus_te && mkdir -p corpus_te
  for f in assets_te/te_mtg-sud-*.conllu; do
    $PY -m spacy convert "$f" corpus_te/ --converter conllu -n 10 >/dev/null 2>&1
  done
  ls corpus_te/
}

do_base()   { train_arm seg configs/config_te_seg.cfg; }
do_layers() { bash scripts/train_morph.sh te; bash scripts/train_lemma.sh te; }

do_order() {
  echo "### ORDER: word-order augmentation"
  # p_hyperbaton is 0 for Telugu and that is measured, not conservative: ONE of 1 051 training
  # sentences carries a crossing arc, so generating displacement would be inventing a construction
  # the language does not have. The check asserts head-finality comes out unchanged.
  $PY scripts/check_dravidian_order.py "$TRAIN" --lang te
  $PY scripts/make_dravidian_order_config.py configs/config_te_seg.cfg --lang te \
      --out configs/config_te_order.cfg --labels-dir labels_te_order
  $PY scripts/init_aug_labels.py configs/config_te_order.cfg labels_te_order $CODE --passes 6 \
      --paths.train "$TRAIN" --paths.dev "$DEV"
  train_arm order configs/config_te_order.cfg
}

do_eval() {
  echo "### EVAL (gold-preproc)"
  for kind in seg order; do
    d="training_te_${kind}/model-best"
    [ -d "$d" ] || { echo "== $kind: MISSING -- skip"; continue; }
    printf "%-8s " "$kind"
    $PY -m spacy evaluate "$d" "corpus_te/$P-test.spacy" --gold-preproc $CODE \
        --output "metrics/te/metrics_te_${kind}.json" 2>/dev/null \
      | grep -E '^(TAG|UAS|LAS|POS|MORPH|LEMMA) ' | tr -s ' \n' ' '
    echo
  done
}

case "$phase" in
  prep)   do_prep ;;
  base)   do_base ;;
  layers) do_layers ;;
  order)  do_order ;;
  eval)   do_eval ;;
  all)    do_prep; do_base; do_layers; do_order; do_eval ;;
  *) echo "unknown phase: $phase (prep|base|layers|order|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
