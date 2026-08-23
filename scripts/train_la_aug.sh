#!/bin/bash
# Latin, orthographically augmented: ONE copy of the treebank, resampled into a new edition style
# every epoch, replacing the plain+macron UNION the released arm trains on.
#
# The released la arm (training_la_seg, via retrain_seg.sh) is trained on corpus_la_ext_union --
# two literal copies of the same 586 604 tokens, one plain and one macronised. That buys exactly
# those two spellings. This trains the same architecture on the MACRONISED copy alone, with
# scripts/la_augment.py rewriting each document per epoch into a sampled orthography: macrons kept
# or dropped per word, a breve on a random short vowel, j/v vs i/u, æ/œ vs ae/oe, and the sentence
# opening capitalised or not. See scripts/la_orth.py for what licenses each transform.
#
# Everything downstream is unchanged in kind: the morph and lemma layers stack on the augmented
# base by the usual freeze recipe, with the same augmenter on their own corpora.
#
# Phases (run all, or name one): variants | labels | base | morph | lemma | eval
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
MACRON_DEV=corpus_la_ext_macron/$P-dev.relabeled_ext.macron.spacy
# Dev stays the UNION, unaugmented: checkpoint selection has to be the same question the baseline
# arm was asked, or the two model-bests are not comparable.
DEV=corpus_la_ext_union/dev

# The styles the test set is rendered in. `plain` and `macron` are the two the union arm was
# trained for; the rest are the ones only augmentation can reach.
VARIANTS="plain macron mixed breve v vj lig caps lower all"

phase=${1:-all}

do_variants() {
  echo "### VARIANTS: render the test set in each orthography"
  local src=$A/$P-test.relabeled_ext.macron.conllu
  $PY scripts/make_la_variant_conllu.py "$src" --check
  rm -rf $A/variants corpus_la_variants && mkdir -p $A/variants corpus_la_variants
  for v in $VARIANTS; do
    $PY scripts/make_la_variant_conllu.py "$src" "$A/variants/test.$v.conllu" --style "$v"
    $PY -m spacy convert "$A/variants/test.$v.conllu" corpus_la_variants/ --converter conllu -n 10 \
      >/dev/null 2>&1
  done
  ls corpus_la_variants/
}

make_cfg() {  # $1=base config  $2=out config  $3=labels dir  [$4=retarget]
  $PY scripts/make_la_aug_config.py "$1" --out "$2" --labels-dir "$3" \
    ${4:+--retarget "$4"}
}

init_labels() {  # $1=config  $2=labels dir  $3=train  $4=passes
  $PY scripts/init_aug_labels.py "$1" "$2" $CODE --passes "$4" \
    --paths.train "$3" --paths.dev "$DEV"
}

do_labels() {
  # The glide lexicon is gitignored (treebank-derived, rebuilt in seconds), so a fresh clone has
  # none and the u/v axis would be silently inert. Build it if it is not there.
  [ -f scripts/la_glide_lut.json.gz ] || $PY scripts/build_la_glide_lut.py

  # Streaming (`max_epochs = -1`) initialises from the first 100 examples only, so every trained
  # component needs its labels handed to it. The lemmatiser's labels are EDIT TREES, which depend
  # on the word forms, so its set is collected over several augmented passes -- and a tree that is
  # missing does not raise, it silently trains against label 0. See scripts/init_aug_labels.py.
  echo "### LABELS: collect label sets over augmented passes"
  make_cfg configs/config_la_seg.cfg   configs/config_la_aug.cfg        labels_la_aug
  make_cfg configs/config_la_morph.cfg configs/config_la_aug_morph.cfg  labels_la_aug_morph \
           training_la_seg=training_la_aug
  make_cfg configs/config_la_lemma.cfg configs/config_la_aug_lemma.cfg  labels_la_aug_lemma \
           training_la_morph=training_la_aug_morph
  init_labels configs/config_la_aug.cfg labels_la_aug "$MACRON_TRAIN" 2
}

do_base() {
  echo "### BASE: train training_la_aug/ (augmented, one copy)"
  $PY -m spacy train configs/config_la_aug.cfg $CODE --output training_la_aug/ \
    --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" > train_la_aug.log 2>&1
  [ -d training_la_aug/model-best ] || { echo "!! FAILED"; tail -15 train_la_aug.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_aug.log | tail -1
}

do_morph() {
  echo "### MORPH: morphologizer on the augmented base"
  init_labels configs/config_la_aug_morph.cfg labels_la_aug_morph "$MACRON_TRAIN" 2
  $PY -m spacy train configs/config_la_aug_morph.cfg $CODE --output training_la_aug_morph/ \
    --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" > train_la_aug_morph.log 2>&1
  [ -d training_la_aug_morph/model-best ] || { echo "!! FAILED"; tail -15 train_la_aug_morph.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_aug_morph.log | tail -1
  cmp training_la_aug/model-best/parser/model training_la_aug_morph/model-best/parser/model \
    && echo "  parser byte-identical (freeze recipe holds)"
}

do_lemma() {
  echo "### LEMMA: edit-tree lemmatiser on the augmented morph arm"
  # 10 passes, not the 2 the other layers need: edit trees are FORM-derived, so the set keeps
  # growing with style draws (union 18 512 trees -> +1 pass 20 132 -> +5 22 498 -> +10 26 029).
  # Sub-linear, so this is not an explosion, but it has not converged either -- and every tree
  # that is missing costs a token silently, so buy the coverage; it is only a few minutes.
  init_labels configs/config_la_aug_lemma.cfg labels_la_aug_lemma "$MACRON_TRAIN" 10
  $PY -m spacy train configs/config_la_aug_lemma.cfg $CODE --output training_la_aug_lemma/ \
    --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" > train_la_aug_lemma.log 2>&1
  [ -d training_la_aug_lemma/model-best ] || { echo "!! FAILED"; tail -15 train_la_aug_lemma.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_aug_lemma.log | tail -1
  cmp training_la_aug_morph/model-best/parser/model training_la_aug_lemma/model-best/parser/model \
    && echo "  parser byte-identical (freeze recipe holds)"
}

do_eval() {
  echo "### EVAL: union arm vs augmented arm, across every variant"
  local base_aug=training_la_aug/model-best
  local base_union=training_la_seg/model-best
  [ -d training_la_aug_lemma/model-best ] && base_aug=training_la_aug_lemma/model-best
  [ -d training_la_lemma/model-best ] && base_union=training_la_lemma/model-best
  $PY scripts/eval_la_variants.py --model "union=$base_union" --model "aug=$base_aug" \
    --corpus-dir corpus_la_variants --out metrics/la/metrics_la_variants.json \
    --metrics LAS,UAS,TAG,LEMMA | tee eval_la_variants.log
}

case "$phase" in
  variants) do_variants ;;
  labels)   do_labels ;;
  base)     do_base ;;
  morph)    do_morph ;;
  lemma)    do_lemma ;;
  eval)     do_eval ;;
  all)      do_variants; do_labels; do_base; do_morph; do_lemma; do_eval ;;
  *) echo "unknown phase: $phase (variants|labels|base|morph|lemma|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
