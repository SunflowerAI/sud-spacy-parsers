#!/bin/bash
# Tamil: the base parsers, then the Latin/Sanskrit recipe on top — parse off LEMMA and DECOMPOSED
# MORPHOLOGY rather than surface forms alone, with a tight capacity control beside it.
#
# TWO BASE ARMS, and both are kept on purpose. SUD_Tamil-TTB ships train/dev/test (400 sentences,
# 6 329 words); SUD_Tamil-MWTT ships TEST ONLY (534 sentences, 2 584 words) and is carved 80/10/10
# by scripts/prep_ta.py, the way Cantonese-HK is. Folding MWTT in roughly doubles the training data
# — but the two treebanks DISAGREE about annotation, not merely about tagset: MWTT writes
# `mod@poss` where TTB writes plain `mod` for the same genitive, `subj@nc` where TTB writes `subj`,
# and subtyped `udep@tmod`/`@lmod`/`@inst` where TTB writes bare `udep`. No map fixes that without
# deciding which treebank is right. So `ttb` is the control and `both` is the candidate, and the
# eval reports the TTB test slice separately — the same way docs/latin.md prices Perseus.
#
# Phases (run all, or name one): prep | base | layers | lemvec | order | eval
#
# ⚠ THE SIZE DEVIATIONS ARE IN scripts/make_dravidian_config.py, NOT HERE: `min_action_freq = 1`
# (the default 30 deletes 7 of TTB's 19 deprels and 19 of the combined arm's 33 — silently, with
# their recall pinned to zero) and `tag_acc = 0.0` in score_weights. Read that docstring first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"

#: arm -> training corpus prefix. `ttb` is TTB alone; `both` is TTB + the MWTT split.
declare -A PREFIX=([ttb]=ta_ttb-sud [both]=ta_ttb_mwtt-sud)
ARMS="${ARMS:-ttb both}"

phase=${1:-all}

do_prep() {
  echo "### PREP: stage both treebanks, project MWTT onto TTB's XPOS tagset, convert"
  $PY scripts/prep_ta.py
  $PY scripts/normalise_ta_xpos.py --learn assets_ta/ta_ttb-sud-train.conllu --report \
      --holdout assets_ta/ta_ttb-sud-test.conllu
  rm -rf corpus_ta && mkdir -p corpus_ta
  for f in assets_ta/ta_*-sud-*.conllu; do
    $PY -m spacy convert "$f" corpus_ta/ --converter conllu -n 10 >/dev/null 2>&1
  done
  ls corpus_ta/
}

# $1=arm suffix (what lands in training_ta_<suffix>), $2=config, $3=train, $4=dev
train_arm() {
  local name=$1 cfg=$2 tr=$3 dv=$4
  echo "### TRAIN training_ta_${name}/"
  $PY -u -m spacy train "$cfg" $CODE --output "training_ta_${name}/" \
    --paths.train "$tr" --paths.dev "$dv" > "train_ta_${name}.log" 2>&1
  [ -d "training_ta_${name}/model-best" ] || {
    echo "!! FAILED"; tail -20 "train_ta_${name}.log"; exit 1; }
  grep -E '^[[:space:]]*[0-9]' "train_ta_${name}.log" | tail -1
}

do_base() {
  for arm in $ARMS; do
    local p=${PREFIX[$arm]}
    train_arm "${arm}_seg" configs/config_ta_seg.cfg \
      "corpus_ta/${p}-train.spacy" "corpus_ta/${p}-dev.spacy"
  done
}

do_layers() {
  # The freeze recipe: source the base arm, freeze it, train ONLY the new component with its own
  # small HashEmbedCNN. The lemvec arm needs BOTH of these, because it moves them to the FRONT of
  # the parser and reads their PREDICTED output.
  for arm in $ARMS; do
    local p=${PREFIX[$arm]}
    bash scripts/train_morph.sh "ta_${arm}"
    bash scripts/train_lemma.sh "ta_${arm}"
  done
}

do_lemvec() {
  for arm in $ARMS; do
    local p=${PREFIX[$arm]}
    $PY scripts/make_ta_lemvec_config.py --arm "$arm" --out "configs/config_ta_${arm}_lemvec.cfg"
    $PY scripts/make_ta_lemvec_config.py --arm "$arm" --control \
        --out "configs/config_ta_${arm}_lemvec_ctl.cfg"
    train_arm "${arm}_lemvec"     "configs/config_ta_${arm}_lemvec.cfg" \
      "corpus_ta/${p}-train.spacy" "corpus_ta/${p}-dev.spacy"
    train_arm "${arm}_lemvec_ctl" "configs/config_ta_${arm}_lemvec_ctl.cfg" \
      "corpus_ta/${p}-train.spacy" "corpus_ta/${p}-dev.spacy"
  done
}

do_order() {
  # Grafted onto BOTH the plain base and the lemvec arm, so the two channels can be read apart:
  # `order` against `seg` prices the augmentation alone, `order_lemvec` against `lemvec` prices it
  # on top of the lemma+morphology channel.
  echo "### ORDER: word-order augmentation, on the base and on the lemvec arm"
  $PY scripts/check_dravidian_order.py corpus_ta/ta_ttb_mwtt-sud-train.spacy --lang ta
  for arm in $ARMS; do
    local p=${PREFIX[$arm]}
    for on in seg lemvec; do
      local src=configs/config_ta_seg.cfg
      [ "$on" = lemvec ] && src=configs/config_ta_${arm}_lemvec.cfg
      [ -f "$src" ] || { echo "== ${arm}/${on}: $src missing -- skip"; continue; }
      local name="${arm}_order" ; [ "$on" = lemvec ] && name="${arm}_order_lemvec"
      local cfg="configs/config_ta_${name}.cfg" labels="labels_ta_${name}"
      $PY scripts/make_dravidian_order_config.py "$src" --lang ta --out "$cfg" \
          --labels-dir "$labels"
      # SIX passes, not one. Under word-order augmentation the PARSER's labels are properties of
      # the ORDER (pseudo-projectivised arcs carry a `||` suffix naming the lifted arc), so one
      # pass under-collects and a missing parser label is a KeyError, not a silent loss.
      $PY scripts/init_aug_labels.py "$cfg" "$labels" $CODE --passes 6 \
          --paths.train "corpus_ta/${p}-train.spacy" --paths.dev "corpus_ta/${p}-dev.spacy"
      train_arm "$name" "$cfg" "corpus_ta/${p}-train.spacy" "corpus_ta/${p}-dev.spacy"
    done
  done
}

do_eval() {
  echo "### EVAL (gold-preproc). TTB slice reported apart from the combined test."
  for arm in $ARMS; do
    for kind in seg lemvec lemvec_ctl order order_lemvec; do
      d="training_ta_${arm}_${kind}/model-best"
      [ -d "$d" ] || { echo "== ${arm}/${kind}: MISSING -- skip"; continue; }
      echo "== ${arm}/${kind}"
      for slice in ta_ttb-sud ta_mwtt-sud ta_ttb_mwtt-sud; do
        t="corpus_ta/${slice}-test.spacy"
        [ -f "$t" ] || continue
        printf "   %-16s " "$slice"
        $PY -m spacy evaluate "$d" "$t" --gold-preproc $CODE \
            --output "metrics_ta_${arm}_${kind}_${slice}.json" 2>/dev/null \
          | grep -E '^(TAG|UAS|LAS|POS|MORPH|LEMMA) ' | tr -s ' \n' ' '
        echo
      done
    done
  done
}

case "$phase" in
  prep)   do_prep ;;
  base)   do_base ;;
  layers) do_layers ;;
  lemvec) do_lemvec ;;
  order)  do_order ;;
  eval)   do_eval ;;
  all)    do_prep; do_base; do_layers; do_lemvec; do_order; do_eval ;;
  *) echo "unknown phase: $phase (prep|base|layers|lemvec|order|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
