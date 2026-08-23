#!/bin/bash
# Arabic and Persian, vocalisation-augmented: one copy of the treebank, resampled every epoch, so
# the arm reads text at any level of pointing instead of only the level its treebank happens to use.
#
# WHY. Measured on the SAME trees with only the FORM column rewritten (make_ar_variant_conllu.py /
# make_fa_variant_conllu.py, scored by eval_ar_variants.py), the released arms fall off a cliff on
# text they were never shown:
#
#     ar   bare 72.92 LAS   shadda-only 63.72   half-pointed 44.81   fully pointed 18.50   spread 54.42
#     fa   bare 87.18 LAS   no-ZWNJ     82.93   Arabic ی/ک  57.55   fully pointed 33.28   spread 64.40
#
# ar's 54.42 is, to the decimal, the spread Latin had before its own augmentation. fa's Arabic-
# letterform row is the one to notice: ی/ي and ک/ك are what an Arabic keyboard produces, so that is
# a large share of real Persian text rather than an exotic edition, and it costs 29.6 LAS.
#
# TWO DIRECTIONS, forced by the data. ar stores the corpus FULLY POINTED (FORM = PADT's gold Vform)
# and the augmenter only ever REMOVES marks -- a strict superset, exactly as Latin stores the
# macronised copy and strips. fa has no vocalised gold anywhere, so its corpus stays as the treebank
# writes it and the augmenter ADDS marks from the same reconstructed table and the same
# syntactically-derived ezafe rules fa_vocalise ships against.
#
# Phases (run all, or name one): corpus | variants | labels | base | morph | lemma | eval
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE="--code scripts/vocal_augment.py"
LANGS="${LANGS:-ar fa}"

src_train() { case $1 in
  ar) echo corpus_ar_vocal/ar_padt-sud-train.relabeled_ext.vocalised.spacy ;;
  fa) echo corpus_fa_ext/fa_perdt-sud-train.relabeled_ext.spacy ;; esac; }
# Dev stays BARE and un-augmented for both. Undiacritised text is what these arms are judged on, so
# `model-best` must be selected on it -- otherwise the checkpoint drifts toward whichever spellings
# the augmenter happened to sample and the headline regresses to buy a robustness nobody measured.
src_dev() { case $1 in
  ar) echo corpus_ar_ext/ar_padt-sud-dev.relabeled_ext.spacy ;;
  fa) echo corpus_fa_ext/fa_perdt-sud-dev.relabeled_ext.spacy ;; esac; }
test_conllu() { case $1 in
  ar) echo assets_ar/SUD_Arabic-PADT/ar_padt-sud-test.relabeled_ext.conllu ;;
  fa) echo assets_fa/SUD_Persian-PerDT/fa_perdt-sud-test.relabeled_ext.conllu ;; esac; }

make_cfg() {  # $1=base cfg  $2=out cfg  $3=lang  $4=labels dir  [$5=retarget]
  $PY scripts/make_vocal_aug_config.py "$1" --out "$2" --lang "$3" --labels-dir "$4" \
      ${5:+--retarget "$5"}
}
labels() {  # $1=cfg  $2=outdir  $3=lang  $4=passes
  $PY scripts/init_aug_labels.py "$1" "$2" $CODE --passes "$4" \
      --paths.train "$(src_train "$3")" --paths.dev "$(src_dev "$3")"
}

phase="${1:-all}"
for L in $LANGS; do
  if [ "$phase" = all ] || [ "$phase" = corpus ]; then
    if [ "$L" = ar ]; then
      $PY scripts/make_ar_vocalised_corpus.py \
          assets_ar/SUD_Arabic-PADT/ar_padt-sud-{train,dev}.relabeled_ext.conllu \
          --out-dir assets_ar/vocalised
      mkdir -p corpus_ar_vocal
      for f in assets_ar/vocalised/*.conllu; do
        $PY -m spacy convert "$f" corpus_ar_vocal/ --converter conllu -n 10 >/dev/null
      done
    fi
  fi
  if [ "$phase" = all ] || [ "$phase" = variants ]; then
    mk=scripts/make_${L}_variant_conllu.py
    [ "$L" = ar ] && mk=scripts/make_ar_variant_conllu.py
    $PY "$mk" "$(test_conllu "$L")" "corpus_${L}_variants" --prefix "$L"
    for f in corpus_${L}_variants/*.conllu; do
      $PY -m spacy convert "$f" "corpus_${L}_variants/" --converter conllu -n 10 >/dev/null
    done
  fi
  if [ "$phase" = all ] || [ "$phase" = labels ]; then
    make_cfg "configs/config_${L}_seg.cfg" "configs/config_${L}_vocal.cfg" "$L" "labels_${L}_vocal"
    labels "configs/config_${L}_vocal.cfg" "labels_${L}_vocal" "$L" 3
  fi
  if [ "$phase" = all ] || [ "$phase" = base ]; then
    $PY -u -m spacy train "configs/config_${L}_vocal.cfg" $CODE --output "training_${L}_vocal/" \
        --paths.train "$(src_train "$L")" --paths.dev "$(src_dev "$L")" \
        2>&1 | tee "train_${L}_vocal.log"
  fi
  if [ "$phase" = all ] || [ "$phase" = morph ]; then
    # Freeze recipe, unchanged in kind: source the augmented base, freeze it, train ONLY the
    # morphologiser with its own small encoder. It is trained THROUGH the same augmenter, because a
    # morphologiser reading NORM/PREFIX/SUFFIX/SHAPE off spellings it never met would be the arm's
    # own weak point -- the argument that put la's SUD layer through the augmenter too.
    $PY scripts/make_morph_config.py "configs/config_${L}_vocal.cfg" "training_${L}_vocal/model-best" \
        --out "configs/config_${L}_vocal_morph.cfg"
    make_cfg "configs/config_${L}_vocal_morph.cfg" "configs/config_${L}_vocal_morph.cfg" "$L" \
             "labels_${L}_vocal_morph" "training_${L}_seg=training_${L}_vocal"
    labels "configs/config_${L}_vocal_morph.cfg" "labels_${L}_vocal_morph" "$L" 3
    $PY -u -m spacy train "configs/config_${L}_vocal_morph.cfg" $CODE \
        --output "training_${L}_vocal_morph/" \
        --paths.train "$(src_train "$L")" --paths.dev "$(src_dev "$L")" \
        2>&1 | tee "train_${L}_vocal_morph.log"
  fi
  if [ "$phase" = all ] || [ "$phase" = lemma ]; then
    # ⚠ The LEMMATISER is the component that really needs init_aug_labels, and it fails SILENTLY
    # without it: edit-tree labels are properties of the FORM, so كتاب and كِتاب are different
    # trees, and a tree missing from the initial set does not raise -- get_loss maps it to label 0
    # and the token is quietly taught the wrong edit. 10 passes, as la uses.
    $PY scripts/make_lemma_config.py "configs/config_${L}_vocal_morph.cfg" \
        "training_${L}_vocal_morph/model-best" --out "configs/config_${L}_vocal_lemma.cfg"
    make_cfg "configs/config_${L}_vocal_lemma.cfg" "configs/config_${L}_vocal_lemma.cfg" "$L" \
             "labels_${L}_vocal_lemma" "training_${L}_morph=training_${L}_vocal_morph"
    labels "configs/config_${L}_vocal_lemma.cfg" "labels_${L}_vocal_lemma" "$L" 10
    $PY -u -m spacy train "configs/config_${L}_vocal_lemma.cfg" $CODE \
        --output "training_${L}_vocal_lemma/" \
        --paths.train "$(src_train "$L")" --paths.dev "$(src_dev "$L")" \
        2>&1 | tee "train_${L}_vocal_lemma.log"
  fi
  if [ "$phase" = all ] || [ "$phase" = eval ]; then
    rel=training_${L}_sud_xpos/model-best
    $PY scripts/eval_ar_variants.py --model "released=$rel" \
        --model "vocal=training_${L}_vocal_lemma/model-best" \
        --corpus-dir "corpus_${L}_variants" --prefix "$L" \
        --out "metrics/${L}/metrics_${L}_variants.json"
  fi
done
