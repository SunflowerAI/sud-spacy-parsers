#!/bin/bash
# Three-way comparison for the Latin morphology-informed parsing experiment.
#
#   training_la_seg                the released base arm  [tok2vec, tagger, parser]
#   training_la_morphfirst         + a FROZEN morphologizer moved to the FRONT, its predicted
#                                  FEATS read by the shared encoder via MORPH
#   training_la_capacity_control   the same TOTAL embedding rows as morphfirst (the 4096 MORPH
#                                  rows spent on NORM instead), so any morphfirst gain that is
#                                  really just extra parameters shows up here too
#
# Evaluated gold-preproc on four slices: the combined test split and the ITTB+PROIEL / Perseus
# slices separately (Perseus is out-of-domain classical poetry, LAS ~55 vs ~78), each plain and
# macronised -- the release trains on the union of plain and macron.
#
# TWO CAVEATS when reading the output:
#  * DEV is not a fair selector here. The frozen morphologizer's own model-best was chosen on the
#    SAME corpus_la_ext_union/dev that selects morphfirst's model-best, so morphfirst's dev score
#    is optimistic. These TEST numbers are the honest comparison.
#  * TAG runs ~80 on the combined test but ~91 on dev purely because Perseus's XPOS is blanked
#    (20 % of combined-test tokens, 0 % of dev) and scores as error. It hits every arm equally.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=.venv/bin/python
P=la_ittbproiel-sud
S=corpus_la_eval_slices

slice_path() {  # $1=slice  $2=plain|macron
  case "$1/$2" in
    all/plain)      echo corpus_la_ext/$P-test.relabeled_ext.spacy ;;
    all/macron)     echo corpus_la_ext_macron/$P-test.relabeled_ext.macron.spacy ;;
    itp/plain)      echo $S/itp-test.relabeled_ext.spacy ;;
    itp/macron)     echo $S/itp-test.relabeled_ext.macron.spacy ;;
    perseus/plain)  echo $S/perseus-test.relabeled_ext.spacy ;;
    perseus/macron) echo $S/perseus-test.relabeled_ext.macron.spacy ;;
  esac
}

for arm in seg morphfirst capacity_control; do
  d=training_la_$arm/model-best
  [ -d "$d" ] || { echo "== $arm: MISSING ($d) -- skip"; continue; }
  echo "== $arm"
  for sl in all itp perseus; do
    for v in plain macron; do
      t=$(slice_path $sl $v)
      [ -f "$t" ] || { printf "   %-8s %-7s MISSING\n" "$sl" "$v"; continue; }
      printf "   %-8s %-7s " "$sl" "$v"
      # grep the summary table only -- `spacy evaluate` also prints a "LAS (per type)" section
      # whose banner line would otherwise be swept up by a bare grep for LAS
      $PY -m spacy evaluate "$d" "$t" --gold-preproc \
          --code scripts/seg_code.py \
          --output metrics_la_${arm}_${sl}_${v}.json 2>/dev/null \
        | grep -E '^(TAG|UAS|LAS) ' | tr -s ' \n' ' '
      echo
    done
  done
done
