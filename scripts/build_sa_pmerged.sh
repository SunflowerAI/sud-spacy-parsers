#!/usr/bin/env bash
# Rebuild the clause-merged Sanskrit corpora WITH punctuation (daṇḍa / double daṇḍa).
# See merge_sa_reparse.py for what the merge and the mark mapping do.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=training_sa_mp2_s1/model-best
V=assets_sa/SUD_Sanskrit-Vedic
RAW=corpus_sa_pmerged_raw
OUT=corpus_sa_pmerged_norm

$PY scripts/merge_sa_reparse.py $M $V/sa_vedic-sud-dev.relabeled_ext.csl_mwt.conllu \
    $V/sa_vedic-sud-dev.pmerged.csl_mwt.conllu --punct
$PY scripts/merge_sa_reparse.py $M $V/sa_vedic-sud-test.relabeled_ext.csl_mwt.conllu \
    $V/sa_vedic-sud-test.pmerged.csl_mwt.conllu --punct
$PY scripts/merge_sa_reparse.py $M corpus_sa_mwt_rl2/train.csl_mwt.conllu \
    $V/sa_vedic-sud-train.pmerged.csl_mwt.conllu --punct

mkdir -p $RAW
for f in train dev test; do
  $PY -m spacy convert $V/sa_vedic-sud-$f.pmerged.csl_mwt.conllu $RAW/ --converter conllu -n 10
done

# NORM carries the padapāṭha; `spacy convert` does not write it, so the transducer pass must follow.
rm -rf $OUT
$PY scripts/make_norm_corpus.py --in $RAW --out $OUT
ls -l $OUT/
