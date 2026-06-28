#!/bin/bash
# Macron experiment (Latin).  Restores vowel-length macrons to the SUD treebank with
# the Alatius macroniser (see scripts/macronise_la.py) and trains ONE parser on the
# UNION of the un-macronised and macronised data (data augmentation), so the model is
# robust to input either way.  Evaluation (gold-preproc throughout) shows:
#   * union model on plain test                  (metrics_la_macron_union_plain.json)
#   * union model on macronised test             (metrics_la_macron_union_macron.json)
#   * baseline (plain-only) model on macron test (metrics_la_baseline_on_macron.json)
# compared to the existing plain baseline metrics_la.json.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python
P=la_ittbproiel-sud

echo "### convert macronised conllu -> corpus_la_macron/"
rm -rf corpus_la_macron && mkdir -p corpus_la_macron
for s in train dev test; do
  $PY -m spacy convert assets_la/$P-$s.macron.conllu corpus_la_macron/ \
    --converter conllu -n 10 >/dev/null 2>&1
done

echo "### build union corpus dirs (plain + macron)"
rm -rf corpus_la_union && mkdir -p corpus_la_union/train corpus_la_union/dev
cp corpus_la/$P-train.spacy              corpus_la_union/train/$P-train.spacy
cp corpus_la_macron/$P-train.macron.spacy corpus_la_union/train/$P-train.macron.spacy
cp corpus_la/$P-dev.spacy                corpus_la_union/dev/$P-dev.spacy
cp corpus_la_macron/$P-dev.macron.spacy   corpus_la_union/dev/$P-dev.macron.spacy

echo "### train union (augmented) parser -> training_la_macron_union/"
$PY -m spacy train configs/config_la.cfg \
  --output training_la_macron_union/ \
  --paths.train corpus_la_union/train \
  --paths.dev   corpus_la_union/dev > train_la_macron_union.log 2>&1
if [ ! -d training_la_macron_union/model-best ]; then
  echo "!! TRAIN FAILED"; tail -15 train_la_macron_union.log; exit 1
fi
tail -2 train_la_macron_union.log

echo "### evaluate (gold-preproc)"
echo "-- union model on PLAIN test --"
$PY -m spacy evaluate training_la_macron_union/model-best corpus_la/$P-test.spacy \
  --gold-preproc --output metrics_la_macron_union_plain.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS'
echo "-- union model on MACRONISED test --"
$PY -m spacy evaluate training_la_macron_union/model-best corpus_la_macron/$P-test.macron.spacy \
  --gold-preproc --output metrics_la_macron_union_macron.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS'
echo "-- baseline (plain-only) model on MACRONISED test (robustness control) --"
$PY -m spacy evaluate training_la/model-best corpus_la_macron/$P-test.macron.spacy \
  --gold-preproc --output metrics_la_baseline_on_macron.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS'
echo "DONE"
