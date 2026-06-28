#!/bin/bash
# ext + macron release arm (Latin).  The current release la_sud_ittbproiel is the ext
# (extended-scope relabel) model; this trains its macron-augmented successor: ONE parser
# on the UNION of plain-ext and macronised-ext data, so the released model keeps the ext
# udep disambiguation AND is robust to macronised input.  Macrons come from
# transfer_macrons.py (FORM transform composed onto the ext deprels) -- no macroniser needed.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python
P=la_ittbproiel-sud

echo "### convert ext+macron conllu -> corpus_la_ext_macron/"
rm -rf corpus_la_ext_macron && mkdir -p corpus_la_ext_macron
for s in train dev test; do
  $PY -m spacy convert assets_la/$P-$s.relabeled_ext.macron.conllu corpus_la_ext_macron/ \
    --converter conllu -n 10 >/dev/null 2>&1
done

echo "### build union corpus dirs (plain-ext + macron-ext)"
rm -rf corpus_la_ext_union && mkdir -p corpus_la_ext_union/train corpus_la_ext_union/dev
cp corpus_la_ext/$P-train.relabeled_ext.spacy             corpus_la_ext_union/train/plain.spacy
cp corpus_la_ext_macron/$P-train.relabeled_ext.macron.spacy corpus_la_ext_union/train/macron.spacy
cp corpus_la_ext/$P-dev.relabeled_ext.spacy               corpus_la_ext_union/dev/plain.spacy
cp corpus_la_ext_macron/$P-dev.relabeled_ext.macron.spacy   corpus_la_ext_union/dev/macron.spacy

echo "### train ext+macron union parser -> training_la_ext_macron_union/"
$PY -m spacy train configs/config_la.cfg \
  --output training_la_ext_macron_union/ \
  --paths.train corpus_la_ext_union/train \
  --paths.dev   corpus_la_ext_union/dev > train_la_ext_macron_union.log 2>&1
if [ ! -d training_la_ext_macron_union/model-best ]; then
  echo "!! TRAIN FAILED"; tail -15 train_la_ext_macron_union.log; exit 1
fi
tail -2 train_la_ext_macron_union.log

echo "### evaluate (gold-preproc) vs ext baseline (metrics_la_ext.json)"
echo "-- ext+macron union model on PLAIN-ext test --"
$PY -m spacy evaluate training_la_ext_macron_union/model-best \
  corpus_la_ext/$P-test.relabeled_ext.spacy \
  --gold-preproc --output metrics_la_ext_macron_union_plain.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS'
echo "-- ext+macron union model on MACRON-ext test --"
$PY -m spacy evaluate training_la_ext_macron_union/model-best \
  corpus_la_ext_macron/$P-test.relabeled_ext.macron.spacy \
  --gold-preproc --output metrics_la_ext_macron_union_macron.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS'
echo "DONE"
