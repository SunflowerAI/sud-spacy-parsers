#!/bin/bash
# Train + evaluate baseline tagger+parser for zh/ko/id on their SUD treebanks.
# gold_preproc (in the config) + --gold-preproc (eval) make train/eval use the
# treebank's gold tokens, so the jieba/mecab tokenizers don't break alignment.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}   # Korean tokenizer (natto)
PY=.venv/bin/python

declare -A PREF=( [zh]=zh_gsdsimp-sud [ko]=ko_gsd-sud [id]=id_gsd-sud )

for lang in zh ko id; do
  p=${PREF[$lang]}
  echo "############## TRAIN $lang ($p) ##############"
  $PY -m spacy train configs/config_$lang.cfg \
    --output training_$lang/ \
    --paths.train corpus_$lang/$p-train.spacy \
    --paths.dev   corpus_$lang/$p-dev.spacy > train_$lang.log 2>&1
  if [ ! -d training_$lang/model-best ]; then
    echo "!! TRAIN $lang FAILED"; tail -5 train_$lang.log; continue
  fi
  tail -2 train_$lang.log
  echo "############## EVAL $lang (gold-preproc) ##############"
  $PY -m spacy evaluate training_$lang/model-best corpus_$lang/$p-test.spacy \
    --gold-preproc --output metrics/$lang/metrics_$lang.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS' | head -5
done
echo "ALL BASELINES DONE"
