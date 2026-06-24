#!/bin/bash
# Train + evaluate baseline tagger+parser for the five new languages on their SUD treebanks.
# gold_preproc (in the config) + --gold-preproc (eval) make train/eval use the treebank's
# gold tokens, so the rule/char tokenisers don't break alignment.
#   fa  Persian-PerDT          ar  Arabic-PADT         la  Latin-ITTB+PROIEL (merged)
#   sa  Sanskrit-Vedic         lzh Classical_Chinese-Kyoto (custom char tokeniser, needs --code)
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python

declare -A PREF=( [fa]=fa_perdt-sud [ar]=ar_padt-sud [la]=la_ittbproiel-sud \
                  [sa]=sa_vedic-sud [lzh]=lzh_kyoto-sud [ja]=ja_gsd-sud )
CODE_lzh="--code scripts/lzh_tokenizer.py"

for lang in "$@"; do
  p=${PREF[$lang]}
  code_var="CODE_$lang"; code=${!code_var}
  echo "############## TRAIN $lang ($p) ##############"
  $PY -m spacy train configs/config_$lang.cfg $code \
    --output training_$lang/ \
    --paths.train corpus_$lang/$p-train.spacy \
    --paths.dev   corpus_$lang/$p-dev.spacy > train_$lang.log 2>&1
  if [ ! -d training_$lang/model-best ]; then
    echo "!! TRAIN $lang FAILED"; tail -8 train_$lang.log; continue
  fi
  tail -2 train_$lang.log
  echo "############## EVAL $lang (gold-preproc) ##############"
  $PY -m spacy evaluate training_$lang/model-best corpus_$lang/$p-test.spacy $code \
    --gold-preproc --output metrics_$lang.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS' | head -5
done
echo "DONE: $*"
