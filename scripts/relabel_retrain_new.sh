#!/bin/bash
# Baseline (verb-ADP scope) relabel udep -> comp:obl/mod, then convert + retrain + evaluate,
# for the new prepositional languages. Sanskrit is case-based (negligible ADP udep) so it is
# handled only by the extended-scope driver. Set OLLAMA_MODEL to the Phase-3 winner per run.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
declare -A DIR=( [fa]=assets_fa/SUD_Persian-PerDT [ar]=assets_ar/SUD_Arabic-PADT \
                 [la]=assets_la [lzh]=assets_lzh/SUD_Classical_Chinese-Kyoto \
                 [ja]=assets_ja/SUD_Japanese-GSD )
declare -A PREF=( [fa]=fa_perdt-sud [ar]=ar_padt-sud [la]=la_ittbproiel-sud [lzh]=lzh_kyoto-sud \
                  [ja]=ja_gsd-sud )
declare -A CODE=( [lzh]="--code scripts/lzh_tokenizer.py" )

for lang in "$@"; do
  p=${PREF[$lang]}; dir=${DIR[$lang]}; code=${CODE[$lang]}
  echo "############## RELABEL $lang (model=${OLLAMA_MODEL:-qwen3:8b}) ##############"
  $PY scripts/lang_relabel.py --lang $lang
  mkdir -p corpus_${lang}_rl
  for split in train dev test; do
    $PY -m spacy convert $dir/$p-$split.relabeled.conllu corpus_${lang}_rl/ --converter conllu -n 10 \
      2>&1 | grep -oE 'Generated output file \([0-9]+ documents\)' | sed "s/^/  $lang $split: /"
  done
  echo "############## RETRAIN $lang ##############"
  $PY -m spacy train configs/config_$lang.cfg $code --output training_${lang}_rl/ \
    --paths.train corpus_${lang}_rl/$p-train.relabeled.spacy \
    --paths.dev   corpus_${lang}_rl/$p-dev.relabeled.spacy > train_${lang}_rl.log 2>&1
  if [ ! -d training_${lang}_rl/model-best ]; then echo "!! $lang retrain FAILED"; tail -6 train_${lang}_rl.log; continue; fi
  echo "############## EVAL $lang (relabeled, gold-preproc) ##############"
  $PY -m spacy evaluate training_${lang}_rl/model-best corpus_${lang}_rl/$p-test.relabeled.spacy $code \
    --gold-preproc --output metrics_${lang}_rl.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS' | head -4
done
echo "ALL RELABEL+RETRAIN DONE: $*"
