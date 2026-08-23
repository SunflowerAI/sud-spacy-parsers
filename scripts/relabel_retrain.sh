#!/bin/bash
# Relabel udep -> comp:obl/mod, then convert + retrain + evaluate, per language.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
PY=.venv/bin/python
declare -A DIR=( [zh]=assets_zh/SUD_Chinese-GSDSimp [ko]=assets_ko/SUD_Korean-GSD [id]=assets_id/SUD_Indonesian-GSD )
declare -A PREF=( [zh]=zh_gsdsimp-sud [ko]=ko_gsd-sud [id]=id_gsd-sud )

for lang in id zh ko; do
  echo "############## RELABEL $lang ##############"
  $PY scripts/lang_relabel.py --lang $lang
  p=${PREF[$lang]}; dir=${DIR[$lang]}
  mkdir -p corpus_${lang}_rl
  for split in train dev test; do
    $PY -m spacy convert $dir/$p-$split.relabeled.conllu corpus_${lang}_rl/ --converter conllu -n 10 \
      2>&1 | grep -oE 'Generated output file \([0-9]+ documents\)' | sed "s/^/  $lang $split: /"
  done
  echo "############## RETRAIN $lang ##############"
  $PY -m spacy train configs/config_$lang.cfg --output training_${lang}_rl/ \
    --paths.train corpus_${lang}_rl/$p-train.relabeled.spacy \
    --paths.dev   corpus_${lang}_rl/$p-dev.relabeled.spacy > train_${lang}_rl.log 2>&1
  if [ ! -d training_${lang}_rl/model-best ]; then echo "!! $lang retrain FAILED"; tail -5 train_${lang}_rl.log; continue; fi
  echo "############## EVAL $lang (relabeled) ##############"
  $PY -m spacy evaluate training_${lang}_rl/model-best corpus_${lang}_rl/$p-test.relabeled.spacy \
    --gold-preproc --output metrics/${lang}/metrics_${lang}_rl.json 2>&1 | grep -E 'TOK|TAG|UAS|LAS' | head -4
done
echo "ALL RELABEL+RETRAIN DONE"
