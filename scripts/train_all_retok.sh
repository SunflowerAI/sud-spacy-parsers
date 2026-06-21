#!/bin/bash
# Prep retokenised/coarsened data + train zh (pkuseg/GSDSimp), id (coarsened),
# ko (morpheme functional-head), sequentially. English is unchanged -> not retrained.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

echo "================ PREP: Korean morpheme retokenise ================"
mkdir -p assets_ko_retok corpus_ko_retok
for s in train dev test; do
  $PY scripts/retokenize.py --lang ko \
    assets_ko/SUD_Korean-GSD/ko_gsd-sud-$s.conllu \
    assets_ko_retok/ko_gsd-retok-$s.conllu
  $PY -m spacy convert assets_ko_retok/ko_gsd-retok-$s.conllu corpus_ko_retok/ \
    --converter conllu -n 10 >/dev/null 2>&1
done

echo "================ PREP: Indonesian coarsen ================"
mkdir -p assets_id_coarse corpus_id_coarse
for s in train dev test; do
  $PY scripts/coarsen_id.py \
    assets_id/SUD_Indonesian-GSD/id_gsd-sud-$s.conllu \
    assets_id_coarse/id_gsd-coarse-$s.conllu
  $PY -m spacy convert assets_id_coarse/id_gsd-coarse-$s.conllu corpus_id_coarse/ \
    --converter conllu -n 10 >/dev/null 2>&1
done

echo "================ TRAIN zh (pkuseg + GSDSimp) ================"
$PY -m spacy train configs/config_zh.cfg --output training_zh_simp/ \
  --paths.train corpus_zh_simp/zh_gsdsimp-sud-train.spacy \
  --paths.dev   corpus_zh_simp/zh_gsdsimp-sud-dev.spacy   > train_zh_simp.log 2>&1
echo "zh done -> training_zh_simp/ (log: train_zh_simp.log)"

echo "================ TRAIN id (coarsened) ================"
$PY -m spacy train configs/config_id.cfg --output training_id_coarse/ \
  --paths.train corpus_id_coarse/id_gsd-coarse-train.spacy \
  --paths.dev   corpus_id_coarse/id_gsd-coarse-dev.spacy   > train_id_coarse.log 2>&1
echo "id done -> training_id_coarse/ (log: train_id_coarse.log)"

echo "================ TRAIN ko (morpheme functional-head) ================"
$PY -m spacy train configs/config_ko.cfg --output training_ko_retok/ \
  --paths.train corpus_ko_retok/ko_gsd-retok-train.spacy \
  --paths.dev   corpus_ko_retok/ko_gsd-retok-dev.spacy     > train_ko_retok.log 2>&1
echo "ko done -> training_ko_retok/ (log: train_ko_retok.log)"

echo "================ ALL DONE ================"
for L in zh_simp id_coarse ko_retok; do
  echo "--- $L final scores ---"
  grep -E "^[0-9]" train_${L}.log | tail -1
done
