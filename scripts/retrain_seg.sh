#!/bin/bash
# Retrain released models so the PARSER learns sentence boundaries, instead of collapsing raw
# multi-sentence input into one tree (the gold_preproc-trained parsers never saw a boundary).
# Each language reuses its released training recipe but with the *_seg config (corpus readers
# swapped to sud.GoldTokCorpus.v1 — whole multi-sentence docs with gold tokenisation — and sents_f
# given a small score weight). The gold-token reader means zero tokeniser skew (matters for zh/yue
# pkuseg). See scripts/gold_tok_corpus.py and scripts/make_seg_config.py.
#
# Usage: bash scripts/retrain_seg.sh fa ja la zh yue id ko
# (ar already done as the pilot. id/ko need their coarse/retok corpora regenerated first —
#  see scripts/regen_idko_corpora.sh.)
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
CODE="--code scripts/seg_code.py"

train() {  # $1=lang ; expects CFG/TRAIN/DEV set ; $2.. extra spacy-train args
  local lang=$1; shift
  echo "########## TRAIN $lang -> training_${lang}_seg ##########"
  $PY -m spacy train "$CFG" $CODE --output training_${lang}_seg/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" "$@" > train_${lang}_seg.log 2>&1
  if [ -d training_${lang}_seg/model-best ]; then
    echo "  $lang OK:"; grep -E '^[[:space:]]*[0-9]' train_${lang}_seg.log | tail -1
  else
    echo "  $lang FAILED:"; tail -8 train_${lang}_seg.log
  fi
}

for lang in "$@"; do
case $lang in
  fa)  CFG=configs/config_fa_seg.cfg
       TRAIN=corpus_fa_ext/fa_perdt-sud-train.relabeled_ext.spacy
       DEV=corpus_fa_ext/fa_perdt-sud-dev.relabeled_ext.spacy;   train fa ;;
  ja)  CFG=configs/config_ja_seg.cfg
       TRAIN=corpus_ja_ext/ja_gsd-sud-train.relabeled_ext.spacy
       DEV=corpus_ja_ext/ja_gsd-sud-dev.relabeled_ext.spacy;     train ja ;;
  la)  CFG=configs/config_la_seg.cfg
       TRAIN=corpus_la_ext_union/train  DEV=corpus_la_ext_union/dev; train la ;;
  zh)  CFG=configs/config_zh_both_seg.cfg
       TRAIN=corpus_zh_both/zh_gsdboth-sud-train.relabeled_ext.spacy
       DEV=corpus_zh_both/zh_gsdboth-sud-dev.relabeled_ext.spacy;  train zh ;;
  yue) CFG=configs/config_yue_seg.cfg
       TRAIN=corpus_yue_ext/yue_hk-sud-train.relabeled_ext.spacy
       DEV=corpus_yue_ext/yue_hk-sud-dev.relabeled_ext.spacy
       train yue --paths.init_tok2vec zh_both_tok2vec.bin ;;
  id)  CFG=configs/config_id_seg.cfg
       TRAIN=corpus_id_coarse_rl/id_gsd-coarse-train.relabeled.spacy
       DEV=corpus_id_coarse_rl/id_gsd-coarse-dev.relabeled.spacy;  train id ;;
  ko)  CFG=configs/config_ko_seg.cfg
       TRAIN=corpus_ko_retok_rl/ko_gsd-retok-train.relabeled.spacy
       DEV=corpus_ko_retok_rl/ko_gsd-retok-dev.relabeled.spacy;    train ko ;;
  *) echo "unknown lang: $lang" ;;
esac
done
echo "########## retrain_seg done ##########"
