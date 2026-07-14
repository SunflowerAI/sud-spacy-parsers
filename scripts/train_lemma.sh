#!/bin/bash
# Add a trainable EditTreeLemmatizer to each released arm WITHOUT changing parser/tagger/morph/seg.
# Recipe (see scripts/make_lemma_config.py): source + FREEZE the released tok2vec/tagger/parser/
# morphologizer from training_<lang>_morph/model-best, then train ONLY a new lemmatizer that carries
# its OWN small HashEmbedCNN encoder. The frozen components stay byte-identical (no re-verification).
# Each arm reuses its *_morph config (same GoldTokCorpus/plain reader + train/dev data as morph).
#
# Usage: bash scripts/train_lemma.sh en ar fa ja id ko la zh yue lzh sa
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
CODE="--code scripts/seg_code.py"

train() {  # $1=lang $2=train $3=dev  (source is always training_<lang>_morph/model-best)
  local lang=$1 tr=$2 dv=$3
  local morphcfg=configs/config_${lang}_morph.cfg
  local src=training_${lang}_morph/model-best
  local cfg=configs/config_${lang}_lemma.cfg
  if [ ! -d "$src" ]; then echo "$lang: SRC $src missing — skip"; return 1; fi
  $PY scripts/make_lemma_config.py "$morphcfg" "$src" --out "$cfg" || { echo "$lang: cfg FAIL"; return 1; }
  echo "########## lemma $lang -> training_${lang}_lemma ##########"
  $PY -m spacy train "$cfg" $CODE --output training_${lang}_lemma/ \
    --paths.train "$tr" --paths.dev "$dv" > train_${lang}_lemma.log 2>&1
  if [ -d training_${lang}_lemma/model-best ]; then
    $PY -c "import json;p=json.load(open('training_${lang}_lemma/model-best/meta.json'))['performance'];print(f'  $lang OK  lemma_acc {p.get(\"lemma_acc\",0):.4f}  pos_acc {p.get(\"pos_acc\",0):.4f}  dep_las {p.get(\"dep_las\",0):.4f}  tag_acc {p.get(\"tag_acc\",0):.4f}')"
  else echo "  $lang FAILED:"; tail -15 train_${lang}_lemma.log; fi
}

for lang in "$@"; do
case $lang in
 en)  train en  corpus_en_ewt_ext/en_ewt-sud-train.relabeled_ext.spacy  corpus_en_ewt_ext/en_ewt-sud-dev.relabeled_ext.spacy ;;
 ar)  train ar  corpus_ar_ext/ar_padt-sud-train.relabeled_ext.spacy    corpus_ar_ext/ar_padt-sud-dev.relabeled_ext.spacy ;;
 fa)  train fa  corpus_fa_ext/fa_perdt-sud-train.relabeled_ext.spacy   corpus_fa_ext/fa_perdt-sud-dev.relabeled_ext.spacy ;;
 ja)  train ja  corpus_ja_ext/ja_gsd-sud-train.relabeled_ext.spacy     corpus_ja_ext/ja_gsd-sud-dev.relabeled_ext.spacy ;;
 id)  train id  corpus_id_coarse_rl/id_gsd-coarse-train.relabeled.spacy corpus_id_coarse_rl/id_gsd-coarse-dev.relabeled.spacy ;;
 ko)  train ko  corpus_ko_retok_rl/ko_gsd-retok-train.relabeled.spacy  corpus_ko_retok_rl/ko_gsd-retok-dev.relabeled.spacy ;;
 la)  train la  corpus_la_ext_union/train  corpus_la_ext_union/dev ;;
 zh)  train zh  corpus_zh_both/zh_gsdboth-sud-train.relabeled_ext.spacy corpus_zh_both/zh_gsdboth-sud-dev.relabeled_ext.spacy ;;
 yue) train yue corpus_yue_ext/yue_hk-sud-train.relabeled_ext.spacy    corpus_yue_ext/yue_hk-sud-dev.relabeled_ext.spacy ;;
 lzh) train lzh corpus_lzh_both/lzh_kyotoboth-sud-train.relabeled_ext.spacy corpus_lzh_both/lzh_kyotoboth-sud-dev.relabeled_ext.spacy ;;
 sa)  train sa  corpus_sa_csl_rev/train.csl_rev.spacy  corpus_sa_csl_rev/sa_vedic-sud-dev.csl_rev.spacy ;;
 *) echo "unknown lang: $lang" ;;
esac
done
echo "########## train_lemma done ##########"
