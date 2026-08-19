#!/bin/bash
# Add a UPOS+morph `morphologizer` to each released arm WITHOUT changing parser/tagger/segmentation.
# Recipe (see scripts/make_morph_config.py): source + FREEZE the released tok2vec/tagger/parser, then
# train ONLY a new morphologizer that carries its OWN small HashEmbedCNN encoder (width 64 / depth 3 /
# embed 2000). The frozen components stay byte-identical (no parse/seg re-verification needed); the
# dedicated encoder is immune to treebanks whose XPOS is orthogonal to UPOS (e.g. id — 33/46 XPOS map
# to >1 UPOS). Each language reuses its released training recipe: the *_seg config + GoldTokCorpus for
# the learned-boundary arms, the plain config for en (gold_preproc=false) and lzh/sa (clause_parser,
# re-added at packaging time). Mirrors the per-language tables in retrain_seg.sh / package_seg.sh.
#
# Usage: bash scripts/train_morph.sh en ar fa ja id ko la zh yue lzh sa
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=.venv/bin/python
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
CODE="--code scripts/seg_code.py"
# Arms whose derived config is HAND-MAINTAINED and must not be regenerated: sa's morphologiser uses
# spacy.Tok2Vec.v2 + MultiHashEmbed so it can read MORPH (the tokeniser's Compound=Yes) as an INPUT
# feature, which the generator's flat HashEmbedCNN cannot express. make_morph_config.py also refuses
# to clobber these (scripts/config_guard.py); this list just skips the call instead of erroring.
HAND_CFG=" sa "

train() {  # $1=lang $2=base_cfg $3=src_model $4=train $5=dev
  local lang=$1 base=$2 src=$3 tr=$4 dv=$5
  local cfg=configs/config_${lang}_morph.cfg
  if [[ "$HAND_CFG" == *" $lang "* ]]; then
    [ -f "$cfg" ] || { echo "$lang: hand-maintained $cfg missing"; return 1; }
    echo "  $lang: using hand-maintained $cfg (skipping make_morph_config.py)"
  else
    $PY scripts/make_morph_config.py "$base" "$src" --out "$cfg" || { echo "$lang: cfg FAIL"; return 1; }
  fi
  echo "########## morph $lang -> training_${lang}_morph ##########"
  $PY -m spacy train "$cfg" $CODE --output training_${lang}_morph/ \
    --paths.train "$tr" --paths.dev "$dv" > train_${lang}_morph.log 2>&1
  if [ -d training_${lang}_morph/model-best ]; then
    $PY -c "import json;p=json.load(open('training_${lang}_morph/model-best/meta.json'))['performance'];print(f'  $lang OK  pos_acc {p[\"pos_acc\"]:.4f}  morph_acc {p.get(\"morph_acc\",0):.4f}  dep_las {p.get(\"dep_las\",0):.4f}  tag_acc {p.get(\"tag_acc\",0):.4f}  sents_f {p.get(\"sents_f\",0):.4f}')"
  else echo "  $lang FAILED:"; tail -15 train_${lang}_morph.log; fi
}

for lang in "$@"; do
case $lang in
 en)  train en  configs/config.cfg             training_en_ewt_ext/model-best  corpus_en_ewt_ext/en_ewt-sud-train.relabeled_ext.spacy  corpus_en_ewt_ext/en_ewt-sud-dev.relabeled_ext.spacy ;;
 # en_gum is EWT + the non-NonCommercial part of GUM: a SECOND English arm, not a replacement.
 # Same architecture as en (configs/config.cfg, gold_preproc=false); only the corpus differs.
 en_gum) train en_gum configs/config.cfg       training_en_gum_ext/model-best  corpus_en_gum_ext/en_ewtgum-sud-train.relabeled_ext.spacy corpus_en_gum_ext/en_ewtgum-sud-dev.relabeled_ext.spacy ;;
 ar)  train ar  configs/config_ar_seg.cfg      training_ar_seg/model-best      corpus_ar_ext/ar_padt-sud-train.relabeled_ext.spacy    corpus_ar_ext/ar_padt-sud-dev.relabeled_ext.spacy ;;
 fa)  train fa  configs/config_fa_seg.cfg      training_fa_seg/model-best      corpus_fa_ext/fa_perdt-sud-train.relabeled_ext.spacy   corpus_fa_ext/fa_perdt-sud-dev.relabeled_ext.spacy ;;
 ja)  train ja  configs/config_ja_seg.cfg      training_ja_seg/model-best      corpus_ja_ext/ja_gsd-sud-train.relabeled_ext.spacy     corpus_ja_ext/ja_gsd-sud-dev.relabeled_ext.spacy ;;
 id)  train id  configs/config_id_seg.cfg      training_id_seg/model-best      corpus_id_coarse_rl/id_gsd-coarse-train.relabeled.spacy corpus_id_coarse_rl/id_gsd-coarse-dev.relabeled.spacy ;;
 ko)  train ko  configs/config_ko_seg.cfg      training_ko_seg/model-best      corpus_ko_retok_rl/ko_gsd-retok-train.relabeled.spacy  corpus_ko_retok_rl/ko_gsd-retok-dev.relabeled.spacy ;;
 la)  train la  configs/config_la_seg.cfg      training_la_seg/model-best      corpus_la_ext_union/train  corpus_la_ext_union/dev ;;
 zh)  train zh  configs/config_zh_both_seg.cfg training_zh_seg/model-best      corpus_zh_both/zh_gsdboth-sud-train.relabeled_ext.spacy corpus_zh_both/zh_gsdboth-sud-dev.relabeled_ext.spacy ;;
 yue) train yue configs/config_yue_seg.cfg     training_yue_seg/model-best     corpus_yue_ext/yue_hk-sud-train.relabeled_ext.spacy    corpus_yue_ext/yue_hk-sud-dev.relabeled_ext.spacy ;;
 lzh) train lzh configs/config_lzh.cfg         training_lzh_both_ext/model-seg corpus_lzh_both/lzh_kyotoboth-sud-train.relabeled_ext.spacy corpus_lzh_both/lzh_kyotoboth-sud-dev.relabeled_ext.spacy ;;
 sa)  train sa  configs/config_sa.cfg          training_sa_csl_rev/model-seg   corpus_sa_csl_rev/train.csl_rev.spacy  corpus_sa_csl_rev/sa_vedic-sud-dev.csl_rev.spacy ;;
 ta_ttb)  train ta_ttb  configs/config_ta_seg.cfg  training_ta_ttb_seg/model-best   corpus_ta/ta_ttb-sud-train.spacy       corpus_ta/ta_ttb-sud-dev.spacy ;;
 ta_both) train ta_both configs/config_ta_seg.cfg  training_ta_both_seg/model-best  corpus_ta/ta_ttb_mwtt-sud-train.spacy  corpus_ta/ta_ttb_mwtt-sud-dev.spacy ;;
 # te's FEATS column is all but empty (115 of 6 465 tokens carry anything), so morph_acc here is
 # near-vacuous and the pipe is trained for its UPOS half, which MTG does have.
 te)      train te      configs/config_te_seg.cfg  training_te_seg/model-best       corpus_te/te_mtg-sud-train.spacy       corpus_te/te_mtg-sud-dev.spacy ;;
 *) echo "unknown lang: $lang" ;;
esac
done
echo "########## train_morph done ##########"
