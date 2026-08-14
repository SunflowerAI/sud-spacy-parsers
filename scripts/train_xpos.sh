#!/bin/bash
# Retrain each arm's XPOS tagger so that it is DOWNSTREAM of UPOS and FEATS.
#
# Every arm here grew the same way: the base pipeline was [tok2vec, tagger, parser], and the
# morphologiser was added LATER as a frozen layer. So the one component whose target is largely a
# restatement of UPOS+FEATS -- the XPOS tagger -- is the only one that cannot see them, purely
# because of the order the layers were built in. This driver moves the tagger to the END of the
# pipeline, behind the morphologiser, and gives its encoder POS and MORPH channels alongside the
# token embedding it already had (scripts/make_xpos_config.py).
#
# Ordinary freeze recipe otherwise: every other component is sourced and frozen, so it comes out
# byte-identical and no published LAS/UAS/lemma figure needs re-measuring. Only the tagger moves.
#
# Measured headroom (majority-class maps fitted on train, scored on test -- how much gold UPOS+FEATS
# adds ON TOP OF the form): ar +19.6, zh +14.2, la +13.8, en +13.2, en_gum +13.0, yue +11.7,
# id +8.2, ko +4.3, fa +4.1, sa 0 (its XPOS is a copy of UPOS). These are ORACLE deltas: at
# inference the tagger reads PREDICTED UPOS/FEATS, so expect a fraction of them.
#
#   bash scripts/train_xpos.sh ar en en_gum fa ja id ko la zh yue lzh sa
#   XPOS_CTL=1 bash scripts/train_xpos.sh ar        # the capacity control (no POS/MORPH channels)
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=.venv/bin/python
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
CODE="--code scripts/seg_code.py"

# The capacity control writes to a separate arm and passes --no-cond: the two extra channels are
# also extra parameters, so a gain that does not survive this is parameters, not the feature.
# `_xposdown` (not `_xpos`): la and en_gum already have a `*_xpos` arm, which is the
# XPOS-NORMALISATION tagger -- a different experiment on the same column.
#
#   (default)      POS + the hashed MORPH bundle       -> _xposdown       MEASURED AND LOST
#   XPOS_CTL=1     neither: the capacity control       -> _xposdown_ctl
#   XPOS_FEATS=1   POS + ONE TABLE PER FEATURE         -> _xposfeat
#
# The bundle arm is kept runnable because it is the baseline the per-feature arm has to beat.
#   XPOS_TOP=1     inject ABOVE the encoder instead of below: the tagger keeps its
#                  Tok2VecListener on the FROZEN shared encoder -- so its token representation is
#                  EXACTLY the released tagger's -- and the morphology is concatenated just under
#                  the softmax. Combines with XPOS_FEATS (per-feature side channel) and with
#                  XPOS_CTL, which then means listener-only: the released tagger with a retrained
#                  head, and therefore the tightest control the harness can produce.
#   XPOS_WARM=1    as XPOS_TOP, but the tagger STARTS as the released one: its head is copied in
#                  and the side channel's columns zeroed, so at step 0 the model is the released
#                  tagger to the bit and the new channel has to earn every column. This is also
#                  what covers la and en_gum, whose released taggers carry their OWN HashEmbedCNN
#                  rather than a listener -- that encoder is copied too.

# Where each language's RELEASED tagger lives. For most arms that is the same lemma arm the frozen
# components are sourced from, but la and en_gum ship the tagger built by the XPOS-NORMALISATION
# work (2 342 Index Thomisticus codes / the punctuation-normalised PTB set), which lives in its own
# arm and was grafted in -- their lemma arms still hold the superseded tagger.
tagger_arm() { case "$1" in
  la)     echo training_la_aug_xpos/model-best ;;
  en_gum) echo training_en_gum_xpos/model-best ;;
  lzh)    echo training_lzh_trad_morph/model-best ;;
  id)     echo training_id_split_lemma/model-best ;;
  ko)     echo training_ko_eojeol_lemma/model-best ;;
  zh)     echo training_zh_trad_lemma/model-best ;;
  sa)     echo training_sa_multitask/model-best ;;
  # XPOS_SRC_ARM overrides it. Needed for the vocalisation-augmented chains: the conditioned
  # tagger must be trained on the AUGMENTED corpus, or a tagger that has only seen bare text gets
  # grafted into a pipeline whose whole point is reading pointed text -- and the tagger is the
  # component most sensitive to spelling of the lot.
  *)      echo "${XPOS_SRC_ARM:-training_$1_lemma/model-best}" ;;
esac; }

# The treebank each language's XPOS gold comes from -- read ONLY to derive the feature inventory.
feats_conllu() { case "$1" in
  en)  echo assets/en_ewt-sud-train.relabeled_ext.conllu ;;
  en_gum) echo assets/en_ewtgum-sud-train.relabeled_ext.conllu ;;
  ar)  echo assets_ar/SUD_Arabic-PADT/ar_padt-sud-train.relabeled_ext.conllu ;;
  fa)  echo assets_fa/SUD_Persian-PerDT/fa_perdt-sud-train.relabeled_ext.conllu ;;
  ja)  echo assets_ja/SUD_Japanese-GSD/ja_gsd-sud-train.relabeled_ext.udep_ruled.conllu ;;
  id)  echo assets_id/SUD_Indonesian-GSD/id_gsd-sud-train.relabeled_ext.conllu ;;
  ko)  echo assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu ;;
  la)  echo assets_la/la_ittbproiel-sud-train.relabeled_ext.conllu ;;
  zh)  echo assets_zh/SUD_Chinese-GSD/zh_gsd-sud-train.relabeled_ext.conllu ;;
  yue) echo assets_yue/SUD_Cantonese-HK/yue_hk-sud-train.relabeled_ext.conllu ;;
  lzh) echo assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu ;;
  sa)  echo corpus_sa_csl_rev/train.csl_rev.conllu ;;
esac; }

# DERIVED at run time, never hardcoded: build_feats_inventory.py ranks each morphological category
# by the information it carries about XPOS once the FORM is already known, and several treebanks
# have none that clears the bar (zh 0, id 0, ko 0 -- the form alone all but determines their XPOS).
# Those arms have nothing to condition on and are skipped rather than trained on empty channels.
derive_feats() {
  local lang=$1 src; src=$(feats_conllu "$lang")
  [ -f "$src" ] || { echo "|"; return; }
  $PY scripts/build_feats_inventory.py "$src" --emit 2>/dev/null \
    | awk -F'[][]' '/^feats  *=/{f=$2} /^feat_rows  *=/{r=$2} END{gsub(/[" ]/,"",f);gsub(/ /,"",r);print f"|"r}'
}

train() {  # $1=lang $2=base_cfg $3=src_arm $4=train $5=dev  [$6=labels-dir, streaming configs only]
  local lang=$1 base=$2 src=$3 tr=$4 dv=$5 labels=$6
  local SUF COND TOP
  if [ -n "$XPOS_TOP" ] || [ -n "$XPOS_WARM" ]; then
    TOP=--top
    if [ -n "$XPOS_WARM" ]; then
      TOP="--top --warm-start $(tagger_arm "$lang")"
    fi
    if [ -n "$XPOS_CTL" ]; then SUF=${XPOS_WARM:+_xposwarm_ctl}; SUF=${SUF:-_xpostop_ctl}; COND=--no-cond
    else SUF=${XPOS_WARM:+_xposwarm}; SUF=${SUF:-_xpostop}; COND=; fi
  elif [ -n "$XPOS_CTL" ]; then SUF=_xposdown_ctl; COND=--no-cond; TOP=
  elif [ -n "$XPOS_FEATS" ]; then SUF=_xposfeat; COND=; TOP=
  else SUF=_xposdown; COND=; TOP=; fi
  local cfg=configs/config_${lang}${SUF}.cfg arm=training_${lang}${SUF}
  [ -d "$src" ] || { echo "  $lang: SRC $src missing -- skip"; return 1; }
  local FEATARGS=""
  if [ -n "$XPOS_FEATS" ] && [ -z "$XPOS_CTL" ]; then
    local d; d=$(derive_feats "$lang")
    local fl=${d%%|*} rl=${d##*|}
    if [ -z "$fl" ]; then
      # No FEATS key clears the bar (zh, id, ko: their XPOS is a function of the spelling). Under
      # BOTTOM injection there is nothing to condition on and the run is pointless. Under TOP
      # injection the side channel still carries UPOS, which is a different and much better
      # predicted signal (pos_acc .93-.96), so fall through to the POS+MORPH side channel.
      if [ -z "$XPOS_TOP" ] && [ -z "$XPOS_WARM" ]; then
        echo "  $lang: no feature clears the bar -- nothing to condition on, SKIP"; return 0
      fi
      echo "  $lang: no FEATS key clears the bar -- side channel is POS+MORPH only"
      FEATARGS=""
    else
      FEATARGS="--feats $fl --feat-rows $rl"
      echo "  $lang: derived feats $fl"
    fi
  fi
  $PY scripts/make_xpos_config.py "$base" "$src" --out "$cfg" $COND $TOP $FEATARGS \
      ${labels:+--labels-dir "$labels"} --force >/dev/null || { echo "  $lang: cfg FAIL"; return 1; }
  echo "########## xpos $lang -> $arm ##########"
  # verify the conditioning inputs actually reach the training docs BEFORE burning a training run:
  # a missing annotating component leaves the new channels constant and nothing raises.
  $PY scripts/check_xpos_inputs.py "$cfg" --train "$tr" --dev "$dv" 2>&1 | grep -E "POS |MORPH |order|FAIL"
  $PY -u -m spacy train "$cfg" $CODE --output "$arm/" \
    --paths.train "$tr" --paths.dev "$dv" > "train_${lang}${SUF}.log" 2>&1
  if [ -d "$arm/model-best" ]; then
    $PY -c "import json;p=json.load(open('$arm/model-best/meta.json'))['performance'];print(f'  $lang OK  tag_acc {p[\"tag_acc\"]:.4f}  (pos {p.get(\"pos_acc\",0):.4f}  morph {p.get(\"morph_acc\",0):.4f})')"
  else echo "  $lang FAILED:"; tail -15 "train_${lang}${SUF}.log"; fi
}

for lang in "$@"; do
case $lang in
 en)  train en  configs/config_en_lemma.cfg  training_en_lemma/model-best \
        corpus_en_ewt_ext/en_ewt-sud-train.relabeled_ext.spacy corpus_en_ewt_ext/en_ewt-sud-dev.relabeled_ext.spacy ;;
 en_gum) train en_gum configs/config_en_gum_lemma.cfg training_en_gum_lemma/model-best \
        corpus_en_gum_ext/en_ewtgum-sud-train.relabeled_ext.spacy corpus_en_gum_ext/en_ewtgum-sud-dev.relabeled_ext.spacy ;;
 ar)  train ar  configs/config_ar_lemma.cfg  training_ar_lemma/model-best \
        corpus_ar_ext/ar_padt-sud-train.relabeled_ext.spacy corpus_ar_ext/ar_padt-sud-dev.relabeled_ext.spacy ;;
 fa)  train fa  configs/config_fa_lemma.cfg  training_fa_lemma/model-best \
        corpus_fa_ext/fa_perdt-sud-train.relabeled_ext.spacy corpus_fa_ext/fa_perdt-sud-dev.relabeled_ext.spacy ;;
 ja)  train ja  configs/config_ja_lemma.cfg  training_ja_lemma/model-best \
        corpus_ja_ext/ja_gsd-sud-train.relabeled_ext.spacy corpus_ja_ext/ja_gsd-sud-dev.relabeled_ext.spacy ;;
 # id ships the SPLIT chain (char segmenter, enclitics separated), not the older coarsened one.
 id)  train id  configs/config_id_split_lemma.cfg training_id_split_lemma/model-best \
        corpus_id_split/id_gsd-sud-train.relabeled_ext.spacy corpus_id_split/id_gsd-sud-dev.relabeled_ext.spacy ;;
 # ko ships the EOJEOL arm, trained on the ORIGINAL SUD_Korean-GSD.
 ko)  train ko  configs/config_ko_eojeol_lemma.cfg training_ko_eojeol_lemma/model-best \
        corpus_ko_eojeol/ko_gsd-sud-train.relabeled_ext.spacy corpus_ko_eojeol/ko_gsd-sud-dev.relabeled_ext.spacy ;;
 # la's config STREAMS (max_epochs = -1, orthographic augmenter), so `init_nlp` sees only the first
 # 100 examples and the label set must be handed in. labels_la_aug_xpos holds the 2 342 NORMALISED
 # Index Thomisticus tags (labels_la_aug is the superseded 1 952-tag inventory -- do not use it).
 la)  train la  configs/config_la_aug_lemma.cfg training_la_aug_lemma/model-best \
        corpus_la_ext_union/train corpus_la_ext_union/dev labels_la_aug_xpos ;;
 zh)  train zh  configs/config_zh_trad_lemma.cfg training_zh_trad_lemma/model-best \
        corpus_zh_trad/zh_gsd-sud-train.relabeled_ext.spacy corpus_zh_trad/zh_gsd-sud-dev.relabeled_ext.spacy ;;
 yue) train yue configs/config_yue_lemma.cfg training_yue_lemma/model-best \
        corpus_yue_ext/yue_hk-sud-train.relabeled_ext.spacy corpus_yue_ext/yue_hk-sud-dev.relabeled_ext.spacy ;;
 # lzh has NO trained lemmatizer in its arm -- han_lemma_lut replaces it at packaging time.
 lzh) train lzh configs/config_lzh_trad_morph.cfg training_lzh_trad_morph/model-best \
        corpus_lzh_trad/lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.spacy \
        corpus_lzh_trad/lzh_kyoto-sud-dev.relabeled_ext.udep_ruled.punct.rulemerged.spacy ;;
 # sa is the one arm that is JOINT multi-task rather than freeze-recipe, and its XPOS is a copy of
 # UPOS on 100 % of tokens -- so conditioning is near-tautological here and the run is a check, not
 # an expected gain. Its reader is sud.CompoundCorpus.v1 (the tokeniser's Compound=Yes input feat).
 sa)  train sa  configs/config_sa_multitask.cfg training_sa_multitask/model-best \
        corpus_sa_multitask/train.spacy corpus_sa_multitask/dev.spacy ;;
 *) echo "unknown lang: $lang" ;;
esac
done
echo "########## train_xpos done ##########"
