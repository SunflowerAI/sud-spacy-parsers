#!/usr/bin/env bash
# Retrain the arms whose treebanks changed materially under the rule pass.
#
# `apply_udep_rules.py` committed 10 730 residual `udep` tokens from the treebanks' own majority
# behaviour. Only three languages moved enough to be worth a retrain:
#
#     fa   6454 deprels (1.41 % of tokens)  -- 5060 of them the relativiser `که` -> mod
#     lzh  1588 (0.42 %)                    -- temporal 今/後/初 under a verb -> comp:obj
#     ja    684 (0.41 %)                    -- adnominal/copular た/だ -> mod
#
# ar (0.10 %), en (0.22 %) and id/ko/zh/yue (<= 0.05 %) are skipped: too few tokens to move a metric,
# and each retrain is base + morph + lemma + package. The point of these three is not accuracy — a
# 1 % relabel will not move LAS — but OUTPUT CORRECTNESS: the released fa model currently emits
# `udep` on 5060 `که` tokens where the treebank's own convention is `mod`.
#
# The pre-rule treebanks are kept as *.pre_ruled beside each file, so this is reversible.
set -euo pipefail

cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE=scripts/seg_code.py

for lang in fa lzh ja; do
  echo "########## $lang ##########"
  case $lang in
    fa)  SRC=assets_fa/SUD_Persian-PerDT;   PFX=fa_perdt-sud ;;
    lzh) SRC=assets_lzh/SUD_Classical_Chinese-Kyoto; PFX=lzh_kyoto-sud ;;
    ja)  SRC=assets_ja/SUD_Japanese-GSD;    PFX=ja_gsd-sud ;;
  esac
  CORP=corpus_${lang}_ext
  echo "--- rebuild $CORP from the ruled treebank ---"
  for s in train dev test; do
    $PY -m spacy convert "$SRC/${PFX}-${s}.relabeled_ext.conllu" "$CORP/" \
        --converter conllu -n 10 2>&1 | grep -o "([0-9]* documents)" | sed "s/^/    $s /"
  done

  # base: lzh has no seg config (its sentences are 句讀 units); fa/ja learn boundaries
  BASE_CFG=configs/config_${lang}_seg.cfg
  [ -f "$BASE_CFG" ] || BASE_CFG=configs/config_${lang}.cfg
  echo "--- base ($BASE_CFG) ---"
  $PY -m spacy train "$BASE_CFG" --output training_${lang}_ext/ --code $CODE \
      --paths.train "$CORP/${PFX}-train.relabeled_ext.spacy" \
      --paths.dev   "$CORP/${PFX}-dev.relabeled_ext.spacy" 2>&1 | tail -2

  echo "--- morphologizer ---"
  $PY scripts/make_morph_config.py "$BASE_CFG" training_${lang}_ext/model-best \
      --out configs/config_${lang}_morph.cfg >/dev/null
  $PY -m spacy train configs/config_${lang}_morph.cfg --output training_${lang}_morph/ \
      --code $CODE --paths.train "$CORP/${PFX}-train.relabeled_ext.spacy" \
      --paths.dev "$CORP/${PFX}-dev.relabeled_ext.spacy" 2>&1 | tail -2

  echo "--- lemmatizer ---"
  $PY scripts/make_lemma_config.py configs/config_${lang}_morph.cfg \
      training_${lang}_morph/model-best --out configs/config_${lang}_lemma.cfg >/dev/null
  $PY -m spacy train configs/config_${lang}_lemma.cfg --output training_${lang}_lemma/ \
      --code $CODE --paths.train "$CORP/${PFX}-train.relabeled_ext.spacy" \
      --paths.dev "$CORP/${PFX}-dev.relabeled_ext.spacy" 2>&1 | tail -2

  echo "--- does the model now emit the ruled labels? ---"
  $PY -m spacy evaluate training_${lang}_lemma/model-best \
      "$CORP/${PFX}-test.relabeled_ext.spacy" --gold-preproc --code $CODE \
      --output metrics/${lang}/metrics_${lang}_ruled.json 2>&1 | grep -E "TAG|POS|LEMMA|UAS|LAS" | sed 's/^/    /'
done
echo "########## done ##########"
