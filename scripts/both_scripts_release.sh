#!/bin/bash
# Build the dual-script (simplified + traditional) release models for Chinese and
# Classical Chinese, and package them as the zh_sud_gsdboth / lzh_sud_kyoto wheels.
#
# Chinese has two real treebanks for the same sentences: SUD_Chinese-GSD (the original
# TRADITIONAL annotation) and SUD_Chinese-GSDSimp (its simplified auto-conversion). We
# train on the union of both — authentic traditional + simplified — rather than re-
# traditionalising GSDSimp with OpenCC (lossy: simplification is many-to-one). The
# ext relabel lives on GSDSimp; transfer_relabel_gsd.py overlays it onto the aligned
# GSD tokens (udep-only, with an alignment guard), since the comp:obl/mod decision is
# script-independent. Classical Chinese has NO simplified counterpart treebank, so its
# simplified half IS auto-converted from Kyoto with OpenCC t2s (scripts/opencc_conllu.py;
# char-level + length-preserving, so deprels/heads carry over untouched). Parsers train
# on gold tokens (gold_preproc), so the added script is pure extra signal.
#
# Prereqs: .venv, qwen3-relabelled *.relabeled_ext.conllu in assets_zh/SUD_Chinese-GSDSimp
# and assets_lzh/... (built by relabel_ext.py), the raw SUD_Chinese-GSD extracted, and
# `pip install opencc-python-reimplemented spacy-pkuseg`.
set -e
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python

############################## Chinese (zh) = GSD + GSDSimp ##############################
SIMP=assets_zh/SUD_Chinese-GSDSimp
GSD=assets_zh/SUD_Chinese-GSD
ZBOTH=assets_zh/SUD_Chinese-GSDBoth
mkdir -p "$ZBOTH" corpus_zh_both
[ -f "$GSD/zh_gsd-sud-train.conllu" ] || tar xzf assets_zh/SUD_Chinese-GSD.tgz -C assets_zh/
# overlay the GSDSimp ext relabel onto the real traditional GSD
$PY scripts/transfer_relabel_gsd.py
for s in train dev test; do
  for ext in ".conllu" ".relabeled_ext.conllu"; do
    # GSDSimp half verbatim; GSD half with sent_ids suffixed -hant to avoid id collision
    cp "$SIMP/zh_gsdsimp-sud-$s$ext" "$ZBOTH/zh_gsdboth-sud-$s$ext"
    sed -E 's/^(# (sent_id|newdoc id|newpar id) =.*)$/\1-hant/' \
        "$GSD/zh_gsd-sud-$s$ext" >> "$ZBOTH/zh_gsdboth-sud-$s$ext"
  done
  $PY -m spacy convert "$ZBOTH/zh_gsdboth-sud-$s.relabeled_ext.conllu" corpus_zh_both/ \
      --converter conllu -n 10
done

# retrain the pkuseg segmenter on both scripts (raw-text inference; gold_preproc
# training is unaffected). Writes models/zh_gsdboth_pkuseg + corpus_zh_pkuseg_both/userdict.txt.
$PY scripts/train_pkuseg_zh.py --treebank "$ZBOTH" \
    --out models/zh_gsdboth_pkuseg --utf8 corpus_zh_pkuseg_both

# config_zh_both.cfg points pkuseg at models/zh_gsdboth_pkuseg + the both-script userdict
$PY -m spacy train configs/config_zh_both.cfg --output training_zh_both_ext/ \
  --paths.train corpus_zh_both/zh_gsdboth-sud-train.relabeled_ext.spacy \
  --paths.dev   corpus_zh_both/zh_gsdboth-sud-dev.relabeled_ext.spacy

############################## Classical Chinese (lzh) ##############################
KTRAD=assets_lzh/SUD_Classical_Chinese-Kyoto
KHANS=assets_lzh/SUD_Classical_Chinese-Kyoto-Hans
KBOTH=assets_lzh/SUD_Classical_Chinese-Kyoto-Both
mkdir -p "$KHANS" "$KBOTH" corpus_lzh_both
for s in train dev test; do
  $PY scripts/opencc_conllu.py "$KTRAD/lzh_kyoto-sud-$s.relabeled_ext.conllu" \
      "$KHANS/lzh_kyotohans-sud-$s.relabeled_ext.conllu" --config t2s --suffix=-hans
  cat "$KTRAD/lzh_kyoto-sud-$s.relabeled_ext.conllu" \
      "$KHANS/lzh_kyotohans-sud-$s.relabeled_ext.conllu" \
      > "$KBOTH/lzh_kyotoboth-sud-$s.relabeled_ext.conllu"
  $PY -m spacy convert "$KBOTH/lzh_kyotoboth-sud-$s.relabeled_ext.conllu" corpus_lzh_both/ \
      --converter conllu -n 10
done

$PY -m spacy train configs/config_lzh.cfg --output training_lzh_both_ext/ \
  --code scripts/lzh_tokenizer.py \
  --paths.train corpus_lzh_both/lzh_kyotoboth-sud-train.relabeled_ext.spacy \
  --paths.dev   corpus_lzh_both/lzh_kyotoboth-sud-dev.relabeled_ext.spacy

# clause_parser is a non-trainable last pipe — add it after training (raw multi-clause inference)
$PY scripts/add_clause_parser.py training_lzh_both_ext/model-best training_lzh_both_ext/model-clause

############################## package wheels ##############################
mkdir -p build_zh_both build_lzh_both
$PY -m spacy package training_zh_both_ext/model-best build_zh_both \
  --name sud_gsdboth --version 0.1.0 --build wheel --force
$PY -m spacy package training_lzh_both_ext/model-clause build_lzh_both \
  --name sud_kyoto --version 0.1.0 \
  --code scripts/lzh_tokenizer.py,scripts/clause_parser.py --build wheel --force

echo "Wheels:"
find build_zh_both build_lzh_both -name '*.whl'
echo "Upload with:  gh release upload v0.1.0 <wheel> <wheel> --clobber"
