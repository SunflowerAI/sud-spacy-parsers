#!/bin/bash
# Package the sentence-segmentation wheels (v0.1.0, to clobber the existing release). Each language
# reuses its released packaging recipe pointed at training_<lang>_seg/model-best. sa/lzh are NOT
# retrained — they are repackaged from model-seg (new clause_parser; see add_clause_parser.py).
# Usage: bash scripts/package_seg.sh ar fa ja la id ko zh yue sa lzh
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

pkg() {  # $1=lang  $2=src model dir  $3=--name value  $4=optional --code arg
  local lang=$1 src=$2 name=$3 code=$4
  if [ ! -d "$src" ]; then echo "  $lang: SRC $src missing — skip"; return; fi
  rm -rf build_seg/$lang && mkdir -p build_seg/$lang
  $PY -m spacy package "$src" build_seg/$lang --name "$name" --version 0.1.0 $code \
    --build wheel --force >/dev/null 2>&1
  local whl=$(find build_seg/$lang -name '*.whl')
  echo "  $lang -> ${whl:-FAILED}"
}

for lang in "$@"; do
case $lang in
  ar)  pkg ar  training_ar_seg/model-best  sud_padt              "--code scripts/ar_tokenizer.py" ;;
  fa)  pkg fa  training_fa_seg/model-best  sud_perdt             "" ;;
  ja)  pkg ja  training_ja_seg/model-best  sud_gsd               "" ;;
  la)  pkg la  training_la_seg/model-best  sud_ittbproielperseus "" ;;
  id)  pkg id  training_id_seg/model-best  sud_gsd               "" ;;
  ko)  pkg ko  training_ko_seg/model-best  sud_gsd               "" ;;
  zh)  pkg zh  training_zh_seg/model-best  sud_gsdboth           "" ;;
  yue) $PY scripts/bundle_yue_pkuseg.py --src training_yue_seg/model-best \
            --out training_yue_seg_pkuseg >/dev/null 2>&1
       pkg yue training_yue_seg_pkuseg     sud_hk                "--code scripts/yue_tokenizer.py" ;;
  sa)  pkg sa  training_sa_csl_rev/model-seg   sud_sandhi_csl \
            "--code scripts/sa_tokenizer.py,scripts/clause_parser.py" ;;
  lzh) pkg lzh training_lzh_both_ext/model-seg sud_kyoto \
            "--code scripts/lzh_tokenizer.py,scripts/clause_parser.py" ;;
  *) echo "  unknown lang: $lang" ;;
esac
done