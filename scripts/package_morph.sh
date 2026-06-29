#!/bin/bash
# Package the morphologizer-equipped wheels (v0.1.0, to clobber the existing release). Each language
# reuses its released packaging recipe pointed at training_<lang>_morph/model-best (see package_seg.sh).
# lzh/sa get clause_parser re-appended first (it must run AFTER the morphologizer so pos_ from the
# whole-doc pass is preserved); yue gets the pkuseg tokenizer swapped in.
# Usage: bash scripts/package_morph.sh en ar fa ja id ko la zh yue lzh sa
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

pkg() {  # $1=lang  $2=src model dir  $3=--name value  $4=optional --code arg
  local lang=$1 src=$2 name=$3 code=$4
  if [ ! -d "$src" ]; then echo "  $lang: SRC $src missing — skip"; return; fi
  rm -rf build_morph/$lang && mkdir -p build_morph/$lang
  $PY -m spacy package "$src" build_morph/$lang --name "$name" --version 0.1.0 $code \
    --build wheel --force >build_morph/$lang.log 2>&1
  local whl=$(find build_morph/$lang -name '*.whl')
  echo "  $lang -> ${whl:-FAILED}"
  [ -z "$whl" ] && tail -8 build_morph/$lang.log
}

for lang in "$@"; do
case $lang in
  en)  pkg en  training_en_morph/model-best  sud_ewt               "" ;;
  ar)  pkg ar  training_ar_morph/model-best  sud_padt              "--code scripts/ar_tokenizer.py" ;;
  fa)  pkg fa  training_fa_morph/model-best  sud_perdt             "" ;;
  ja)  pkg ja  training_ja_morph/model-best  sud_gsd               "" ;;
  id)  pkg id  training_id_morph/model-best  sud_gsd               "" ;;
  ko)  pkg ko  training_ko_morph/model-best  sud_gsd               "" ;;
  la)  pkg la  training_la_morph/model-best  sud_ittbproielperseus "" ;;
  zh)  pkg zh  training_zh_morph/model-best  sud_gsdboth           "" ;;
  yue) $PY scripts/bundle_yue_pkuseg.py --src training_yue_morph/model-best \
            --out training_yue_morph_pkuseg >/dev/null 2>&1
       pkg yue training_yue_morph_pkuseg     sud_hk                "--code scripts/yue_tokenizer.py" ;;
  sa)  $PY scripts/add_clause_parser.py training_sa_morph/model-best training_sa_morph/model-seg \
            --punct-tag PUNCT --sent-punct "।॥|/.?!…" >/dev/null 2>&1
       pkg sa  training_sa_morph/model-seg   sud_sandhi_csl \
            "--code scripts/sa_tokenizer.py,scripts/clause_parser.py" ;;
  lzh) $PY scripts/add_clause_parser.py training_lzh_morph/model-best training_lzh_morph/model-seg \
            >/dev/null 2>&1
       pkg lzh training_lzh_morph/model-seg  sud_kyoto \
            "--code scripts/lzh_tokenizer.py,scripts/clause_parser.py" ;;
  *) echo "  unknown lang: $lang" ;;
esac
done
echo "Wheels in build_morph/*/dist/. Upload with:"
echo "  gh release upload v0.1.0 \$(find build_morph -name '*.whl') --clobber"
