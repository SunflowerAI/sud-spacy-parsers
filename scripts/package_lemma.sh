#!/bin/bash
# Package the lemmatiser-equipped wheels (v0.1.0, to clobber the existing release). Each arm is
# training_<lang>_lemma/model-best = [tok2vec, tagger, parser, morphologizer, lemmatizer]. lzh/sa
# get clause_parser re-appended first (it runs AFTER the lemmatizer and now carries lemma/morph
# through its per-clause re-parse); yue gets the pkuseg tokenizer swapped in. zh/la/sa also carry
# the renamed package names (sud_gsd_simp_trad / sud_ittb_proiel_perseus / sud_vedic_ufal_csl).
# Usage: bash scripts/package_lemma.sh en ar fa ja id ko la zh yue lzh sa
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

pkg() {  # $1=lang  $2=src model dir  $3=--name value  $4=optional --code arg
  local lang=$1 src=$2 name=$3 code=$4
  if [ ! -d "$src" ]; then echo "  $lang: SRC $src missing — skip"; return; fi
  rm -rf build_lemma/$lang && mkdir -p build_lemma/$lang
  $PY -m spacy package "$src" build_lemma/$lang --name "$name" --version 0.1.0 $code \
    --build wheel --force >build_lemma/$lang.log 2>&1
  local whl=$(find build_lemma/$lang -name '*.whl')
  echo "  $lang -> ${whl:-FAILED}"
  [ -z "$whl" ] && tail -8 build_lemma/$lang.log
}

for lang in "$@"; do
case $lang in
  en)  pkg en  training_en_lemma/model-best  sud_ewt                 "" ;;
  ar)  pkg ar  training_ar_lemma/model-best  sud_padt                "--code scripts/ar_tokenizer.py" ;;
  fa)  pkg fa  training_fa_lemma/model-best  sud_perdt               "" ;;
  ja)  pkg ja  training_ja_lemma/model-best  sud_gsd                 "" ;;
  id)  pkg id  training_id_lemma/model-best  sud_gsd                 "" ;;
  ko)  pkg ko  training_ko_lemma/model-best  sud_gsd                 "" ;;
  la)  pkg la  training_la_lemma/model-best  sud_ittb_proiel_perseus "" ;;
  zh)  pkg zh  training_zh_lemma/model-best  sud_gsd_simp_trad       "" ;;
  yue) $PY scripts/bundle_yue_pkuseg.py --src training_yue_lemma/model-best \
            --out training_yue_lemma_pkuseg >/dev/null 2>&1
       pkg yue training_yue_lemma_pkuseg     sud_hk                  "--code scripts/yue_tokenizer.py" ;;
  sa)  $PY scripts/add_clause_parser.py training_sa_lemma/model-best training_sa_lemma/model-seg \
            --punct-tag PUNCT --sent-punct "।॥|/.?!…" >/dev/null 2>&1
       pkg sa  training_sa_lemma/model-seg   sud_vedic_ufal_csl \
            "--code scripts/sa_tokenizer.py,scripts/clause_parser.py" ;;
  lzh) $PY scripts/add_clause_parser.py training_lzh_lemma/model-best training_lzh_lemma/model-seg \
            >/dev/null 2>&1
       pkg lzh training_lzh_lemma/model-seg  sud_kyoto \
            "--code scripts/lzh_tokenizer.py,scripts/clause_parser.py" ;;
  *) echo "  unknown lang: $lang" ;;
esac
done
echo "Wheels in build_lemma/*/dist/. Upload with:"
echo "  gh release upload v0.1.0 \$(find build_lemma -name '*.whl') --clobber"
