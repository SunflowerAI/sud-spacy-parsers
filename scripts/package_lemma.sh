#!/bin/bash
# Package the lemmatiser-equipped wheels (v0.1.0, to clobber the existing release). Each arm is
# training_<lang>_lemma/model-best = [tok2vec, tagger, parser, morphologizer, lemmatizer]. lzh/sa
# get clause_parser re-appended first (it runs AFTER the lemmatizer and now carries lemma/morph
# through its per-clause re-parse); yue gets the pkuseg tokenizer swapped in; id gets
# id_lemma_case_fix re-appended (safety-net override for the trainable_lemmatizer's
# sentence-initial-capitalisation gap on hyphenated forms, see scripts/id_lemma_case_fix.py).
# zh/la/sa also carry the renamed package names (sud_gsd_simp_trad / sud_ittb_proiel_perseus /
# sud_vedic_ufal_dcs).
# Usage: bash scripts/package_lemma.sh en ar fa ja id ko la zh yue lzh sa
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
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
  id)  $PY scripts/add_id_lemma_case_fix.py training_id_lemma/model-best training_id_lemma/model-final \
            >/dev/null 2>&1
       pkg id  training_id_lemma/model-final sud_gsd \
            "--code scripts/id_lemma_case_fix.py" ;;
  ko)  pkg ko  training_ko_lemma/model-best  sud_gsd                 "" ;;
       # la ships la_macronise as an OPT-IN component here: --code registers the factory (verified:
       # it is imported on spacy.load even though the pipe is absent from the pipeline), so a caller
       # can nlp.add_pipe("la_macronise", config={"lut": ...}). The lookup TABLE is deliberately
       # NOT bundled -- it is Morpheus-derived (CC BY-SA 3.0) and this wheel is CC BY-NC-SA; see
       # NOTICE.md. Build it with scripts/build_la_macron.sh.
       # ⚠ THIS IS NOT WHAT SHIPS. `scripts/package_sud.sh` supersedes this script for the released
       # wheels -- it builds from the SUD arm and adds the sud_misc/sud_idiom/sud_subject layer --
       # and the la_macronise pipe is attached THERE (`--no-lut`, in the pipeline). Adding it here
       # instead produced a la wheel 1.8 MB SMALLER than the released one, having quietly dropped
       # sud_subject and the three sud_* code modules; the size going DOWN is what caught it. If you
       # are changing what the released Latin model contains, change package_sud.sh.
  la)  pkg la  training_la_lemma/model-best  sud_ittb_proiel_perseus \
            "--code scripts/la_macronise.py" ;;
  zh)  pkg zh  training_zh_lemma/model-best  sud_gsd_simp_trad       "" ;;
  yue) $PY scripts/bundle_yue_pkuseg.py --src training_yue_lemma/model-best \
            --out training_yue_lemma_pkuseg >/dev/null 2>&1
       pkg yue training_yue_lemma_pkuseg     sud_hk                  "--code scripts/yue_tokenizer.py" ;;
       # sa source arm is training_sa_lemma3_noannot — the Compound-feature arm (MORPH read as an
       # INPUT feature; +1.30 LAS over the previous training_sa_lemma, which is kept as the
       # pre-Compound baseline). TWO pipes are added post-training: sa_compound FIRST (before
       # tok2vec, so the shared encoder can see the feat) and clause_parser LAST.
  sa)  $PY scripts/add_sa_compound.py training_sa_lemma3_noannot/model-best \
            training_sa_lemma3_noannot/model-compound >/dev/null 2>&1
       $PY scripts/add_clause_parser.py training_sa_lemma3_noannot/model-compound \
            training_sa_lemma3_noannot/model-seg \
            --punct-tag PUNCT --sent-scheme danda >/dev/null 2>&1
       pkg sa  training_sa_lemma3_noannot/model-seg   sud_vedic_ufal_dcs \
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
