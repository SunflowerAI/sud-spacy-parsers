#!/usr/bin/env bash
# Build corpora and train the SUD MISC layer(s) on top of each released (lemma) arm.
#
# Per language: hoist the wanted MISC keys into FEATS (spacy convert discards MISC), convert to
# .spacy, derive a frozen-recipe config, and train ONLY the new sud_tagger pipe(s). The five
# existing components are sourced and frozen, so they stay byte-identical -- verify with:
#     cmp training_<lang>_lemma/model-best/parser/model training_<lang>_sud/model-best/parser/model
#
#   bash scripts/train_sud.sh                 # every Subject language
#   bash scripts/train_sud.sh la sa           # just these
#
# NB `morph_acc` in the logs reads artificially LOW for these arms: the frozen morphologiser is
# scored against gold FEATS that now carry the hoisted Sud* keys, which it was never trained to
# predict. It is frozen and its score weight is zero, so this is cosmetic.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
CODE="--code scripts/seg_code.py"

# Seven arms annotate Subject=SubjRaising/ObjRaising; ja/ar/ko/id carry none (their UD sources
# don't use xcomp, so the SUD converter never emits it).
#
# `sa` is EXCLUDED from the default: 142 training instances and 14 in test, which scored F 10.5
# trained / 12.5 by rule -- neither is shippable, and the test set is too small to tell them apart
# anyway. Pass it explicitly (`train_sud.sh sa`) if you want to revisit it on more data.
# `zh` and `lzh` are trained but ship the RULE instead (see scripts/package_sud.sh).
ALL_LANGS="en yue lzh fa la ar"
LANGS=${*:-$ALL_LANGS}

# Which SUD MISC features each arm trains a pipe for. `Reported` has no treebank gold anywhere, so
# its target files are the bootstrapped *.reported.conllu written by sud_reported_gold.py; ar and sa
# annotate no Subject, and sa's Subject is too sparse to ship (see above).
feats_for() {
  case "$1" in
    en|fa|la) echo "Subject Reported" ;;
    ar)       echo "Reported" ;;
    sa)       echo "Reported" ;;
    *)        echo "Subject" ;;
  esac
}

# Prefer the reported-annotated file where it exists: it is the same CoNLL-U with `Reported=Yes`
# added to MISC, so it carries the Subject gold unchanged.
prefer_reported() {
  local base=$1 rep="${1%.conllu}.reported.conllu"
  if [ -f "$rep" ]; then echo "$rep"; else echo "$base"; fi
}

mkdir -p assets_sud

# src_conllu <lang> <split> -> the derived CoNLL-U the released arm actually trains on
src_conllu() {
  case "$1" in
    en)  echo "assets/en_ewt-sud-$2.relabeled_ext.conllu" ;;
    zh)  echo "assets_zh/SUD_Chinese-GSDBoth/zh_gsdboth-sud-$2.relabeled_ext.conllu" ;;
    yue) echo "assets_yue/SUD_Cantonese-HK/yue_hk-sud-$2.relabeled_ext.conllu" ;;
    lzh) echo "assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-$2.relabeled_ext.conllu" ;;
    fa)  echo "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-$2.relabeled_ext.conllu" ;;
    ar)  echo "assets_ar/SUD_Arabic-PADT/ar_padt-sud-$2.relabeled_ext.conllu" ;;
    la)  echo "assets_la/la_ittbproiel-sud-$2.relabeled_ext.conllu" ;;
    # sa trains on Vedic-train + UFAL combined (rebuild_sa_csl_rev.sh); dev/test are Vedic only
    sa)  if [ "$2" = train ]; then echo "corpus_sa_csl_rev/train.csl_rev.conllu"
         else echo "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-$2.csl_rev.conllu"; fi ;;
  esac
}

# The lemma arm to source and freeze (sa's released arm has a non-standard directory name).
src_model() {
  case "$1" in
    sa) echo "training_sa_lemma3_noannot/model-best" ;;
    *)  echo "training_$1_lemma/model-best" ;;
  esac
}

prep() {   # $1=lang  $2...=features to hoist
  local lang=$1; shift
  local feats="$*"
  local out=corpus_${lang}_sud
  mkdir -p "$out"
  for split in train dev test; do
    local src; src=$(prefer_reported "$(src_conllu "$lang" "$split")")
    if [ ! -f "$src" ]; then echo "  $lang/$split: missing $src -- skip"; continue; fi
    local hoisted=assets_sud/${lang}-${split}.sud.conllu
    $PY scripts/hoist_sud_gold.py "$src" "$hoisted" --keys $feats || return 1
    $PY -m spacy convert "$hoisted" "$out/" --converter conllu -n 10 >/dev/null || return 1
    mv "$out/$(basename "${hoisted%.conllu}").spacy" "$out/$split.spacy"
  done
  # Latin's released arm trains on the plain UNION macron corpus, so the macron half needs the
  # same treatment and both halves live in one directory (spacy reads every .spacy in a dir).
  if [ "$lang" = la ]; then
    for split in train dev test; do
      local msrc; msrc=$(prefer_reported "assets_la/la_ittbproiel-sud-$split.relabeled_ext.macron.conllu")
      [ -f "$msrc" ] || continue
      local mh=assets_sud/la-${split}.macron.sud.conllu
      $PY scripts/hoist_sud_gold.py "$msrc" "$mh" --keys $feats || return 1
      $PY -m spacy convert "$mh" "$out/" --converter conllu -n 10 >/dev/null || return 1
      mv "$out/$(basename "${mh%.conllu}").spacy" "$out/$split.macron.spacy"
    done
    # one dir per split, so --paths.train can point at a directory holding plain+macron
    for split in train dev; do
      mkdir -p "$out/$split.d"
      mv -f "$out/$split.spacy" "$out/$split.d/plain.spacy" 2>/dev/null
      mv -f "$out/$split.macron.spacy" "$out/$split.d/macron.spacy" 2>/dev/null
    done
  fi
}

train_one() {   # $1=lang  $2...=features
  local lang=$1; shift
  local feats="$*"
  local src; src=$(src_model "$lang")
  local base=configs/config_${lang}_lemma.cfg
  local cfg=configs/config_${lang}_sud.cfg
  local out=corpus_${lang}_sud

  if [ ! -d "$src" ]; then echo "$lang: source model $src missing -- skip"; return 1; fi
  if [ ! -f "$base" ]; then echo "$lang: base config $base missing -- skip"; return 1; fi

  local tr="$out/train.spacy" dv="$out/dev.spacy"
  [ "$lang" = la ] && { tr="$out/train.d"; dv="$out/dev.d"; }
  if [ ! -e "$tr" ]; then echo "$lang: corpus $tr missing -- skip"; return 1; fi

  $PY scripts/make_sud_config.py "$base" "$src" --feats $feats --out "$cfg" || return 1
  echo "  $lang: training -> training_${lang}_sud/ (log: train_${lang}_sud.log)"
  $PY -m spacy train "$cfg" $CODE --output "training_${lang}_sud/" \
      --paths.train "$tr" --paths.dev "$dv" > "train_${lang}_sud.log" 2>&1
  if [ -f "training_${lang}_sud/model-best/meta.json" ]; then
    $PY - "$lang" <<'EOF'
import json, sys
lang = sys.argv[1]
m = json.load(open(f"training_{lang}_sud/model-best/meta.json"))["performance"]
keys = [k for k in m if k.startswith("sud_")]
print("   ", lang, {k: round(m[k], 4) for k in sorted(keys)})
EOF
  else
    echo "    $lang: FAILED -- see train_${lang}_sud.log"
    tail -5 "train_${lang}_sud.log"
  fi
}

for lang in $LANGS; do
  echo "== $lang =="
  feats=$(feats_for "$lang")
  prep "$lang" $feats && train_one "$lang" $feats
done
