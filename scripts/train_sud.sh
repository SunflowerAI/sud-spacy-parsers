#!/usr/bin/env bash
# Build corpora and train the SUD MISC layer(s) on top of each released (lemma) arm.
#
# Per language: hoist the wanted MISC keys into FEATS (spacy convert discards MISC), convert to
# .spacy, derive a frozen-recipe config, and train ONLY the new sud_tagger pipe(s). The five
# existing components are sourced and frozen, so they stay byte-identical -- verify with:
#     cmp training_<lang>_lemma/model-best/parser/model training_<lang>_sud/model-best/parser/model
#
#   bash scripts/train_sud.sh                 # every language with a feature to train
#   bash scripts/train_sud.sh la sa           # just these
#
# Three env overrides, for pilots that must not clobber a released arm:
#   SUD_FEATS="Shared"          train only these features, whatever feats_for would say
#   SUD_ENCODERS="tree"         encoder per feature (see make_sud_config.py --encoder)
#   SUD_MASKS="none"            candidate mask per feature; the literal `none` means no mask
#   SUD_SUFFIX="_shared_tree"   write training_<lang><suffix>/ instead of training_<lang>_sud/
#   SUD_PREP_ONLY=1             rebuild corpus_<lang>_sud/ and stop, training nothing
#
# ⚠ `corpus_<lang>_sud/` IS SHARED BY EVERY ARM OF A LANGUAGE, and prep rewrites it with only the
# features of the current run. A solo run (`SUD_FEATS=Shared`) therefore leaves a corpus with no
# Subject/Reported gold in it, which a later combined retrain would train on silently unless prep
# runs again first. Restore with `SUD_PREP_ONLY=1 bash scripts/train_sud.sh <lang>`.
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
# `Shared` is the broadest of the three: EVERY treebank here annotates it, so the language list is
# no longer the Subject list. Only ja is left out, with 27 `Yes` in 168 333 tokens -- too sparse to
# learn, the same call made for sa's Subject. This is what brings zh/id/ko into the layer at all;
# id and ko had no SUD MISC pipe of any kind before.
#
# `sa`'s Subject stays EXCLUDED: 142 training instances and 14 in test, which scored F 10.5 trained
# / 12.5 by rule -- neither is shippable, and the test set is too small to tell them apart anyway.
# Its Shared is not sparse (1 758 Yes / 3 299 No), so sa is now in the default list for that alone.
# `zh` and `lzh` train a Subject pipe but ship the RULE instead (see scripts/package_sud.sh).
ALL_LANGS="en yue lzh fa la ar zh id ko sa"
LANGS=${*:-$ALL_LANGS}

# Which SUD features each arm trains a pipe for. `Reported` has no treebank gold anywhere, so its
# target files are the bootstrapped *.reported.conllu written by sud_reported_gold.py; ar and sa
# annotate no Subject, and sa's Subject is too sparse to ship (see above).
feats_for() {
  if [ -n "${SUD_FEATS:-}" ]; then echo "$SUD_FEATS"; return; fi
  case "$1" in
    en|fa|la) echo "Subject Reported Shared" ;;
    ar|sa)    echo "Reported Shared" ;;
    id|ko)    echo "Shared" ;;
    *)        echo "Subject Shared" ;;
  esac
}

# The encoder each feature wants, in the same order (make_sud_config.py --encoder). This is a
# property of the FEATURE, not of the language: `Subject` is local, `Reported` needs the wide
# structural embed, and `Shared` -- a fact about a coordination -- needs the tree layer, which
# reads the head and the head's other dependents directly.
encoder_for() {
  case "$1" in
    Subject)  echo "default" ;;
    Reported) echo "structural" ;;
    Shared)   echo "tree" ;;
    *)        echo "default" ;;
  esac
}

encoders_for() {   # $@ = features
  if [ -n "${SUD_ENCODERS:-}" ]; then echo "$SUD_ENCODERS"; return; fi
  local out=""
  for f in "$@"; do out="$out $(encoder_for "$f")"; done
  echo "$out"
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
    # lzh trains on the PUNCTUATION-RESTORED, rule-merged chain -- the generation its released arm
    # (training_lzh_rm_morph) was trained on. The plain .relabeled_ext files have no PUNCT tokens,
    # so a corpus built from them would not even align with that arm under gold_preproc.
    # LZH_SRC lets the SUD pipes be retrained on the TRADITIONAL-ONLY treebank while the parser
    # underneath stays the both-scripts arm. Kyoto-Both is the same text plus its OpenCC conversion,
    # and dropping the augmentation costs the PARSER 2.4 LAS (79.0 -> 76.57 measured), so the base is
    # kept; but the layers above it want one script so 遠 pools with itself instead of
    # competing with 远.
    lzh) echo "${LZH_SRC:-assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-$2.relabeled_ext.udep_ruled.punct.rulemerged.conllu}" ;;
    fa)  echo "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-$2.relabeled_ext.conllu" ;;
    ar)  echo "assets_ar/SUD_Arabic-PADT/ar_padt-sud-$2.relabeled_ext.conllu" ;;
    la)  echo "assets_la/la_ittbproiel-sud-$2.relabeled_ext.conllu" ;;
    id)  echo "assets_id/SUD_Indonesian-GSD/id_gsd-sud-$2.relabeled_ext.conllu" ;;
    # ko's released arm is the EOJEOL one, trained on the ORIGINAL SUD_Korean-GSD -- not the
    # superseded mecab-morpheme retokenisation, whose corpora carry no Shared at all.
    ko)  echo "assets_ko/SUD_Korean-GSD/ko_gsd-sud-$2.relabeled_ext.conllu" ;;
    # sa trains on Vedic-train + UFAL combined (rebuild_sa_csl_rev.sh); dev/test are Vedic only
    sa)  if [ "$2" = train ]; then echo "corpus_sa_csl_rev/train.csl_rev.conllu"
         else echo "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-$2.csl_rev.conllu"; fi ;;
  esac
}

# The lemma arm to source and freeze. Two languages do not follow the generic name -- and getting
# this wrong is not cosmetic: the MISC layer READS the arm's own predictions (deprel, UPOS, MORPH),
# so a pipe stacked on a different generation than the one that ships silently mismatches its
# inputs. Keep these in step with package_sud.sh.
# lzh's SUD arm sits on the rule-merged punctuation chain, and is NAMED for it so it cannot be
# confused with the pre-punctuation training_lzh_sud -- which is a different model, with a
# different parse and therefore a different coordination mask.
arm_suffix() {
  case "$1" in
    lzh) echo "_rm_sud" ;;
    *)   echo "_sud" ;;
  esac
}

src_model() {
  case "$1" in
    # The rule-merged punctuation arm, and the MORPH storey of it: lzh has no trained lemmatizer
    # any more (han_lemma_lut replaces it at packaging), so _morph is the top of its chain.
    lzh) echo "training_lzh_rm_morph/model-best" ;;
    sa) echo "training_sa_multitask/model-best" ;;
    ko) echo "training_ko_eojeol_lemma/model-best" ;;
    # id's released arm is the SPLIT chain (char segmenter, enclitics separated). The generic
    # training_id_lemma is the older COARSENED arm -- exactly the fall-through that shipped a
    # stale id wheel in v0.1.0. Both read the same CoNLL-U; only the tokeniser differs.
    id) echo "training_id_split_lemma/model-best" ;;
    *)  echo "training_$1_lemma/model-best" ;;
  esac
}

# The base config matching src_model (same exception list).
base_config() {
  case "$1" in
    lzh) echo "configs/config_lzh_rm_morph.cfg" ;;
    sa) echo "configs/config_sa_multitask.cfg" ;;
    ko) echo "configs/config_ko_eojeol_lemma.cfg" ;;
    id) echo "configs/config_id_split_lemma.cfg" ;;
    *)  echo "configs/config_$1_lemma.cfg" ;;
  esac
}

prep() {   # $1=lang  $2...=features to hoist
  local lang=$1; shift
  local feats="$*"
  local out=corpus_${lang}$(arm_suffix "$lang")
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
  local encoders; encoders=$(encoders_for $feats)
  local src; src=$(src_model "$lang")
  local base; base=$(base_config "$lang")
  local suffix=${SUD_SUFFIX:-$(arm_suffix "$lang")}
  local arm=training_${lang}${suffix}
  local cfg=configs/config_${lang}${suffix}.cfg
  local out=corpus_${lang}$(arm_suffix "$lang")

  if [ ! -d "$src" ]; then echo "$lang: source model $src missing -- skip"; return 1; fi
  if [ ! -f "$base" ]; then echo "$lang: base config $base missing -- skip"; return 1; fi

  local tr="$out/train.spacy" dv="$out/dev.spacy"
  [ "$lang" = la ] && { tr="$out/train.d"; dv="$out/dev.d"; }
  if [ ! -e "$tr" ]; then echo "$lang: corpus $tr missing -- skip"; return 1; fi

  # `none` is spelled out because an empty --mask value cannot survive word splitting here.
  local mask_args=()
  if [ -n "${SUD_MASKS:-}" ]; then
    for m in $SUD_MASKS; do [ "$m" = none ] && mask_args+=("") || mask_args+=("$m"); done
    mask_args=(--mask "${mask_args[@]}")
  fi
  $PY scripts/make_sud_config.py "$base" "$src" --feats $feats --encoder $encoders \
      "${mask_args[@]}" --out "$cfg" || return 1
  echo "  $lang: training -> $arm/ (feats:$feats; encoders:$encoders; log: ${arm#training_}.log)"
  $PY -m spacy train "$cfg" $CODE --output "$arm/" \
      --paths.train "$tr" --paths.dev "$dv" > "train_${lang}${suffix}.log" 2>&1
  if [ -f "$arm/model-best/meta.json" ]; then
    $PY - "$arm" <<'EOF'
import json, sys
arm = sys.argv[1]
m = json.load(open(f"{arm}/model-best/meta.json"))["performance"]
keys = [k for k in m if k.startswith("sud_")]
print("   ", arm, {k: round(m[k], 4) for k in sorted(keys)})
EOF
  else
    echo "    $lang: FAILED -- see train_${lang}${suffix}.log"
    tail -5 "train_${lang}${suffix}.log"
  fi
}

for lang in $LANGS; do
  echo "== $lang =="
  feats=$(feats_for "$lang")
  if [ -n "${SUD_PREP_ONLY:-}" ]; then prep "$lang" $feats; else
    prep "$lang" $feats && train_one "$lang" $feats
  fi
done
