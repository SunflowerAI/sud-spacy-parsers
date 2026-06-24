#!/bin/bash
# Extended-scope relabel retrain + eval for the five new languages (scripts/relabel_ext.py
# output: *.relabeled_ext.conllu). gold-preproc eval throughout (comparable to metrics_<lang>.json).
# Run relabel_ext.py for each language first. Prints base vs verb-rl vs extended comp:obl F + LAS.
# (Sanskrit has no verb-ADP baseline relabel — its signal is case-based — so verb-rl == base.)
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
declare -A DIR=( [fa]=assets_fa/SUD_Persian-PerDT [ar]=assets_ar/SUD_Arabic-PADT [la]=assets_la \
                 [sa]=assets_sa/SUD_Sanskrit-Vedic [lzh]=assets_lzh/SUD_Classical_Chinese-Kyoto \
                 [ja]=assets_ja/SUD_Japanese-GSD )
declare -A PREF=( [fa]=fa_perdt-sud [ar]=ar_padt-sud [la]=la_ittbproiel-sud \
                  [sa]=sa_vedic-sud [lzh]=lzh_kyoto-sud [ja]=ja_gsd-sud )
declare -A CODE=( [lzh]="--code scripts/lzh_tokenizer.py" )
declare -A RL=(   [fa]=metrics_fa_rl.json [ar]=metrics_ar_rl.json [la]=metrics_la_rl.json \
                  [sa]=metrics_sa.json [lzh]=metrics_lzh_rl.json [ja]=metrics_ja_rl.json )

for lang in "$@"; do
  p=${PREF[$lang]}; dir=${DIR[$lang]}; code=${CODE[$lang]}
  if [ ! -f "$dir/$p-test.relabeled_ext.conllu" ]; then
    echo "!! $lang: $dir/$p-test.relabeled_ext.conllu missing — run relabel_ext.py first"; continue
  fi
  echo "############## CONVERT $lang ##############"
  mkdir -p corpus_${lang}_ext
  for split in train dev test; do
    $PY -m spacy convert "$dir/$p-$split.relabeled_ext.conllu" corpus_${lang}_ext/ --converter conllu -n 10 \
      2>&1 | grep -oE 'Generated output file \([0-9]+ documents\)' | sed "s/^/  $lang $split: /"
  done
  echo "############## RETRAIN $lang ##############"
  $PY -m spacy train configs/config_$lang.cfg $code --output training_${lang}_ext/ \
    --paths.train corpus_${lang}_ext/$p-train.relabeled_ext.spacy \
    --paths.dev   corpus_${lang}_ext/$p-dev.relabeled_ext.spacy > train_${lang}_ext.log 2>&1
  if [ ! -d training_${lang}_ext/model-best ]; then echo "!! $lang retrain FAILED"; tail -6 train_${lang}_ext.log; continue; fi
  echo "############## EVAL $lang (extended) ##############"
  $PY -m spacy evaluate training_${lang}_ext/model-best corpus_${lang}_ext/$p-test.relabeled_ext.spacy \
    $code --gold-preproc --output metrics_${lang}_ext.json >/dev/null 2>&1
  $PY - "$lang" "metrics_${lang}_ext.json" "${RL[$lang]}" "metrics_${lang}.json" <<'EOF'
import json, sys
lang, ext_f, rl_f, base_f = sys.argv[1:5]
def load(f):
    try: return json.load(open(f))
    except Exception: return {}
def cf(d):
    t=(d.get("dep_las_per_type") or {}).get("comp:obl"); return t["f"] if t else 0.0
def las(d): return d.get("dep_las", 0.0)
ext, rl, base = load(ext_f), load(rl_f), load(base_f)
print(f"  {lang}:  LAS  base={las(base):.3f}  verb-rl={las(rl):.3f}  EXT={las(ext):.3f}")
print(f"  {lang}:  comp:obl F  base={cf(base):.3f}  verb-rl={cf(rl):.3f}  EXT={cf(ext):.3f}")
EOF
done
echo "================ EXT RETRAIN DONE: $* ================"
