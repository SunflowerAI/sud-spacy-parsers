#!/bin/bash
# Retrain + evaluate on the EXTENDED-scope relabelling (scripts/relabel_ext.py output:
# *.relabeled_ext.conllu). Plain tokenisation + gold-preproc eval for zh/ko/id (comparable to
# metrics_<lang>_rl.json), plain for en. Assumes relabel_ext.py has already produced the
# *.relabeled_ext.conllu files (run it first). Writes corpus_*_ext / training_*_ext /
# metrics_*_ext.json and prints baseline vs verb-scope-rl vs extended comp:obl F + LAS.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

# lang -> conllu dir, file prefix, config, whether eval needs --gold-preproc, baseline-rl metrics
declare -A DIR=( [zh]=assets_zh/SUD_Chinese-GSDSimp [ko]=assets_ko/SUD_Korean-GSD \
                 [id]=assets_id/SUD_Indonesian-GSD  [en]=assets )
declare -A PREF=( [zh]=zh_gsdsimp-sud [ko]=ko_gsd-sud [id]=id_gsd-sud [en]=en_sud )
declare -A CFG=(  [zh]=configs/config_zh.cfg [ko]=configs/config_ko.cfg \
                  [id]=configs/config_id.cfg [en]=configs/config.cfg )
declare -A GP=(   [zh]=--gold-preproc [ko]=--gold-preproc [id]=--gold-preproc [en]="" )
declare -A RL=(   [zh]=metrics_zh_rl.json [ko]=metrics_ko_rl.json \
                  [id]=metrics_id_rl.json [en]=metrics_relabeled.json )
declare -A BASE=( [zh]=metrics_zh.json [ko]=metrics_ko.json \
                  [id]=metrics_id.json [en]=metrics.json )

for lang in en zh ko id; do
  p=${PREF[$lang]}; dir=${DIR[$lang]}
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
  $PY -m spacy train ${CFG[$lang]} --output training_${lang}_ext/ \
    --paths.train corpus_${lang}_ext/$p-train.relabeled_ext.spacy \
    --paths.dev   corpus_${lang}_ext/$p-dev.relabeled_ext.spacy > train_${lang}_ext.log 2>&1
  if [ ! -d training_${lang}_ext/model-best ]; then echo "!! $lang retrain FAILED"; tail -6 train_${lang}_ext.log; continue; fi
  echo "############## EVAL $lang (extended) ##############"
  $PY -m spacy evaluate training_${lang}_ext/model-best corpus_${lang}_ext/$p-test.relabeled_ext.spacy \
    ${GP[$lang]} --output metrics_${lang}_ext.json >/dev/null 2>&1
  $PY - "$lang" "metrics_${lang}_ext.json" "${RL[$lang]}" "${BASE[$lang]}" <<'EOF'
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
echo "================ EXT RETRAIN DONE ================"
