#!/bin/bash
# Relabel udep->comp:obl/mod (cached qwen3:8b pipeline) at original tokenisation,
# transfer the relabels through the retokenise/coarsen transforms (which preserve
# deprels), retrain the retokenised models, and evaluate. zh = GSDSimp+pkuseg
# (word-level, no transform), id = coarsened, ko = morpheme functional-head.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

echo "================ RELABEL (cached) ================"
for lang in id zh ko; do
  echo "--- $lang ---"; $PY scripts/lang_relabel.py --lang $lang 2>&1 | tail -1
done

echo "================ TRANSFORM + CONVERT ================"
# zh: relabeled GSDSimp is already word-level (pkuseg uses gold tokens at train) -> convert
mkdir -p corpus_zh_simp_rl
for s in train dev test; do
  $PY -m spacy convert assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-$s.relabeled.conllu \
    corpus_zh_simp_rl/ --converter conllu -n 10 >/dev/null 2>&1
done
# id: coarsen the relabeled file
mkdir -p assets_id_coarse_rl corpus_id_coarse_rl
for s in train dev test; do
  $PY scripts/coarsen_id.py assets_id/SUD_Indonesian-GSD/id_gsd-sud-$s.relabeled.conllu \
    assets_id_coarse_rl/id_gsd-coarse-$s.relabeled.conllu
  $PY -m spacy convert assets_id_coarse_rl/id_gsd-coarse-$s.relabeled.conllu \
    corpus_id_coarse_rl/ --converter conllu -n 10 >/dev/null 2>&1
done
# ko: retokenise the relabeled file (comp:obl/mod transfers to the case particle)
mkdir -p assets_ko_retok_rl corpus_ko_retok_rl
for s in train dev test; do
  $PY scripts/retokenize.py --lang ko assets_ko/SUD_Korean-GSD/ko_gsd-sud-$s.relabeled.conllu \
    assets_ko_retok_rl/ko_gsd-retok-$s.relabeled.conllu
  $PY -m spacy convert assets_ko_retok_rl/ko_gsd-retok-$s.relabeled.conllu \
    corpus_ko_retok_rl/ --converter conllu -n 10 >/dev/null 2>&1
done

echo "================ RETRAIN (relabeled) ================"
$PY -m spacy train configs/config_zh.cfg --output training_zh_simp_rl/ \
  --paths.train corpus_zh_simp_rl/zh_gsdsimp-sud-train.relabeled.spacy \
  --paths.dev   corpus_zh_simp_rl/zh_gsdsimp-sud-dev.relabeled.spacy   > train_zh_simp_rl.log 2>&1
echo "zh_rl done"
$PY -m spacy train configs/config_id.cfg --output training_id_coarse_rl/ \
  --paths.train corpus_id_coarse_rl/id_gsd-coarse-train.relabeled.spacy \
  --paths.dev   corpus_id_coarse_rl/id_gsd-coarse-dev.relabeled.spacy   > train_id_coarse_rl.log 2>&1
echo "id_rl done"
$PY -m spacy train configs/config_ko.cfg --output training_ko_retok_rl/ \
  --paths.train corpus_ko_retok_rl/ko_gsd-retok-train.relabeled.spacy \
  --paths.dev   corpus_ko_retok_rl/ko_gsd-retok-dev.relabeled.spacy     > train_ko_retok_rl.log 2>&1
echo "ko_rl done"

echo "================ EVAL (relabeled: gold-preproc vs raw) ================"
test_zh_simp="corpus_zh_simp_rl/zh_gsdsimp-sud-test.relabeled.spacy"
test_id_coarse="corpus_id_coarse_rl/id_gsd-coarse-test.relabeled.spacy"
test_ko_retok="corpus_ko_retok_rl/ko_gsd-retok-test.relabeled.spacy"
for L in zh_simp id_coarse ko_retok; do
  eval "T=\$test_${L}"
  [ -d "training_${L}_rl/model-best" ] || { echo "!! $L retrain FAILED"; tail -4 train_${L}_rl.log; continue; }
  echo "######## $L (relabeled) ########"
  $PY -m spacy evaluate "training_${L}_rl/model-best" "$T" --gold-preproc --output "metrics_${L}_rl_gp.json"  >/dev/null 2>&1
  $PY -m spacy evaluate "training_${L}_rl/model-best" "$T"                 --output "metrics_${L}_rl_raw.json" >/dev/null 2>&1
  $PY - "$L" <<'EOF'
import json, sys
L = sys.argv[1]
def load(f):
    try: return json.load(open(f))
    except Exception: return {}
def comp(d):
    t=(d.get("dep_las_per_type") or {}).get("comp:obl"); return t.get("f",0) if t else 0
gp, raw = load(f"metrics_{L}_rl_gp.json"), load(f"metrics_{L}_rl_raw.json")
old = load(f"metrics_{L}_gp.json")
print(f"  gold-preproc: LAS={gp.get('dep_las',0):.3f}  comp:obl F={comp(gp):.3f}   "
      f"(non-relabeled LAS={old.get('dep_las',0):.3f} comp:obl F={comp(old):.3f})")
print(f"  raw (e2e)   : LAS={raw.get('dep_las',0):.3f}  tok={raw.get('token_acc',0):.3f}  comp:obl F={comp(raw):.3f}")
EOF
done
echo "================ ALL RELABEL+RETRAIN DONE ================"
