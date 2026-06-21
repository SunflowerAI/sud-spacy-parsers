#!/bin/bash
# Watch the three training logs; as each finishes ("Saved pipeline"), evaluate its
# model-best on the test split both WITH --gold-preproc (gold tokens) and WITHOUT
# (raw end-to-end: model re-tokenises). raw token_acc shows the tokeniser match.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

langs="zh_simp id_coarse ko_retok"
test_zh_simp="corpus_zh_simp/zh_gsdsimp-sud-test.spacy"
test_id_coarse="corpus_id_coarse/id_gsd-coarse-test.spacy"
test_ko_retok="corpus_ko_retok/ko_gsd-retok-test.spacy"

done=""
for i in $(seq 1 480); do          # ~4h backstop
  alldone=1
  for L in $langs; do
    case " $done " in *" $L "*) continue;; esac
    if [ -f "train_${L}.log" ] && grep -q "Saved pipeline" "train_${L}.log" 2>/dev/null; then
      eval "T=\$test_${L}"
      echo "######## EVAL $L ########"
      $PY -m spacy evaluate "training_${L}/model-best" "$T" --gold-preproc \
        --output "metrics_${L}_gp.json" >/dev/null 2>&1
      $PY -m spacy evaluate "training_${L}/model-best" "$T" \
        --output "metrics_${L}_raw.json" >/dev/null 2>&1
      $PY - "$L" <<'EOF'
import json, sys
L = sys.argv[1]
def load(f):
    try: return json.load(open(f))
    except Exception: return {}
def row(d): return (f"tok={d.get('token_acc',0):.3f}  TAG={d.get('tag_acc',0):.3f}  "
                    f"UAS={d.get('dep_uas',0):.3f}  LAS={d.get('dep_las',0):.3f}")
gp, raw = load(f"metrics_{L}_gp.json"), load(f"metrics_{L}_raw.json")
print(f"  gold-preproc : {row(gp)}")
print(f"  raw (e2e)    : {row(raw)}")
for tag, d in [("gp", gp), ("raw", raw)]:
    t = (d.get("dep_las_per_type") or {}).get("comp:obl")
    if t: print(f"  comp:obl F ({tag}): {t.get('f',0):.3f}")
EOF
      done="$done $L"
    else
      alldone=0
    fi
  done
  [ "$alldone" = 1 ] && break
  sleep 30
done
echo "######## ALL EVALS DONE ########"
