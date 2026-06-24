#!/bin/bash
# Phase 3: benchmark comp/mod prompts for the five new languages, qwen3:8b vs gemma4, on the
# confident gold (gold_<lang>.jsonl). Writes bench_<lang>_<model>.log; pick the winner per lang.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python
PERCLASS=${PERCLASS:-60}

for lang in "$@"; do
  for model in qwen3:8b gemma4:latest; do
    tag=$(echo "$model" | tr ':' '_')
    echo "######## BENCH $lang / $model (per-class $PERCLASS) ########"
    OLLAMA_MODEL=$model $PY scripts/lang_bench.py --lang $lang --per-class $PERCLASS \
      2>&1 | tee bench_${lang}_${tag}.log | grep -E "===|acc="
  done
done
echo "BENCH DONE: $*"
