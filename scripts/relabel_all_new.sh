#!/bin/bash
# Phase 4 relabelling (Ollama-heavy, ~hours, resumable via per-language caches) for the five
# new languages, each with its Phase-3 benchmark-winning model. Baseline (verb-ADP) relabel for
# the prepositional langs, then extended-scope relabel for all five (sa is case-based: ext only).
# Grouped by model to minimise Ollama reloads. Retraining is a separate step (relabel_retrain_*).
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
declare -A MODEL=( [fa]=qwen3:8b [lzh]=qwen3:8b [sa]=qwen3:8b [ar]=gemma4:latest [la]=gemma4:latest )

echo "===================== BASELINE (verb-ADP) RELABEL ====================="
for lang in lzh fa ar la; do
  echo "### baseline relabel $lang  (model=${MODEL[$lang]}) ###"
  OLLAMA_MODEL=${MODEL[$lang]} $PY scripts/lang_relabel.py --lang $lang
done

echo "===================== EXTENDED-SCOPE RELABEL ====================="
for lang in lzh sa fa ar la; do
  echo "### ext relabel $lang  (model=${MODEL[$lang]}) ###"
  OLLAMA_MODEL=${MODEL[$lang]} $PY scripts/relabel_ext.py --lang $lang
done
echo "===================== ALL RELABEL DONE ====================="
