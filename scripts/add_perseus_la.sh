#!/bin/bash
# Add the SUD_Latin-Perseus treebank to the Latin training data and rebuild the
# released ext+macron parser end to end.
#
# The Latin model is trained on a plain concatenation of the SUD Latin treebanks
# (each keeps its own sent_ids).  Until now that was ITTB + PROIEL; this adds
# Perseus, which ships only train + test (no dev), so:
#     train = ITTB-train + PROIEL-train + Perseus-train
#     dev   = ITTB-dev   + PROIEL-dev                       (unchanged)
#     test  = ITTB-test  + PROIEL-test  + Perseus-test
# The output prefix stays la_ittbproiel-sud so the rest of the toolchain (configs,
# relabel_ext.py FILES, the macron drivers) is untouched.
#
# Prerequisites:
#   * Docker (macroniser image vedph2020/macronizer:0.1.3 already pulled)
#   * Ollama running with gemma4 (Latin's relabel model)
#   * the three source treebanks extracted (this script extracts/downloads if missing)
#
# Phases (run all, or pass a phase name: merge | macron | relabel | train):
set -euo pipefail
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy
PY=.venv/bin/python
P=la_ittbproiel-sud
A=assets_la

ITTB=$A/SUD_Latin-ITTB/la_ittb-sud
PROIEL=assets_la2/SUD_Latin-PROIEL/la_proiel-sud
PERSEUS=$A/SUD_Latin-Perseus/la_perseus-sud

phase=${1:-all}

ensure_sources() {
  [ -f "$ITTB-train.conllu" ]   || { tar xzf $A/SUD_Latin-ITTB.tgz -C $A/; }
  [ -f "$PROIEL-train.conllu" ] || { tar xzf assets_la2/SUD_Latin-PROIEL.tgz -C assets_la2/; }
  if [ ! -f "$PERSEUS-train.conllu" ]; then
    curl -sSLo $A/SUD_Latin-Perseus.tgz https://grew.fr/download/SUD_2.18/SUD_Latin-Perseus.tgz
    tar xzf $A/SUD_Latin-Perseus.tgz -C $A/
  fi
}

do_merge() {
  echo "### MERGE: build $P-{train,dev,test}.conllu (ITTB + PROIEL + Perseus)"
  ensure_sources
  cat "$ITTB-train.conllu" "$PROIEL-train.conllu" "$PERSEUS-train.conllu" > "$A/$P-train.conllu"
  cat "$ITTB-dev.conllu"   "$PROIEL-dev.conllu"                            > "$A/$P-dev.conllu"
  cat "$ITTB-test.conllu"  "$PROIEL-test.conllu"  "$PERSEUS-test.conllu"   > "$A/$P-test.conllu"
  # Blank Perseus's (incompatible, sparse 9-position) XPOS so it does not confuse the
  # tagger; Perseus is the tail of each split, so blank from the ITTB+PROIEL count on.
  # FORM/UPOS/dependencies are untouched, so macron + relabel below carry the blank through.
  itp_train=$(( $(grep -c '^# sent_id' "$ITTB-train.conllu") + $(grep -c '^# sent_id' "$PROIEL-train.conllu") ))
  itp_test=$((  $(grep -c '^# sent_id' "$ITTB-test.conllu")  + $(grep -c '^# sent_id' "$PROIEL-test.conllu") ))
  $PY scripts/blank_perseus_xpos.py "$A/$P-train.conllu" "$A/$P-train.conllu" --from-sent "$itp_train"
  $PY scripts/blank_perseus_xpos.py "$A/$P-test.conllu"  "$A/$P-test.conllu"  --from-sent "$itp_test"
  for s in train dev test; do
    printf "  %-5s sentences: " "$s"; grep -c '^# sent_id' "$A/$P-$s.conllu"
  done
}

do_macron() {
  echo "### MACRON: start macroniser container + macronise the merged files"
  if ! docker ps --format '{{.Names}}' | grep -qx macronizer; then
    docker rm -f macronizer 2>/dev/null || true
    docker run -d --name macronizer --platform linux/amd64 \
      -p 51234:105 -e PYTHONUNBUFFERED=1 vedph2020/macronizer:0.1.3
    echo "  waiting for the macroniser API ..."
    for _ in $(seq 1 60); do
      curl -s -o /dev/null -X POST http://localhost:51234/macronize \
        -H 'Content-Type: application/json' -d '{"text":"Gallia"}' && break || sleep 2
    done
  fi
  for s in train dev test; do
    $PY scripts/macronise_la.py "$A/$P-$s.conllu" "$A/$P-$s.macron.conllu"
  done
  docker stop macronizer >/dev/null && docker rm macronizer >/dev/null
  echo "  macroniser container stopped + removed"
}

do_relabel() {
  echo "### RELABEL: ext relabel (gemma4) + transfer macrons onto ext deprels"
  OLLAMA_MODEL=gemma4:latest $PY scripts/relabel_ext.py --lang la
  for s in train dev test; do
    $PY scripts/transfer_macrons.py \
      "$A/$P-$s.macron.conllu" "$A/$P-$s.relabeled_ext.conllu" \
      "$A/$P-$s.relabeled_ext.macron.conllu"
  done
}

do_train() {
  echo "### TRAIN: ext model (+metrics_la_ext.json) then the ext+macron union release"
  bash scripts/relabel_retrain_ext_new.sh la
  bash scripts/train_la_ext_macron.sh
}

case "$phase" in
  merge)   do_merge ;;
  macron)  do_macron ;;
  relabel) do_relabel ;;
  train)   do_train ;;
  all)     do_merge; do_macron; do_relabel; do_train ;;
  *) echo "unknown phase: $phase (merge|macron|relabel|train|all)"; exit 1 ;;
esac
echo "DONE: $phase"
