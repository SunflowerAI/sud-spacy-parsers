#!/usr/bin/env bash
# Seed spread of the en_gum BASE arm on two corpora, to tell seed noise from the effect of a relabelling.
#   bash scripts/seed_check_en_gum.sh NEWTAG OLDCORPUSDIR SEED...      e.g.  vnrulings6 corpus_en_gum_udepresolved 1 2
# Trains configs/config.cfg with --system.seed on each corpus (in parallel), scores each model on its own test split.
set -uo pipefail
cd "$(dirname "$0")/.."
NEW=$1; OLDC=$2; shift 2
PY=.venv/bin/python
NC=corpus_en_gum_$NEW/en_ewtgum-sud
OC=$OLDC/en_ewtgum-sud
run() {  # $1=label $2=corpus prefix $3=seed
  local out=training_en_gum_seed_$1_s$3
  PYTHONUNBUFFERED=1 $PY -u -m spacy train configs/config.cfg --output $out/ --system.seed $3 \
    --paths.train "$2-train.relabeled_ext.udep_resolved.spacy" --paths.dev "$2-dev.relabeled_ext.udep_resolved.spacy" > train_en_gum_seed_$1_s$3.log 2>&1
  $PY -m spacy evaluate $out/model-best "$2-test.relabeled_ext.udep_resolved.spacy" --output metrics/en_gum/metrics_en_gum_seed_$1_s$3.json > /dev/null 2>&1
}
for s in "$@"; do run new "$NC" $s & run old "$OC" $s & done
wait
echo SEEDS_DONE
