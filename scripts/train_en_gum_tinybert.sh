#!/usr/bin/env bash
# Train the EWT+GUM base arm with a TinyBERT-4L-312D tok2vec (sud.TinyBertTok2Vec.v1) on the same corpus as
# the shipped-recipe baseline (corpus_en_gum_vnrulings6, LAS 81.71 seed 0), and score it on that corpus's test split.
#   bash scripts/train_en_gum_tinybert.sh ARM [SEED]   (ARM = the suffix of configs/config_en_gum_tinybert[_ARM].cfg)
# mix = learned softmax mix of all 5 hidden states; last = last layer only (the control);
# hash_last = last layer + the base arm's hash-embed CNN, concatenated (restores the case an uncased encoder drops).
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?mix|last}; SEED=${2:-0}
PY=.venv/bin/python
CFG=configs/config_en_gum_tinybert$([ "$ARM" = mix ] || echo _$ARM).cfg
OUT=training_en_gum_tinybert_${ARM}_s$SEED
C=corpus_en_gum_vnrulings6/en_ewtgum-sud
mkdir -p metrics/en_gum
PYTHONUNBUFFERED=1 $PY -u -m spacy train $CFG --code scripts/sud_tinybert_embed.py --output $OUT/ --system.seed $SEED \
  --paths.train $C-train.relabeled_ext.udep_resolved.spacy --paths.dev $C-dev.relabeled_ext.udep_resolved.spacy \
  > train_en_gum_tinybert_${ARM}_s$SEED.log 2>&1
$PY -m spacy evaluate $OUT/model-best $C-test.relabeled_ext.udep_resolved.spacy --code scripts/sud_tinybert_embed.py \
  --output metrics/en_gum/metrics_en_gum_tinybert_${ARM}_s$SEED.json > eval_en_gum_tinybert_${ARM}_s$SEED.log 2>&1
echo DONE $ARM $SEED
