#!/usr/bin/env bash
# ⚠ SUPERSEDED GENERATION. This driver reads `corpus_sa_csl_rev` and chains
# `training_sa_csl_rev*` -- the pausa-normalised representation and the freeze-recipe arms
# built on it. Neither is what ships: the sa wheel is a JOINT MULTI-TASK arm
# (`training_sa_mp2_sub_s1`, `SA_BASE` in package_sud.sh) trained on the DCS/MWT
# representation. Kept as the experimental record; do not treat its output as current.
# CLAUDE.md lists `rebuild_sa_csl_rev.sh` under "Superseded but kept"; the authority on
# which sa corpus feeds what is the BUILD PROVENANCE table in docs/sanskrit.md.
# Retrain the whole Sanskrit chain on the widened affix windows (PREFIX 1->3, SUFFIX 3->6) and the
# tokeniser-supplied Compound=Yes input feature. All three stages must be retrained together: the
# lexeme attributes are inputs to EVERY component, so a model trained on the old windows cannot be
# mixed with one trained on the new ones.
#
# Trains into *2 directories, leaving training_sa_csl_rev / _morph / _lemma untouched
# (they were the released chain WHEN THIS WAS WRITTEN; they are not any more -- see the banner)
# for comparison. The `source` paths baked into the morph/lemma configs point at the released dirs,
# so they are overridden on the command line to chain the new arms together.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CODE=scripts/seg_code.py                 # sa_tokenizer (lex attrs + tokenizer) + CompoundCorpus reader
C=corpus_sa_csl_rev
TRAIN=$C/train.csl_rev.spacy
DEV=$C/sa_vedic-sud-dev.csl_rev.spacy
TEST=$C/sa_vedic-sud-test.csl_rev.spacy

echo "=== [1/4] base (tok2vec + tagger + parser)"
$PY -m spacy train configs/config_sa.cfg --output training_sa_csl_rev3/ --code $CODE \
  --paths.train $TRAIN --paths.dev $DEV 2>&1 | tee train_sa_csl_rev3.log

echo "=== [2/4] morphologizer (frozen base + dedicated encoder)"
$PY -m spacy train configs/config_sa_morph.cfg --output training_sa_morph3/ --code $CODE \
  --paths.train $TRAIN --paths.dev $DEV \
  --components.tok2vec.source training_sa_csl_rev3/model-best \
  --components.tagger.source  training_sa_csl_rev3/model-best \
  --components.parser.source  training_sa_csl_rev3/model-best 2>&1 | tee train_sa_morph3.log

echo "=== [3/4] lemmatizer (frozen base+morph, annotating_components=[morphologizer])"
$PY -m spacy train configs/config_sa_lemma.cfg --output training_sa_lemma3/ --code $CODE \
  --paths.train $TRAIN --paths.dev $DEV \
  --components.tok2vec.source       training_sa_morph3/model-best \
  --components.tagger.source        training_sa_morph3/model-best \
  --components.parser.source        training_sa_morph3/model-best \
  --components.morphologizer.source training_sa_morph3/model-best 2>&1 | tee train_sa_lemma3.log

echo "=== [4/4] evaluate on the Vedic test set (gold-preproc, Compound supplied)"
$PY scripts/eval_sa_compound.py training_sa_lemma3/model-best $TEST --out metrics_sa_compound_only.json
echo "=== done"
