#!/usr/bin/env bash
# Rebuild the whole en_gum chain on a retrained BASE arm, into TAG-named directories (nothing existing is overwritten).
# This is the recipe docs/udep-relabel.md records for the 2026-09-27 `udepresolved` release, as a script:
#
#   base (tok2vec + tagger + parser)           training_en_gum_$TAG                (scripts/build_en_vnrulings.sh)
#   morphologiser, frozen base                 training_en_gum_morph_$TAG
#   lemmatiser, frozen base + morph            training_en_gum_lemma_$TAG
#   GUM-punctuation-normalised tagger          training_en_gum_xpos_$TAG
#   xpos tagger grafted into the lemma arm     training_en_gum_lemma_xpos_$TAG     (graft_pipe.py)
#   Subject / Reported / Shared pipes          training_en_gum_sud_xpos_$TAG       (corpus_en_gum_sud_$TAG)
#   tagger conditioned on UPOS + FEATS         training_en_gum_xposwarm_$TAG       (warm-started from the xpos tagger)
#   conditioned tagger moved behind the morphologiser   training_en_gum_sud_xw_$TAG (graft_xpos_tagger.py, checked)
#
# Each stage's config is the previous release's configs/config_en_gum_<stage>_udepresolved.cfg with the arm suffix
# swapped; the recipe is therefore the released one, only the data differs.
#
#     [RESUME=1] bash scripts/build_en_gum_chain.sh [TAG]        # default vnrulings6; package afterwards with
#     SUD_BASE=training_en_gum_sud_xw_$TAG bash scripts/package_sud.sh en_gum
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-vnrulings6}
# CORPUS_TAG: which corpus_en_gum_<tag>/ to read when the BASE arm's TAG names an architecture rather than a
# corpus (e.g. TAG=tinybert on corpus vnrulings6: `CORPUS_TAG=vnrulings6 bash scripts/build_en_gum_chain.sh tinybert`,
# with training_en_gum_tinybert pointing at the base arm). Default: same as TAG, i.e. unchanged behaviour.
CORPUS_TAG=${CORPUS_TAG:-$TAG}
PY=.venv/bin/python
# seg_code registers the TinyBERT tok2vec (when torch is present) so a sourced base with that encoder opens
CODE="--code scripts/seg_code.py"
CE=corpus_en_gum_$CORPUS_TAG/en_ewtgum-sud
TR=$CE-train.relabeled_ext.udep_resolved.spacy; DV=$CE-dev.relabeled_ext.udep_resolved.spacy; TE=$CE-test.relabeled_ext.udep_resolved.spacy
[ -d training_en_gum_$TAG/model-best ] || { echo "base arm training_en_gum_$TAG/model-best missing: run scripts/build_en_vnrulings.sh $TAG"; exit 1; }
stage() { echo "=== $(date +%H:%M:%S) $*"; }

for x in morph lemma xpos sud_xpos xposwarm; do
  sed "s/udepresolved/$TAG/g" configs/config_en_gum_${x}_udepresolved.cfg > configs/config_en_gum_${x}_$TAG.cfg
done
mkdir -p labels_config_en_gum_xposwarm_$TAG && cp labels_config_en_gum_xposwarm_udepresolved/tagger.json labels_config_en_gum_xposwarm_$TAG/

train() {  # $1=stage  $2=train  $3=dev    (RESUME=1 keeps a stage whose model-best already exists)
  if [ -n "${RESUME:-}" ] && [ -d training_en_gum_$1_$TAG/model-best ]; then stage "keep $1 (RESUME)"; return; fi
  stage "train $1"
  PYTHONUNBUFFERED=1 $PY -u -m spacy train configs/config_en_gum_$1_$TAG.cfg $CODE --output training_en_gum_$1_$TAG/ \
    --paths.train "$2" --paths.dev "$3" > train_en_gum_$1_$TAG.log 2>&1
  [ -d training_en_gum_$1_$TAG/model-best ] || { tail -15 train_en_gum_$1_$TAG.log; exit 1; }
}

train morph "$TR" "$DV"
train lemma "$TR" "$DV"
train xpos  "$TR" "$DV"

stage "graft xpos tagger into the lemma arm"
mkdir -p training_en_gum_lemma_xpos_$TAG
$PY scripts/graft_pipe.py training_en_gum_lemma_$TAG/model-best training_en_gum_xpos_$TAG/model-best \
  training_en_gum_lemma_xpos_$TAG/model-best tagger > graft_en_gum_lemma_xpos_$TAG.log 2>&1

stage "SUD corpus (Subject / Reported / Shared hoisted from the reported + udep_resolved hybrid)"
S=corpus_en_gum_sud_$TAG; mkdir -p "$S" "assets_sud_$TAG"
for sp in train dev test; do
  h=assets_sud_$TAG/en_gum-$sp.sud.conllu
  $PY scripts/hoist_sud_gold.py assets/en_ewtgum-sud-$sp.relabeled_ext.reported.udep_resolved.conllu "$h" --keys Subject Reported Shared > /dev/null
  $PY -m spacy convert "$h" "$S/" --converter conllu -n 10 > /dev/null
  mv "$S/en_gum-$sp.sud.spacy" "$S/$sp.spacy"
done
train sud_xpos "$S/train.spacy" "$S/dev.spacy"
train xposwarm "$TR" "$DV"

stage "graft the conditioned tagger behind the morphologiser (checked: parse unchanged, donor tags reproduced)"
$PY scripts/graft_xpos_tagger.py training_en_gum_sud_xpos_$TAG/model-best training_en_gum_xposwarm_$TAG/model-best \
  training_en_gum_sud_xw_$TAG --corpus "$TE" > graft_en_gum_sud_xw_$TAG.log 2>&1
cat graft_en_gum_sud_xw_$TAG.log
echo "CHAIN_DONE"
