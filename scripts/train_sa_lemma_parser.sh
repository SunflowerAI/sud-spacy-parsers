#!/usr/bin/env bash
# Can a Sanskrit parser be built on LEMMA identity + DECOMPOSED morphology instead of surface forms?
#
# Motivation, measured on corpus_sa_csl_mwt/train (163 802 tokens): 38 883 form types / 61.6 % hapax
# against 12 024 lemma types / 44.2 %. Under the DCS representation a standalone token keeps its
# SANDHIED surface, so the parser's lexical channel spends most of its capacity on alternations that
# carry no syntax. NORM differs from LEMMA on 74.5 % of tokens.
#
# Both earlier parser experiments on this question (config_sa_morphfirst.cfg 2026-07-31,
# config_la_morphfirst.cfg 2026-08-04) fed morphology in as spaCy's `MORPH` column — ONE hash of the
# whole FEATS bundle, so `Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` arrived as unrelated
# symbols. sud.MultiHashEmbedFeats.v1 (one table per feature) postdates both and had only ever been
# attached to a tagger. For a parser the distinction should matter more than it did there: agreement
# and case-government are relations BETWEEN tokens, and two bundles cannot be compared for shared
# case when case has no separate dimension.
#
# Run order matters only in that the two arms saturate the machine one at a time (10 cores; a single
# spaCy train already puts the load average at ~9).
#
#   [0] norm    NORM := the transducer's padapāṭha. No architecture change, no pipeline reordering:
#               sud_unsandhi lives in the TOKENISER, so its output precedes every component. The
#               cheap half of the idea, and the only one that could ship without surgery.
#   [1] oracle  gold LEMMA + gold per-feature FEATS in the embed. NOT SHIPPABLE — it exists to bound
#               the expensive half in one run, so the pipeline surgery is only paid for if the
#               ceiling justifies it.
#
# Still to come if [1] clears the bar (see the reply that commissioned this): predicted lemma+morph
# via frozen upstream components and annotating_components, with BOTH a capacity control and a
# block-MORPH control, ≥3 seeds, decided on the 18 161-token Vedic test with UFAL reported apart.
set -euo pipefail
cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE=scripts/seg_code.py
C=corpus_sa_csl_mwt
TEST=$C/sa_vedic-sud-test.csl_mwt.spacy

echo "=== [0a] build the padapāṭha-NORM corpus (predicted, never gold Unsandhied)"
$PY scripts/make_norm_corpus.py --transducer training_sa_mwt_unsandhi/model-best \
    --in $C --out corpus_sa_mwt_norm --report

echo "=== [0b] parser on padapāṭha NORM"
N=corpus_sa_mwt_norm
$PY -u -m spacy train configs/config_sa_mwt_norm.cfg --output training_sa_mwt_norm/ --code $CODE \
  --paths.train $N/train.csl_mwt.spacy \
  --paths.dev   $N/sa_vedic-sud-dev.csl_mwt.spacy 2>&1 | tee train_sa_mwt_norm.log

echo "=== [1] ORACLE: gold LEMMA + gold per-feature FEATS"
$PY -u -m spacy train configs/config_sa_mwt_oracle.cfg --output training_sa_mwt_oracle/ --code $CODE \
  --paths.train $C/train.csl_mwt.spacy \
  --paths.dev   $C/sa_vedic-sud-dev.csl_mwt.spacy 2>&1 | tee train_sa_mwt_oracle.log

echo "=== [2] score on the Vedic test, each arm through the reader it was TRAINED through"
# Scoring an arm through the wrong reader deletes one of its inputs — that is what eval_sa_compound
# exists to prevent, and the --reader flag extends it to the two new channels.
$PY scripts/eval_sa_compound.py training_sa_mwt/model-best        $TEST --reader compound \
    --out metrics/sa/metrics_sa_mwt_baseline.json
$PY scripts/eval_sa_compound.py training_sa_mwt_norm/model-best   $N/sa_vedic-sud-test.csl_mwt.spacy \
    --reader norm   --out metrics/sa/metrics_sa_mwt_norm.json
$PY scripts/eval_sa_compound.py training_sa_mwt_oracle/model-best $TEST --reader oracle \
    --out metrics/sa/metrics_sa_mwt_oracle.json
echo "=== done"
