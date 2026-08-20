#!/bin/bash
# Rebuild the Sanskrit corpus on the MWT-aware sandhi representation (see
# scripts/restructure_sa_csl.py for what the representation is and why).
#
# Supersedes rebuild_sa_csl_rev.sh. The old chain was
#   revert_csl_sandhi (pausa-normalise EVERY token) -> hyphen_to_pipe -> strip_pipe
# and its three steps collapse to one here: `restructure_sa_csl.py` writes final/external tokens
# with their sandhied CSL surface untouched and internal members with the treebank's gold
# `Unsandhied` value, so there is no join marker left on any plain token to convert or strip.
#
# Pairs with `sa.SanskritInputTokenizer.v2`, which reproduces these FORMs at 99.79 % exact
# (100.00 % on the 95 % of tokens that are final/external, since those are identity).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
VED=assets_sa/SUD_Sanskrit-Vedic
UFAL=assets_sa_ufal/SUD_Sanskrit-UFAL
CORP=corpus_sa_csl_mwt

for s in train dev test; do
  $PY scripts/restructure_sa_csl.py "$VED/sa_vedic-sud-$s.sandhi.conllu" \
                                    "$VED/sa_vedic-sud-$s.csl_mwt.conllu"
done
$PY scripts/restructure_sa_csl.py "$UFAL/sa_ufal-sud-test.csl.conllu" \
                                  "$UFAL/sa_ufal-sud-test.csl_mwt.conllu"

# XPOS is the UPOS on every sa token, in all three sources — with one cell in UFAL that arrives
# holding a FEATS value shifted one field left. It has to be fixed HERE, before the convert:
# `spacy convert` reads XPOS into `token.tag_`, so leaving it makes `Compound=Yes` a tagger LABEL,
# which is how it reached the released tagger the first time. Idempotent, XPOS column only; a
# comment telling the next person to run it by hand is not the fix, so the driver runs it.
$PY scripts/normalise_sa_xpos.py "$VED"/sa_vedic-sud-{train,dev,test}.csl_mwt.conllu \
                                 "$UFAL/sa_ufal-sud-test.csl_mwt.conllu"

mkdir -p "$CORP"
# UFAL goes wholly into TRAINING (it is the only classical prose available and is tiny beside
# Vedic); Vedic dev/test stay held out.
cat "$VED/sa_vedic-sud-train.csl_mwt.conllu" "$UFAL/sa_ufal-sud-test.csl_mwt.conllu" \
    > "$CORP/train.csl_mwt.conllu"

$PY -m spacy convert "$CORP/train.csl_mwt.conllu"      "$CORP/" --converter conllu -n 10
$PY -m spacy convert "$VED/sa_vedic-sud-dev.csl_mwt.conllu"  "$CORP/" --converter conllu -n 10
$PY -m spacy convert "$VED/sa_vedic-sud-test.csl_mwt.conllu" "$CORP/" --converter conllu -n 10
ls -l "$CORP"
