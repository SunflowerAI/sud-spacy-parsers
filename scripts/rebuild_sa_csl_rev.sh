#!/usr/bin/env bash
# ⚠ SUPERSEDED by scripts/rebuild_sa_csl_mwt.sh. This builds the PAUSA-NORMALISED representation
# (`corpus_sa_csl_rev`, `*.csl_rev.conllu`); the released sa arm is trained on the DCS/MWT one.
# The two differ in FORM and tokenisation, and the csl_rev corpus is additionally UNRELABELLED
# (`udep` on 7.89 % of tokens against 0.00 %). Kept for the historical record only -- see the
# BUILD PROVENANCE table in docs/sanskrit.md before rebuilding anything Sanskrit.
# Rebuild the CSL-reverted Sanskrit corpus (corpus_sa_csl_rev/) from the sandhied-CSL sources,
# so the training data matches whatever scripts/sa_tokenizer.py:desandhi_csl currently produces.
# Run this after changing desandhi_csl (e.g. the pre-pausal normalisation), then retrain
# base -> morph -> lemma and repackage.
#
# Pipeline per source: revert_csl_sandhi (undo sandhi -> pre-pausal wordforms, hyphen-marked
# compound members) -> hyphen_to_pipe_sa (compound join - -> |, CSL convention) -> strip_pipe_sa
# (drop the trailing | on plain member FORMs; the Compound=Yes FEAT + n-m MWT range still record
# the grouping). Vedic-train + UFAL are concatenated for the training split; Vedic dev/test are the
# eval splits (UFAL is entirely in training, held out nowhere). Then spacy convert -n 10.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VED=assets_sa/SUD_Sanskrit-Vedic
UFAL=assets_sa_ufal/SUD_Sanskrit-UFAL
CORP=corpus_sa_csl_rev

prep() {  # <in-sandhied-conllu> <out-csl_rev-conllu>
  local src="$1" out="$2"
  $PY scripts/revert_csl_sandhi.py "$src" "$out"
  $PY scripts/hyphen_to_pipe_sa.py "$out" "$out" --check
  $PY scripts/strip_pipe_sa.py     "$out" "$out" --check
}

echo ">> reverting sandhi (Vedic train/dev/test + UFAL) with the current desandhi_csl"
for split in train dev test; do
  prep "$VED/sa_vedic-sud-$split.sandhi.conllu" "$VED/sa_vedic-sud-$split.csl_rev.conllu"
done
prep "$UFAL/sa_ufal-sud-test.csl.conllu" "$UFAL/sa_ufal-sud-test.csl_rev.conllu"

echo ">> combining Vedic-train + UFAL -> $CORP/train.csl_rev.conllu"
mkdir -p "$CORP"
cat "$VED/sa_vedic-sud-train.csl_rev.conllu" "$UFAL/sa_ufal-sud-test.csl_rev.conllu" \
    > "$CORP/train.csl_rev.conllu"

echo ">> converting to .spacy (whole-doc, -n 10)"
$PY -m spacy convert "$CORP/train.csl_rev.conllu"             "$CORP/" --converter conllu -n 10
$PY -m spacy convert "$VED/sa_vedic-sud-dev.csl_rev.conllu"   "$CORP/" --converter conllu -n 10
$PY -m spacy convert "$VED/sa_vedic-sud-test.csl_rev.conllu"  "$CORP/" --converter conllu -n 10
echo ">> done."
