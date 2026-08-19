#!/usr/bin/env bash
# A REAL tokeniser for Tamil, so the wheel can be fed ordinary Tamil text.
#
# WHY. SUD_Tamil-TTB splits 835 orthographic words into 1 781 syntactic words, and 94.2 % of those
# splits REWRITE at the seam rather than cutting cleanly (நிலையத்துக்குக்கான -> நிலையத்துக்குக்க் + ஆன).
# spaCy's rule tokeniser splits on whitespace and punctuation only, so it can never produce the
# treebank's tokens, and the ceiling on strict token F without a splitter is ~0.92.
#
# THIS IS INVISIBLE IN EVERY PARSING FIGURE, which is the same trap `train_lzh_charseg.sh` records:
# `--gold-preproc` bypasses the tokeniser at evaluation and `sud.GoldTokCorpus.v1` makes the parser
# segmenter-agnostic, so nothing in LAS/UAS/TAG touches tokenisation. Strict token F on RAW text is
# the only number that sees it, and it is what this script reports.
#
# HOW. scripts/ta_sandhi.py turns the rewriting into plain segmentation: decompose every akṣara into
# consonant + virāma + independent vowel, and the gold parts become a clean cut of the decomposed
# surface on 842 of 878 ranges (95.90 %, against 5.8 % on the raw surface), with the round trip exact
# on all 13 043 tokens. So the existing machinery -- make_seg_pairs.py + train_samhita.py +
# sud.CharSegTokenizer.v1, which serve zh/id/lzh -- works unchanged, on a decomposed string.
#
# ⚠ THE DECOMPOSED FORM IS NEVER STORED IN A PARSER CORPUS. Only data_seg_ta/ is decomposed; the
# .spacy corpora keep the treebank's real FORMs. The tokeniser decomposes its input and recomposes
# its output, and records that regime in its bundled vocab.json (standing hazard 10).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_ta

mkdir -p data_seg_ta assets_ta/decomp

echo "### 0. the orthography self-check (round trip must be exact, or nothing below is meaningful)"
$PY scripts/ta_sandhi.py --check || exit 1

echo
echo "### 1. decomposed copies, for the segmenter only"
for s in train dev test; do
  $PY scripts/ta_sandhi.py --conllu "$A/ta_ttb_mwtt-sud-${s}.conllu" \
      --out "$A/decomp/ta-${s}.conllu" | sed 's/^/  /'
done

echo
echo "### 2. character-tagger training pairs, per whitespace chunk"
for s in train dev test; do
  $PY scripts/make_seg_pairs.py "$A/decomp/ta-${s}.conllu" "data_seg_ta/${s}.jsonl" \
      --min-chunk 1 2>&1 | sed 's/^/  /'
done

echo
echo "### 3. train"
# Tamil's character inventory after decomposition is small (37 letters plus the virāma and the
# independent vowels), so the work is in the CONTEXT rather than the embedding -- depth 6 as for
# lzh/id, width 64.
$PY scripts/train_samhita.py data_seg_ta/train.jsonl data_seg_ta/dev.jsonl \
    models/ta_seg_char --width 64 --depth 6 --epochs 40 2>&1 | tail -8

echo
echo "### 4. STRICT TOKEN F on raw text -- the only number that sees a tokeniser"
# Not character accuracy: a model that never splits anything scores ~97 % of characters right and
# would look like success while producing none of the treebank's multiword tokens.
$PY scripts/eval_ta_tokenizer.py --model models/ta_seg_char \
    --conllu "$A/ta_ttb_mwtt-sud-test.conllu"
