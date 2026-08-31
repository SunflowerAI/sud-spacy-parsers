#!/usr/bin/env bash
# Fetch the fastText ALIGNED source spaces for the generic parser's lexical channel (v3).
#
# WHY ONLY THE PUBLISHED ALIGNED SET. fastText publishes 44 languages already rotated into one
# shared space (`vectors-aligned`, RCSLS to English). Those need NO rotation fitting -- route "pre"
# in `align_vectors.py` -- which is the whole reason this arm is cheap. The other 48 training
# languages have a source space but no published alignment, and each would need its own Procrustes
# fit against a bilingual dictionary; v1's fit report shows what that route is worth, and it varies
# from 0.62 to 0.07 hit@1. Extending coverage that way is a separate, later decision.
#
# ⚠ TRAIN AND TEST LANGUAGES ARE BOTH FETCHED, AND THEY ARE NOT INTERCHANGEABLE DOWNSTREAM. The six
# test-side spaces exist ONLY to measure an upper bound -- "how much better would a real aligned
# table have been than an English gloss?" -- and MUST stay out of the joint basis and out of
# training. The basis is a PCA over the aligned spaces, so fitting it on a test language's
# distribution is peeking, even though no gold label is involved. `align_vectors.py --stage basis`
# takes an explicit --langs list for exactly this reason; do not pass it TEST.
#
# ⚠ curl exits 56 (SIGPIPE) EVERY time `head` closes the pipe, so this script must not run under
# `set -e`/`pipefail` -- under those the successful case is read as a failure and nothing is renamed
# off `.part`. Completeness is judged by LINE COUNT instead. This is inherited verbatim from
# `fetch_vec_sources.sh` and is the one idiom in the repo that looks like a bug and is not.
set -u
cd "$(dirname "$0")/.."
N=${N:-200000}
SRC=assets_vec/src
mkdir -p "$SRC"
ALIGNED=https://dl.fbaipublicfiles.com/fasttext/vectors-aligned

# The 32 v2 TRAINING languages that are in the published aligned set (45 % of training tokens).
TRAIN="af ar bg ca cs da de en es et fa fi fr he hi hr id it ko nl no pl pt ro ru sk sl sv ta tr uk zh"
# The 6 v2 TEST languages that are. Diagnostic only -- see the warning above.
TEST="el hu lt lv th vi"

fetch () {
  local l=$1 out=$SRC/align.$1.vec
  if [ -s "$out" ]; then echo "have  $l ($(wc -l < "$out" | tr -d ' ') lines)"; return; fi
  curl -sSL --max-time 3600 "$ALIGNED/wiki.$l.align.vec" | head -n $((N+1)) > "$out.part"
  local got; got=$(wc -l < "$out.part" | tr -d ' ')
  if [ "$got" -ge 1000 ]; then mv "$out.part" "$out"; echo "got   $l ($got lines)"
  else echo "  !! $l only $got lines, keeping .part"; fi
}

for l in $TRAIN; do fetch "$l"; done
for l in $TEST;  do fetch "$l"; done
echo "DONE  train=$(echo $TRAIN | wc -w | tr -d ' ') test=$(echo $TEST | wc -w | tr -d ' ')"
