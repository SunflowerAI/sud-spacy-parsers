#!/usr/bin/env bash
# Stream the frequency-ordered HEAD of a fastText CC-157 .vec and stop -- the files are 1-4 GB and
# we only ever need the common types. The header line carries the row count, so it must be
# rewritten to match what we actually kept or every consumer mis-parses the file.
#
# LICENCE: fastText vectors are CC BY-SA 3.0. Fetching is not redistributing, so experimenting is
# fine, but a table DERIVED from them is a derivative work and share-alike would attach to any
# wheel shipping it -- which is flatly incompatible with the CC BY-NC-SA Latin model (the same
# conflict as Morpheus). For `la`, prefer PPMI+SVD over our own treebanks.
set -uo pipefail
N=${N:-100000}
for lang in "$@"; do
  out="vectors_ft/cc.${lang}.300.head${N}.vec"
  [ -s "$out" ] && { echo "  $lang: already have $out"; continue; }
  echo "  $lang: streaming top-$N rows..."
  curl -sL "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.vec.gz" \
    | gunzip -c 2>/dev/null | tail -n +2 | head -n "$N" > "$out.body" || true
  rows=$(wc -l < "$out.body" | tr -d ' ')
  [ "$rows" -gt 0 ] || { echo "  $lang: FAILED (0 rows)"; rm -f "$out.body"; continue; }
  { echo "$rows 300"; cat "$out.body"; } > "$out" && rm -f "$out.body"
  echo "  $lang: $rows rows -> $out ($(du -h "$out" | cut -f1))"
done
