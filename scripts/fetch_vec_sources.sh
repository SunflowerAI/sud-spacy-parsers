#!/usr/bin/env bash
# Fetch the source vector spaces and bilingual dictionaries for the aligned-vector side assets.
#
# Everything is truncated to the top N lines as it downloads: fastText .vec files are sorted by
# corpus frequency, so `head` gives the most frequent types and curl stops when the pipe closes.
# The full aligned English file is 5.7 GB; we pull ~450 MB of it.
#
# ⚠ curl exits 56 (SIGPIPE) EVERY time head closes the pipe, so this is the one script in the repo
# that must not run under `set -e`/`pipefail` -- under those the successful case is read as failure
# and nothing gets renamed off .part. Completeness is judged by LINE COUNT instead.
set -u
cd "$(dirname "$0")/.."
N=${N:-200000}
SRC=assets_vec/src
DICT=assets_vec/dict
mkdir -p "$SRC" "$DICT"

ALIGNED=https://dl.fbaipublicfiles.com/fasttext/vectors-aligned
CRAWL=https://dl.fbaipublicfiles.com/fasttext/vectors-crawl
WIKI=https://dl.fbaipublicfiles.com/fasttext/vectors-wiki
MUSE=https://dl.fbaipublicfiles.com/arrival/dictionaries

fetch () {   # url  outfile  [gz]
  local url=$1 out=$2 gz=${3:-}
  if [ -s "$out" ]; then echo "have $out ($(wc -l < "$out") lines)"; return; fi
  echo "fetch $out"
  if [ "$gz" = gz ]; then
    curl -sSL --max-time 7200 "$url" | gunzip -c 2>/dev/null | head -n $((N+1)) > "$out.part"
  else
    curl -sSL --max-time 7200 "$url" | head -n $((N+1)) > "$out.part"
  fi
  local got; got=$(wc -l < "$out.part")
  # a truncated download is short; anything at the cap (or a genuinely smaller source) is complete
  if [ "$got" -ge 1000 ]; then mv "$out.part" "$out"; echo "  -> $got lines"
  else echo "  !! only $got lines, keeping $out.part"; fi
}

for l in en zh ko id fa ar ta; do fetch "$ALIGNED/wiki.$l.align.vec" "$SRC/align.$l.vec"; done
for l in ja la te;             do fetch "$CRAWL/cc.$l.300.vec.gz"    "$SRC/cc.$l.vec" gz;  done
fetch "$WIKI/wiki.zh_yue.vec" "$SRC/wiki.yue.vec"

for l in ja ta ko zh ar fa id; do
  [ -s "$DICT/en-$l.txt" ] || curl -sSL --max-time 600 "$MUSE/en-$l.txt" -o "$DICT/en-$l.txt"
  echo "dict en-$l: $(wc -l < "$DICT/en-$l.txt") pairs"
done
echo DONE
