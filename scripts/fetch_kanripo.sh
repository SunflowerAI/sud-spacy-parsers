#!/usr/bin/env bash
# Shallow-clone the Kanseki Repository (CC BY-SA 4.0) as a raw corpus for lzh vector training.
#
# lzh needs NO segmenter: the released tokeniser is one Han character per token, so the type
# inventory IS the character set (~10k) and every type is well attested. That makes lzh the
# cheapest of the three languages to build vectors for, despite having the largest corpus.
#
# Kyoto itself is only 374k tokens and, per its own README, contains NO punctuation -- kanripo is
# both far larger and punctuated. `align_kanripo_punct.py` already reads these files.
set -uo pipefail
DIR=${DIR:-assets_kanripo}
JOBS=${JOBS:-6}
mkdir -p "$DIR"
clone_one() {
  local name=$1 dir=$2
  [ -d "$dir/$name/.git" ] && return 0
  git clone --quiet --depth 1 "https://github.com/kanripo/$name.git" "$dir/$name" 2>/dev/null
}
export -f clone_one
xargs -P "$JOBS" -I{} bash -c 'clone_one "$@"' _ {} "$DIR" < "${1:-kanripo_repos.txt}"
echo "  cloned: $(find "$DIR" -maxdepth 1 -type d -name 'KR*' | wc -l | tr -d ' ')"
