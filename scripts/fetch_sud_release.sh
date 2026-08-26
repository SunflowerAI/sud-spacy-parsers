#!/usr/bin/env bash
# Fetch the whole SUD 2.18 release -- 352 corpora in one tarball -- for the generic parser v2.
#
# The thirteen treebanks this repo already carries under assets*/ are ALSO SUD 2.18, so nothing here
# straddles two releases. That matters: a treebank silently taken from a different generation loads,
# converts and trains exactly like a current one (CLAUDE.md hazard 2), and there would be no tell.
#
# ⚠ The corpus COUNT is the completeness check, not the exit status. A truncated tarball can still
# extract a plausible-looking subtree, and "some treebanks are missing" is exactly the failure that
# would show up much later as a hole in the typological sample rather than as an error here.
set -u
cd "$(dirname "$0")/.."

VER=${VER:-2.18}
URL=${URL:-https://grew.fr/download/sud-treebanks-v$VER.tgz}
OUT=${OUT:-assets_sud218}
TGZ=$OUT/sud-treebanks-v$VER.tgz
EXPECT=${EXPECT:-352}

mkdir -p "$OUT"

if [ ! -s "$TGZ" ]; then
  echo "fetch $URL"
  # -C - resumes a part file; the release is ~560 MB and a dropped connection should not restart it.
  curl -fL --no-progress-meter -C - --retry 3 --retry-delay 5 --max-time 7200 -o "$TGZ.part" "$URL" || {
    echo "!! download failed, keeping $TGZ.part for resume"; exit 1; }
  mv "$TGZ.part" "$TGZ"
fi
echo "have $TGZ ($(du -h "$TGZ" | cut -f1))"

# The tarball's own top-level directory name varies by release; find it rather than assuming.
ROOT=$(tar tzf "$TGZ" | head -1 | cut -d/ -f1)
if [ ! -d "$OUT/$ROOT" ]; then
  echo "extract -> $OUT/$ROOT"
  tar xzf "$TGZ" -C "$OUT"
fi

n=$(find "$OUT/$ROOT" -maxdepth 1 -type d -name '*SUD_*' | wc -l | tr -d ' ')
echo "corpora: $n"
if [ "$n" -ne "$EXPECT" ]; then
  echo "!! expected $EXPECT corpora, found $n -- refusing. Delete $OUT/$ROOT and re-run, or set EXPECT."
  exit 1
fi

# Report the three families separately: the native counts are the thing most likely to drift between
# releases, and prep_generic_v2 treats sud-native as a tie-break, so a wrong count changes sampling.
printf "  SUD_  %s\n  mSUD_ %s\n  pSUD_ %s\n" \
  "$(find "$OUT/$ROOT" -maxdepth 1 -type d -name 'SUD_*'  | wc -l | tr -d ' ')" \
  "$(find "$OUT/$ROOT" -maxdepth 1 -type d -name 'mSUD_*' | wc -l | tr -d ' ')" \
  "$(find "$OUT/$ROOT" -maxdepth 1 -type d -name 'pSUD_*' | wc -l | tr -d ' ')"
echo "DONE $OUT/$ROOT"
