#!/usr/bin/env bash
# Complete the zh TRADITIONAL-ONLY chain. `training_zh_trad` has only [tok2vec, tagger, parser]; the
# morphologizer and lemmatizer were never stacked on it, so the traditional-only zh model has
# never existed end to end.
#
# Same argument as lzh: a both-scripts inventory never pools 個 with 个, so every script-varying
# character competes with its own variant -- which `zh_script.py` already documents and measures for
# zh, and which cost lzh's upper layers directly. Simplified input is converted at the boundary by
# the `zh_script` component that is already written and registered.
#
# Each stage repoints the sourced components into a NEW config rather than editing one in place, so
# the both-scripts chain stays reproducible and no later run can silently pick up the wrong base.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STAMP () { echo; echo "===== $* ===== $(date '+%H:%M')"; }

TR=$(ls corpus_zh_trad/*train*.spacy 2>/dev/null | head -1)
DV=$(ls corpus_zh_trad/*dev*.spacy   2>/dev/null | head -1)
[ -n "$TR" ] && [ -n "$DV" ] || { echo "corpus_zh_trad missing train/dev"; exit 1; }
echo "corpus: $TR / $DV"

repoint () {   # $1 in-config  $2 out-config  $3 new source arm
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys
from thinc.api import Config
src, dst, arm = sys.argv[1:4]
c = Config().from_disk(src, interpolate=False)   # interpolate=False or ${paths.train} -> null (E913)
n = 0
for name, blk in c["components"].items():
    if isinstance(blk, dict) and blk.get("source"): blk["source"] = arm; n += 1
c.to_disk(dst)
print(f"  {dst}: {n} sourced components -> {arm}")
PYEOF
}

STAMP "1/2  morphologizer on the traditional parser"
repoint configs/config_zh_morph.cfg configs/config_zh_trad_morph.cfg training_zh_trad/model-best
$PY -u -m spacy train configs/config_zh_trad_morph.cfg --code scripts/seg_code.py \
    --output training_zh_trad_morph/ --paths.train "$TR" --paths.dev "$DV" \
    > train_zh_trad_morph.log 2>&1
[ -d training_zh_trad_morph/model-best ] || { echo "  MORPH FAILED"; tail -8 train_zh_trad_morph.log; exit 1; }

STAMP "2/2  lemmatizer"
repoint configs/config_zh_lemma.cfg configs/config_zh_trad_lemma.cfg training_zh_trad_morph/model-best
$PY -u -m spacy train configs/config_zh_trad_lemma.cfg --code scripts/seg_code.py \
    --output training_zh_trad_lemma/ --paths.train "$TR" --paths.dev "$DV" \
    > train_zh_trad_lemma.log 2>&1
[ -d training_zh_trad_lemma/model-best ] || { echo "  LEMMA FAILED"; tail -8 train_zh_trad_lemma.log; exit 1; }

STAMP "ALL DONE"
