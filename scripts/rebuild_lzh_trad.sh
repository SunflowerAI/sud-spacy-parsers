#!/usr/bin/env bash
# Rebuild lzh on the TRADITIONAL-ONLY treebank, with script conversion at the pipeline boundary.
#
# WHY. Kyoto-Both is the same text twice, once OpenCC-converted: 12,137 types / 920,780 tokens
# against traditional-only's 9,029 / 460,390. So 3,108 types (26 %) are simplified variants, and the
# penalty is ASYMMETRIC -- a character unchanged by `t2s` appears in both copies and gets DOUBLE the
# counts, while one that differs has its mass SPLIT across two types. Script-varying characters are
# therefore penalised about 4:1 against invariant ones. Harmless for parsing; a systematic handicap
# for anything that ranks types, where a character and its variant turn up as separate
# candidates.
#
# This is the argument zh_script.py already makes and measures for zh. Simplified INPUT is handled by
# `lzh_script` at the boundary (s2t in, t2s out), exactly as zh does it.
#
# ORDER. The base arm must exist before anything stacks on it. The tokeniser is independent of
# the base, so a failure in one does not block the other.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_lzh/SUD_Classical_Chinese-Kyoto
SUF=relabeled_ext.udep_ruled.punct.rulemerged
STAMP () { echo; echo "===== $* ===== $(date '+%H:%M')"; }

echo "starting the lzh traditional-only rebuild at $(date '+%H:%M')"

STAMP "1/4  corpus"
mkdir -p corpus_lzh_trad
for s in train dev test; do
  f="$A/lzh_kyoto-sud-$s.$SUF.conllu"
  [ -f "$f" ] || { echo "  MISSING $f"; exit 1; }
  $PY -m spacy convert "$f" corpus_lzh_trad/ --converter conllu -n 10 2>&1 | tail -1
done
ls -la corpus_lzh_trad/ | awk 'NR>1 {print "  "$9"  "$5"B"}'

STAMP "2/4  base arm (tok2vec + tagger + parser), traditional only"
$PY -u -m spacy train configs/config_lzh.cfg --code scripts/seg_code.py \
    --output training_lzh_trad/ \
    --paths.train corpus_lzh_trad/lzh_kyoto-sud-train.$SUF.spacy \
    --paths.dev  corpus_lzh_trad/lzh_kyoto-sud-dev.$SUF.spacy \
    > train_lzh_trad.log 2>&1
[ -d training_lzh_trad/model-best ] || { echo "  BASE FAILED"; tail -12 train_lzh_trad.log; }

STAMP "3/4  morphologizer, sourced+frozen from the new base"
if [ -d training_lzh_trad/model-best ]; then
  # config_lzh_rm_morph.cfg hardcodes source = training_lzh_both_rulemerged/model-best, so repoint it
  $PY - <<'PYEOF'
from thinc.api import Config
p = "configs/config_lzh_rm_morph.cfg"
c = Config().from_disk(p, interpolate=False)   # interpolate=False or ${paths.train} -> null (E913)
n = 0
for name, blk in c["components"].items():
    if isinstance(blk, dict) and blk.get("source"):
        blk["source"] = "training_lzh_trad/model-best"; n += 1
c.to_disk("configs/config_lzh_trad_morph.cfg")
print(f"  wrote configs/config_lzh_trad_morph.cfg ({n} sourced components repointed)")
PYEOF
  $PY -u -m spacy train configs/config_lzh_trad_morph.cfg --code scripts/seg_code.py \
      --output training_lzh_trad_morph/ \
      --paths.train corpus_lzh_trad/lzh_kyoto-sud-train.$SUF.spacy \
      --paths.dev  corpus_lzh_trad/lzh_kyoto-sud-dev.$SUF.spacy \
      > train_lzh_trad_morph.log 2>&1
  [ -d training_lzh_trad_morph/model-best ] || { echo "  MORPH FAILED"; tail -10 train_lzh_trad_morph.log; }
fi

STAMP "4/4  character segmenter, traditional only"
# Independent of the base arm, so it runs whatever happened above.
mkdir -p data_seg_lzh_trad
for s in train dev test; do
  $PY scripts/make_seg_pairs.py "$A/lzh_kyoto-sud-$s.$SUF.conllu" \
      "data_seg_lzh_trad/$s.jsonl" --min-chunk 1 2>&1 | sed 's/^/  /'
done
$PY scripts/train_samhita.py data_seg_lzh_trad/train.jsonl data_seg_lzh_trad/dev.jsonl \
    models/lzh_seg_char_trad --width 64 --depth 6 --epochs 30 2>&1 | tail -4

STAMP "ALL DONE"
