#!/usr/bin/env bash
# lzh, TRADITIONAL-ONLY END TO END, with s2t/t2s adapters at the pipeline boundary.
#
# The parser is the traditional-only arm (training_lzh_trad, dev LAS 76.57), not the both-scripts one
# (79.0). The 2.4-point cost is the OpenCC script augmentation Kyoto-Both provides, and it is
# ACCEPTED: a both-scripts inventory never pools 遠 with 远 -- 3,108 of 12,137 types (26 %) are
# simplified variants -- so every script-varying character competes with itself, which the ranking
# layers pay for directly. Simplified input is converted in and back out by `lzh_script`, exactly as
# zh does it, so nothing downstream ever sees two scripts.
#
# Everything stacks on the TRADITIONAL base: morphologizer (already built), then the SUD pipes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export LZH_BASE_ARM=training_lzh_trad_morph/model-best
TRAD='assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-$2.relabeled_ext.udep_ruled.punct.rulemerged.conllu'
STAMP () { echo; echo "===== $* ===== $(date '+%H:%M')"; }

for p in training_lzh_trad/model-best training_lzh_trad_morph/model-best; do
  [ -e "$p" ] || { echo "MISSING $p"; exit 1; }
done
echo "base: $LZH_BASE_ARM"

STAMP "1/1  sud_subject + sud_shared on the traditional base"
LZH_SRC="$TRAD" SUD_SUFFIX="_trad_sud" bash scripts/train_sud.sh lzh 2>&1 | tail -12
[ -d training_lzh_trad_sud/model-best ] && $PY -c "
import json;print('  pipeline:', json.load(open('training_lzh_trad_sud/model-best/meta.json'))['pipeline'])" \
  || echo "  SUD RETRAIN FAILED"

STAMP "ALL DONE"
