#!/usr/bin/env bash
# 48 learned + 48 frozen-semantic dimensions vs the released 96 all-learned, on English.
# Single variable: same corpus, same depth, same TOTAL width -- capacity held constant, information
# added. Watch `comp:obl` F, not LAS: the earlier static-vectors test (sud-md-static-vectors) found
# LAS gains inside seed noise while comp:obl F consistently FELL (id -2.27, ko -4.75), and that is
# the metric this project exists for.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for seed in 0 1 2; do
  cfg=configs/config_en_split48_s${seed}.cfg
  $PY - "$seed" <<PYEOF
from thinc.api import Config
import sys
c = Config().from_disk("configs/config_en_split48.cfg", interpolate=False)
c["system"]["seed"] = int(sys.argv[1])
c.to_disk("configs/config_en_split48_s" + sys.argv[1] + ".cfg")
PYEOF
  echo "=== split48 seed $seed ==="
  $PY -u -m spacy train "$cfg" --code scripts/seg_code.py \
      --output "training_en_split48_s${seed}/" \
      --paths.train corpus_en_ewt_ext/en_ewt-sud-train.relabeled_ext.spacy \
      --paths.dev corpus_en_ewt_ext/en_ewt-sud-dev.relabeled_ext.spacy \
      > "train_en_split48_s${seed}.log" 2>&1 \
    || { echo "  FAILED"; tail -12 "train_en_split48_s${seed}.log"; continue; }
  echo "  done"
done
