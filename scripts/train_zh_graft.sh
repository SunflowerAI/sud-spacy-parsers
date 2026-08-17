#!/usr/bin/env bash
# lzh -> zh tok2vec graft, three seeds against three seeds of a MATCHED control.
#
# WHY THIS DIRECTION.  The zh -> lzh graft was ruled out on coverage: 92.6 % of lzh's
# unseen-form test tokens are absent from zh GSD entirely, and the donor is smaller than the
# recipient. Reversed, every quantity flips -- donor 460 k tokens vs recipient 98 k (the shape that
# earned yue +1.15 LAS), zh's OOV rate is 12.46 % against lzh's 1.15 %, and 84.0 % of zh's OOV
# tokens have a first character the donor knows at median frequency 44. The transfer is
# CHARACTER-level (PREFIX), not word-level: only 4.4 % of zh's OOV tokens have their full form as an
# lzh key, so re-tokenising the donor to zh's word regime was ruled out separately.
#
# THE CONTROL SHARES THE CONFIG.  Both arms run configs/config_zh_graft.cfg; the graft passes
# --paths.init_tok2vec and the control passes nothing. Same code path, same [pretraining] block,
# same everything else -- the only kind of control worth reading.
#
# THREE SEEDS IS NOT OPTIONAL.  This arm family's seed spread is ~0.5 LAS and zh's test set is
# 12,010 tokens (a third of lzh's), so the noise floor is WIDER here. The kanripo vectors read
# +0.46 on seed 0 and +0.04 over three. Do not read a single row of this table.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG=configs/config_zh_graft.cfg
CODE="--code scripts/seg_code.py"
C=corpus_zh_trad
TR=$C/zh_gsd-sud-train.relabeled_ext.spacy
DV=$C/zh_gsd-sud-dev.relabeled_ext.spacy
TE=$C/zh_gsd-sud-test.relabeled_ext.spacy
BLOB=lzh_trad_tok2vec.bin

[ -f "$BLOB" ] || { echo "missing $BLOB -- run scripts/extract_tok2vec.py first"; exit 1; }

for seed in 0 1 2; do
  for arm in graft ctl; do
    out=training_zh_${arm}_s${seed}; log=train_zh_${arm}_s${seed}.log; met=metrics_zh_${arm}_s${seed}.json
    [ -f "$met" ] && { echo "=== zh_${arm}_s${seed}: metrics exist, skipping"; continue; }
    echo "=== zh_${arm}_s${seed} ==="
    if [ "$arm" = graft ]; then INIT=(--paths.init_tok2vec "$BLOB"); else INIT=(); fi
    $PY -u -m spacy train "$CFG" $CODE --output "$out/" \
        --system.seed "$seed" "${INIT[@]}" \
        --paths.train "$TR" --paths.dev "$DV" > "$log" 2>&1 \
      || { echo "  TRAIN FAILED"; tail -12 "$log"; continue; }
    # The graft arm MUST have actually loaded the donor; a silent no-op would make the table
    # meaningless. CHECK THE SAVED CONFIG, NOT THE LOG: spaCy's `init_tok2vec` runs
    # `layer.from_bytes(weights_data)` unconditionally and only the `logger.info` line AFTER it is
    # level-gated, so "Loaded pretrained weights" is absent from any run without `--verbose` even
    # though the load happened. Grepping for it rejected all three graft arms on the first sweep.
    if [ "$arm" = graft ] && ! grep -q '^init_tok2vec = "'"$BLOB"'"' "$out/model-best/config.cfg"; then
      echo "  !! graft arm did NOT receive the donor path -- refusing to score it"; continue
    fi
    $PY -m spacy evaluate "$out/model-best" "$TE" $CODE --gold-preproc --output "$met" \
        > "eval_zh_${arm}_s${seed}.log" 2>&1 \
      || { echo "  EVAL FAILED"; tail -8 "eval_zh_${arm}_s${seed}.log"; continue; }
    $PY -c "import json;d=json.load(open('$met'));print('  TAG %.4f UAS %.4f LAS %.4f'%(d['tag_acc'],d['dep_uas'],d['dep_las']))"
  done
done
echo "ZH GRAFT SWEEP DONE"
