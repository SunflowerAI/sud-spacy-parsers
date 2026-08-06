#!/usr/bin/env bash
# Rebuild the zh arm to operate in TRADITIONAL characters only, converting at the boundaries.
#
# WHY.  Training on both real treebanks for the same sentences let the model read either script, but
# it SPLIT the vocabulary: 22.7 % of the type inventory is a cross-script twin (15,848 types collapse
# to 12,248 under t2s), so 個 and 个 never pool their counts and any ranking over types
# came out mixed-script.
# One script inside, either script outside (`zh_script.py`), same shape as sa_deva.
#
# THE CONVERSION IS MEASURED, not assumed -- GSD and GSDSimp are the SAME 98,614 tokens in the two
# scripts, so s2tw can be scored against gold directly: 99.291 % with the punctuation map (665 of
# the raw 1,364 disagreements were `”` vs `」`), and the OUTPUT round trip t2s(s2t(w)) == w holds for
# 99.870 %. So a wrong conversion can cost the parse but essentially never the returned string.
#
# Training data halves in token count (98,614 rather than 197,228) but loses NO information: the
# second half was the same sentences transliterated. What it loses is script-variation robustness,
# which is exactly what the boundary conversion now supplies instead.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
C=corpus_zh_trad
TR=$C/zh_gsd-sud-train.relabeled_ext.spacy
DV=$C/zh_gsd-sud-dev.relabeled_ext.spacy

step () { echo "=== $1 ==="; }

step "base parser (traditional only)"
$PY -u -m spacy train configs/config_zh.cfg $CODE --output training_zh_trad/ \
    --paths.train "$TR" --paths.dev "$DV" > train_zh_trad.log 2>&1 \
  || { echo "  FAILED"; tail -15 train_zh_trad.log; exit 1; }
echo "  LAS $(grep -E '^ *[0-9]+ +[0-9]+ ' train_zh_trad.log | tail -1 | awk '{print $11}')"

# The layer stack, in the order CLAUDE.md requires: seg -> morph -> lemma -> sud. Each SOURCES and
# FREEZES the arm below it and trains only the new component with its own small encoder, so the
# lower layers come out byte-identical.
for layer in seg morph lemma; do
  step "$layer layer"
  $PY scripts/make_${layer}_config.py zh >/dev/null 2>&1 || true
  $PY -u -m spacy train "configs/config_zh_${layer}.cfg" $CODE \
      --output "training_zh_trad_${layer}/" --paths.train "$TR" --paths.dev "$DV" \
      > "train_zh_trad_${layer}.log" 2>&1 \
    || { echo "  $layer FAILED"; tail -12 "train_zh_trad_${layer}.log"; exit 1; }
done
echo "done -- next: bundle the char segmenter, then attach zh_script"
