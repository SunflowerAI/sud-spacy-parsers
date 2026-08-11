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
# The base MUST be trained through the seg recipe (sud.GoldTokCorpus.v1 + sents_f 0.05), not the
# bare configs/config_zh.cfg. `seg` is a BASE recipe, not a stackable frozen layer: a gold_preproc
# base never sees a sentence boundary, so no later layer can teach it one, and the failure is
# INVISIBLE to every gold-preproc metric (SENTS_F reads 100 because each dev example is already one
# sentence). The first build of this arm shipped exactly that -- raw SENT F 0, one sentence per doc.
$PY scripts/make_seg_config.py configs/config_zh.cfg --out configs/config_zh_seg.cfg \
  || { echo "  seg config FAILED"; exit 1; }
$PY -u -m spacy train configs/config_zh_seg.cfg $CODE --output training_zh_trad/ \
    --paths.train "$TR" --paths.dev "$DV" > train_zh_trad.log 2>&1 \
  || { echo "  FAILED"; tail -15 train_zh_trad.log; exit 1; }
echo "  LAS $(grep -E '^ *[0-9]+ +[0-9]+ ' train_zh_trad.log | tail -1 | awk '{print $11}')"
echo "  SENTS_F $(grep -E '^ *[0-9]+ +[0-9]+ ' train_zh_trad.log | tail -1 | awk '{print $12}')"

# The layers above it, in the order CLAUDE.md requires: morph -> lemma. Each SOURCES and FREEZES the
# arm below it and trains only the new component with its own small encoder, so the lower layers
# come out byte-identical. Both config makers take (base_config, source_model) -- passing a LANG
# instead is what silently produced no config the first time round.
$PY scripts/make_morph_config.py configs/config_zh_seg.cfg training_zh_trad/model-best \
    --out configs/config_zh_trad_morph.cfg || { echo "  morph config FAILED"; exit 1; }
$PY scripts/make_lemma_config.py configs/config_zh_trad_morph.cfg training_zh_trad_morph/model-best \
    --out configs/config_zh_trad_lemma.cfg || { echo "  lemma config FAILED"; exit 1; }

for layer in morph lemma; do
  step "$layer layer"
  $PY -u -m spacy train "configs/config_zh_trad_${layer}.cfg" $CODE \
      --output "training_zh_trad_${layer}/" --paths.train "$TR" --paths.dev "$DV" \
      > "train_zh_trad_${layer}.log" 2>&1 \
    || { echo "  $layer FAILED"; tail -12 "train_zh_trad_${layer}.log"; exit 1; }
done

# Prove the thing this script exists to get right, before anything is packaged.
step "sentencising check"
$PY - <<'EOF'
import spacy
nlp = spacy.load("training_zh_trad_lemma/model-best")
n = len(list(nlp("\u4eca\u5929\u5929\u6c23\u5f88\u597d\u3002\u6211\u5011\u53bb\u516c\u5712\u6563\u6b65\u3002").sents))
print(f"  two-sentence input -> {n} sentence(s)")
raise SystemExit(0 if n == 2 else 1)
EOF
[ $? -eq 0 ] || { echo "  BASE DOES NOT SEGMENT -- do not package"; exit 1; }

echo "done -- next: bundle the char segmenter, then attach zh_script"
