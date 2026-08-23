#!/usr/bin/env bash
# Morph + lemma arms for the eojeol Korean model, by the project's standard freeze recipe.
#
# The base (seg) arm is already trained: `training_ko_eojeol_seg/`, raw TOK 99.77 / LAS 55.00 /
# SENT F 83.80. This adds the two layers above it, each sourcing and FREEZING everything below and
# training only its own component with its own dedicated HashEmbedCNN — so parsing cannot regress
# and the frozen weights stay byte-identical (asserted with `cmp` below).
#
# NB the eojeol treebank fuses noun+particle into one token, so FEATS and LEMMA are whatever the
# ORIGINAL SUD_Korean-GSD carries, not the morpheme-level annotation the retokenised arm had. The
# note in CLAUDE.md about `assets_ko_retok_rl` having an entirely empty FEATS column applies to the
# RETOKENISED corpus; this one is checked below rather than assumed, because a vacuous morphologiser
# scores a misleading 100 % against all-empty gold.
set -euo pipefail

cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE=scripts/seg_code.py

echo "=== 0/3  is there anything for morph and lemma to learn? ===================="
$PY - <<'PY'
import collections, pathlib
p = pathlib.Path("assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu")
feats = lemma = tot = 0
for line in p.open(encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    c = line.rstrip("\n").split("\t")
    if len(c) < 10 or "-" in c[0]:
        continue
    tot += 1
    feats += c[5] not in ("_", "")
    lemma += c[2] not in ("_", "")
print(f"  train tokens {tot}")
print(f"    non-empty FEATS : {feats} ({feats/tot:.1%})")
print(f"    non-empty LEMMA : {lemma} ({lemma/tot:.1%})")
if feats / tot < 0.01:
    print("    -> FEATS is effectively empty: the morphologiser would be VACUOUS and its")
    print("       morph_acc meaningless. Do not ship it.")
if lemma / tot < 0.01:
    print("    -> LEMMA is effectively empty: the lemmatiser would be VACUOUS.")
PY

echo "=== 1/3  morphologizer ======================================================"
$PY scripts/make_morph_config.py configs/config_ko_eojeol_seg.cfg \
    training_ko_eojeol_seg/model-best --out configs/config_ko_eojeol_morph.cfg
$PY -m spacy train configs/config_ko_eojeol_morph.cfg --output training_ko_eojeol_morph/ \
    --code $CODE 2>&1 | tail -3

echo "=== 2/3  lemmatizer ========================================================="
$PY scripts/make_lemma_config.py configs/config_ko_eojeol_morph.cfg \
    training_ko_eojeol_morph/model-best --out configs/config_ko_eojeol_lemma.cfg
$PY -m spacy train configs/config_ko_eojeol_lemma.cfg --output training_ko_eojeol_lemma/ \
    --code $CODE 2>&1 | tail -3

echo "=== 3/3  verify the frozen components are byte-identical ===================="
for c in tok2vec tagger parser; do
  if cmp -s "training_ko_eojeol_seg/model-best/$c/model" \
            "training_ko_eojeol_lemma/model-best/$c/model" 2>/dev/null; then
    echo "  $c: identical"
  else
    echo "  $c: DIFFERS -- the freeze recipe did not hold, parsing metrics need re-verification"
  fi
done

echo "=== final: raw end-to-end ==================================================="
$PY -m spacy evaluate training_ko_eojeol_lemma/model-best \
    corpus_ko_eojeol/ko_gsd-sud-test.relabeled_ext.spacy --code $CODE \
    --output metrics/ko/metrics_ko_eojeol_lemma_raw.json 2>&1 | grep -E "TOK|TAG|POS|MORPH|LEMMA|UAS|LAS|SENT"
