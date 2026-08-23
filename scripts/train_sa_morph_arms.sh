#!/bin/bash
# ⚠ SUPERSEDED GENERATION. This driver reads `corpus_sa_csl_rev` and chains
# `training_sa_csl_rev*` -- the pausa-normalised representation and the freeze-recipe arms
# built on it. Neither is what ships: the sa wheel is a JOINT MULTI-TASK arm
# (`training_sa_mp2_sub_s1`, `SA_BASE` in package_sud.sh) trained on the DCS/MWT
# representation. Kept as the experimental record; do not treat its output as current.
# CLAUDE.md lists `rebuild_sa_csl_rev.sh` under "Superseded but kept"; the authority on
# which sa corpus feeds what is the BUILD PROVENANCE table in docs/sanskrit.md.
# Run the sa morphologiser affix ablation (see scripts/make_sa_morph_arms.py for the arm table and
# scripts/sud_affix_embed.py for why the layer exists).
#
# Every arm is the standard morph freeze recipe — source + FREEZE tok2vec/tagger/parser from
# training_sa_csl_rev/model-seg, train ONLY the morphologiser and its own dedicated encoder — so
# parsing CANNOT regress and no parse metric needs re-verification. Arms differ from the baseline
# in the `[components.morphologizer.model.tok2vec]` subtree only; the generator asserts that.
#
# Scoring uses scripts/eval_sa_compound.py, NOT `spacy evaluate`: the sa arms read MORPH as an INPUT
# feature (the tokeniser's Compound=Yes) and the stock reader never supplies it, which silently
# measures the model with one of its inputs deleted (LAS 0.5601 -> 0.5169).
#
# Usage: bash scripts/train_sa_morph_arms.sh [arm ...]      (default: base + every generated arm)
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
TRAIN=corpus_sa_csl_rev/train.csl_rev.spacy
DEV=corpus_sa_csl_rev/sa_vedic-sud-dev.csl_rev.spacy
TEST=corpus_sa_csl_rev/sa_vedic-sud-test.csl_rev.spacy

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(base $($PY scripts/make_sa_morph_arms.py --list | sed 's/^base //'))
fi

for arm in "${ARMS[@]}"; do
  if [ "$arm" = "base" ]; then cfg=configs/config_sa_morph.cfg
  else cfg=configs/config_sa_morph_${arm}.cfg; fi
  if [ ! -f "$cfg" ]; then echo "  $arm: $cfg missing — run make_sa_morph_arms.py"; continue; fi
  out=training_sa_morph_${arm}
  echo "########## sa morph arm: $arm ($cfg) ##########"
  $PY -m spacy train "$cfg" $CODE --output "$out/" \
      --paths.train "$TRAIN" --paths.dev "$DEV" > "train_sa_morph_${arm}.log" 2>&1
  if [ -d "$out/model-best" ]; then
    $PY scripts/eval_sa_compound.py "$out/model-best" "$TEST" \
        --out "metrics/sa/metrics_sa_morph_${arm}.json" > "eval_sa_morph_${arm}.log" 2>&1
    $PY -c "
import json
m=json.load(open('metrics/sa/metrics_sa_morph_${arm}.json'))
per=m.get('morph_per_feat') or {}
f=lambda k: per.get(k,{}).get('f',0)
print(f'  ${arm}: morph_acc {m[\"morph_acc\"]:.4f}  micro_f {m.get(\"morph_micro_f\",0):.4f}  pos {m[\"pos_acc\"]:.4f}  |  Voice {f(\"Voice\"):.3f}  VerbForm {f(\"VerbForm\"):.3f}  Tense {f(\"Tense\"):.3f}  Case {f(\"Case\"):.3f}')"
  else
    echo "  $arm FAILED:"; tail -12 "train_sa_morph_${arm}.log"
  fi
done
echo "########## sa morph arms done ##########"
