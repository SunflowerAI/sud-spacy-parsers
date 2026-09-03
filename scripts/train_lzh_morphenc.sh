#!/usr/bin/env bash
# Does the parser gain from the MORPHOLOGISER'S TRAINED ENCODER, rather than from raw vectors?
#
# The `parservec` arm handed the parser a raw 96-d SikuBERT row and asked it to learn, from parse
# supervision alone and on the 1.15 % of tokens where it matters, what category that row implies.
# Three seeds: nothing on any slice (NEGATIVE-RESULTS.md). This arm hands over the extraction
# ALREADY DONE -- the 64-d UPOS-supervised encoder, frozen -- as a dense channel above the same
# frozen shared encoder, so the ONLY difference from `parservec` is what the side channel carries.
#
# ⚠ CONTROL = the same architecture with a donor trained on the SHUFFLED table. Parameter count,
# depth, width, supervision and recipe all held fixed; only the donor's quality on rare and unseen
# forms varies (73.98 % vs 66.92 % UPOS on treebank-unseen forms). "No side channel" would confound
# the transfer with 499 456 extra parameters.
# ⚠ Each donor is paired with the vector table it was TRAINED against: StaticVectors reads
# doc.vocab.vectors at forward time, so a mismatched host silently runs it out of distribution.
# ⚠ The donor is fixed at seed 0; the PARSER seed varies. That measures parser-seed spread only.
set -u
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
SUF=relabeled_ext.udep_ruled.punct.rulemerged
D=assets_lzh/SUD_Classical_Chinese-Kyoto

for d in training_lzh_sikuvec_s0/model-best training_lzh_sikuvec_ctl_s0/model-best; do
  [ -d "$d" ] || { echo "missing donor $d — run scripts/train_lzh_sikuvec.sh first"; exit 1; }
done
for s in $SEEDS; do
  $PY scripts/make_lzh_morphenc_config.py --variant vectors --seed "$s" \
      --out configs/config_lzh_morphenc_s${s}.cfg
  $PY scripts/make_lzh_morphenc_config.py --variant control --seed "$s" \
      --out configs/config_lzh_morphenc_ctl_s${s}.cfg
done

for s in $SEEDS; do
  for arm in morphenc morphenc_ctl; do
    out="training_lzh_${arm}_s${s}"
    [ -d "$out/model-best" ] && { echo "  $out exists — skip"; continue; }
    echo "=== $out ==="
    $PY -u -m spacy train "configs/config_lzh_${arm}_s${s}.cfg" --output "$out" \
        --code scripts/seg_code.py > "train_lzh_${arm}_s${s}.log" 2>&1 \
      || { echo "  $out FAILED — see train_lzh_${arm}_s${s}.log"; continue; }
    # ⚠ NOT OPTIONAL. A drifted donor is a fine-tune wearing a transfer's name, and an inert one
    # makes the arm its own control. Both are invisible in the training log.
    $PY scripts/check_frozen_pipe_tok2vec.py --arm "$out/model-best" \
        --donor "$([ "$arm" = morphenc ] && echo training_lzh_sikuvec_s0 || echo training_lzh_sikuvec_ctl_s0)/model-best" \
      || echo "  ⚠ $out FAILED ITS FROZEN/LIVE CHECK — do not read its numbers"
  done
done

args=""
for s in $SEEDS; do
  for arm in morphenc morphenc_ctl; do
    [ -d "training_lzh_${arm}_s${s}/model-best" ] && \
      args="$args --arm ${arm}_s${s}=training_lzh_${arm}_s${s}/model-best"
  done
done
# The slice table is the reading that matters; the headline only confirms nothing broke.
$PY -u scripts/eval_lex_slices.py --train "$D/lzh_kyoto-sud-train.${SUF}.conllu" \
    --test "$D/lzh_kyoto-sud-test.${SUF}.conllu" $args
