#!/usr/bin/env bash
# Does the morphologiser tag better when its encoder also reads PCA'd SikuBERT vectors?
#
# The morphologiser's own encoder is `spacy.HashEmbedCNN.v2`, width 64, 2 000 hash rows for 9 029
# training types, NORM/PREFIX/SUFFIX/SHAPE, no vectors. `pretrained_vectors = true` makes
# `MultiHashEmbed` CONCATENATE a StaticVectors projection with those hash channels before the
# Maxout -- "tok2vec (+) PCA'd SikuBERT" in the only sense spaCy offers.
#
# ⚠ THE COMPARISON IS AGAINST THE SHUFFLED CONTROL, NEVER AGAINST NO VECTORS.
# `include_static_vectors` adds a projection and widens the Maxout, so arm-versus-baseline
# confounds the information with the parameters. The control table holds the SAME rows with the
# type-to-row correspondence destroyed.
# ⚠ AND NEVER READ ONE SEED: the kanripo-vector arm's +0.46 on seed 0 was +0.04 over three.
set -u
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
DIM="${DIM:-96}"

# 1. the tables. --shuffle writes the matched control from the same rows.
[ -f vectors_lzh_siku${DIM}.vec ] || \
  $PY -u scripts/build_lzh_sikubert_vectors.py --out vectors_lzh_siku${DIM}.vec --dim $DIM
[ -f vectors_lzh_siku${DIM}_shuf.vec ] || \
  $PY -u scripts/build_lzh_sikubert_vectors.py --out vectors_lzh_siku${DIM}_shuf.vec --dim $DIM --shuffle --from-vec vectors_lzh_siku${DIM}.vec
for v in vectors_lzh_siku${DIM} vectors_lzh_siku${DIM}_shuf; do
  [ -d "$v" ] || $PY scripts/init_lzh_vectors.py "$v.vec" "$v"
done

# 2. configs, one per (variant, seed)
for s in $SEEDS; do
  $PY scripts/make_lzh_sikuvec_config.py --variant vectors --seed "$s" \
      --vectors vectors_lzh_siku${DIM} --out configs/config_lzh_sikuvec_s${s}.cfg
  $PY scripts/make_lzh_sikuvec_config.py --variant control --seed "$s" \
      --control-vectors vectors_lzh_siku${DIM}_shuf --out configs/config_lzh_sikuvec_ctl_s${s}.cfg
done

# 3. train. `python -u` and NO `| tail`: a piped spacy train hides everything until it exits, and
# model-last's mtime is the only reliable progress signal (CLAUDE.md operational notes).
for s in $SEEDS; do
  for arm in sikuvec sikuvec_ctl; do
    out="training_lzh_${arm}_s${s}"
    [ -d "$out/model-best" ] && { echo "  $out exists — skip"; continue; }
    echo "=== $out ==="
    $PY -u -m spacy train "configs/config_lzh_${arm}_s${s}.cfg" --output "$out" \
        --code scripts/seg_code.py > "train_lzh_${arm}_s${s}.log" 2>&1 \
      || { echo "  $out FAILED — see train_lzh_${arm}_s${s}.log"; continue; }
  done
done

# 4. score. UPOS is what the question is about, so report it sliced by the failure population as
# well as in aggregate -- an aggregate is exactly what hid the kanripo vectors' real behaviour.
mkdir -p metrics/lzh
for s in $SEEDS; do
  for arm in sikuvec sikuvec_ctl; do
    out="training_lzh_${arm}_s${s}"
    [ -d "$out/model-best" ] || continue
    $PY -m spacy evaluate "$out/model-best" \
        corpus_lzh_trad/lzh_kyoto-sud-test.relabeled_ext.udep_ruled.punct.rulemerged.spacy \
        --gold-preproc --code scripts/seg_code.py \
        --output "metrics/lzh/metrics_lzh_${arm}_s${s}_gp.json" >/dev/null 2>&1
  done
done
$PY -u scripts/report_lzh_sikuvec.py --seeds $SEEDS
