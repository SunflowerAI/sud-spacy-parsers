#!/usr/bin/env bash
# Does the PARSER improve when PCA'd SikuBERT vectors are injected ABOVE the frozen encoder?
#
# `sud.Tok2VecPlusFeats.v1` concatenates the parser's `Tok2VecListener` on the FROZEN shared
# encoder with `sud.StaticVecChannel.v1`, so the vectors reach the parser's decision and nothing
# else, and the shared encoder is never retrained. That is the injection point NEGATIVE-RESULTS.md
# found decisive for the conditioned XPOS tagger (embed: -0.3 to -0.6; above the encoder: helps).
#
# ⚠ THE HEADLINE CANNOT RESOLVE THIS AND IS NOT THE RESULT. A static vector informs only a decision
# the FORM does not already settle; unseen forms are 1.15 % of Kyoto test tokens, so +15 LAS there
# is +0.17 aggregate against a ~0.5 seed spread. `eval_lex_slices.py` scores the slice the channel's
# own rationale names -- run that, and read the headline only to confirm nothing broke.
# ⚠ AND THE COMPARISON IS THE SHUFFLED TABLE, never "no vectors": StaticVectors adds a projection
# and widens the parser's lower layer.
set -u
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
SUF=relabeled_ext.udep_ruled.punct.rulemerged
D=assets_lzh/SUD_Classical_Chinese-Kyoto

for v in vectors_lzh_siku96 vectors_lzh_siku96_shuf; do
  [ -d "$v" ] || { echo "missing $v — run scripts/train_lzh_sikuvec.sh first"; exit 1; }
done
for s in $SEEDS; do
  $PY scripts/make_lzh_parservec_config.py --variant vectors --seed "$s" \
      --out configs/config_lzh_parservec_s${s}.cfg
  $PY scripts/make_lzh_parservec_config.py --variant control --seed "$s" \
      --out configs/config_lzh_parservec_ctl_s${s}.cfg
done

# `python -u` and NO `| tail`: a piped spacy train hides everything until it exits.
for s in $SEEDS; do
  for arm in parservec parservec_ctl; do
    out="training_lzh_${arm}_s${s}"
    [ -d "$out/model-best" ] && { echo "  $out exists — skip"; continue; }
    echo "=== $out ==="
    $PY -u -m spacy train "configs/config_lzh_${arm}_s${s}.cfg" --output "$out" \
        --code scripts/seg_code.py > "train_lzh_${arm}_s${s}.log" 2>&1 \
      || { echo "  $out FAILED — see train_lzh_${arm}_s${s}.log"; continue; }
  done
done

mkdir -p metrics/lzh
for s in $SEEDS; do
  for arm in parservec parservec_ctl; do
    [ -d "training_lzh_${arm}_s${s}/model-best" ] || continue
    $PY -m spacy evaluate "training_lzh_${arm}_s${s}/model-best" \
        corpus_lzh_trad/lzh_kyoto-sud-test.${SUF}.spacy --gold-preproc \
        --code scripts/seg_code.py \
        --output "metrics/lzh/metrics_lzh_${arm}_s${s}_gp.json" >/dev/null 2>&1
  done
done

# THE READING THAT MATTERS: LAS by train-frequency slice, arm against its own control.
args=""
for s in $SEEDS; do
  for arm in parservec parservec_ctl; do
    [ -d "training_lzh_${arm}_s${s}/model-best" ] && \
      args="$args --arm ${arm}_s${s}=training_lzh_${arm}_s${s}/model-best"
  done
done
$PY -u scripts/eval_lex_slices.py --train "$D/lzh_kyoto-sud-train.${SUF}.conllu" \
    --test "$D/lzh_kyoto-sud-test.${SUF}.conllu" $args
