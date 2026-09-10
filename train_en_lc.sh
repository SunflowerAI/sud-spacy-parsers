#!/usr/bin/env bash
# The incremental left-corner beam parser for English, for psycholinguistic modelling.
#
# Noji & Miyao's (COLING 2014) left-corner DEPENDENCY transition system, trained as a GENERATIVE
# model over SUD English EWT+GUM, decoded with a word-synchronous beam. It emits per-word surprisal
# and per-word memory-load predictors. This is NOT one of the released wheels and does not go
# through `package_sud.sh`; spaCy's parser cannot host it (arc-eager only, and its beam scores are
# unnormalised discriminative scores, so they yield no prefix probability).
#
# ⚠ Non-projective sentences are OUTSIDE the transition system, as in Noji & Miyao, and are dropped:
# 6.29 % of EWT+GUM train, 5.43 % of dev+test. Do NOT reach for pseudo-projective encoding to
# recover them -- `NEGATIVE-RESULTS.md` indicts that representation on accuracy grounds, and here it
# would be worse than useless, since the whole point of the arm is that stack depth means something
# and a `||` composite label makes the depth of a discontinuity meaningless.
#
# ⚠ The perplexities in `train_en_lc.log` are GOLD-PATH numbers over the oracle derivation. The
# psycholinguistic quantity is the beam-marginalised surprisal from `lc_surprisal.py`, which is
# lower. Never quote one for the other.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SCRATCH=${SCRATCH:-/tmp}
A=assets/en_ewtgum-sud
OUT=training_en_lc

mkdir -p "$OUT" metrics/en

# 0. The correctness gate. 100 % of PROJECTIVE sentences must derive and round-trip to the exact
#    labelled gold tree; anything less means the oracle is wrong, not that the treebank is hard.
$PY scripts/lc_transitions.py $A-train.relabeled_ext.conllu | tee metrics/en/lc_oracle_train.txt

# 1. Oracle derivations -> per-step tensors.
(cd scripts && ../$PY lc_features.py \
  --train ../$A-train.relabeled_ext.conllu \
  --dev   ../$A-dev.relabeled_ext.conllu \
  --out   "$SCRATCH/lc_en.pkl")

# 2. Train the generative model.
$PY -u scripts/lc_model.py --data "$SCRATCH/lc_en.pkl" \
  --out $OUT/lc_en_ewtgum.pt --epochs 25 --patience 3 2>&1 | tee train_en_lc.log

# 3. Self-training corpus: public-domain prose, parsed by the arm trained on the SAME relabelled
#    corpus this model uses. NOT training_en_gum_sud -- its labels predate the extended-scope
#    relabelling, and silver trees carrying labels this model never saw train silently.
$PY scripts/fetch_en_selftrain.py --out "$SCRATCH/en_selftrain.txt" --max-words 6000000
$PY scripts/make_silver_conllu.py --model training_en_gum_ext/model-best \
  --text "$SCRATCH/en_selftrain.txt" --out "$SCRATCH/en_silver.conllu"

# 4. ONE vocabulary over gold and silver, then two stages over it. Both stages must read this one
#    pickle: --init-from REFUSES a checkpoint built on a different vocabulary, because a warm start
#    needs the ids to match position for position (CLAUDE.md hazard 7).
(cd scripts && ../$PY lc_features.py \
  --train ../$A-train.relabeled_ext.conllu --dev ../$A-dev.relabeled_ext.conllu \
  --silver "$SCRATCH/en_silver.conllu" --max-silver 120000 \
  --out "$SCRATCH/lc_en_silver.pkl")
$PY -u scripts/lc_model.py --data "$SCRATCH/lc_en_silver.pkl" --split silver \
  --out $OUT/lc_en_silver.pt --epochs 8 --patience 2 2>&1 | tee train_en_lc_silver.log
$PY -u scripts/lc_model.py --data "$SCRATCH/lc_en_silver.pkl" --split train \
  --init-from $OUT/lc_en_silver.pt --out $OUT/lc_en_selftrained.pt \
  --epochs 25 --patience 3 2>&1 | tee train_en_lc_ft.log

# 5. Beam decode the test set: surprisal + memory predictors, and UAS/LAS to confirm the thing is
#    a parser and not only a language model.
$PY scripts/lc_surprisal.py --model $OUT/lc_en_ewtgum.pt \
  --conllu $A-test.relabeled_ext.conllu --beam 10 --expand 16 --score \
  --out metrics/en/lc_en_test_surprisal.tsv 2>&1 | tee metrics/en/lc_en_test_score.txt
$PY scripts/lc_surprisal.py --model $OUT/lc_en_selftrained.pt \
  --conllu $A-test.relabeled_ext.conllu --beam 10 --expand 16 --score \
  --out metrics/en/lc_en_test_surprisal_st.tsv 2>&1 | tee metrics/en/lc_en_test_score_st.txt

# 6. Reading times. Provo comes from OSF; the eye-tracking CSV is 58 MB.
#    curl -sSL -o assets_provo/Provo_Corpus-Eyetracking_Data.csv https://osf.io/download/a32be/
for M in lc_en_ewtgum lc_en_selftrained; do
  $PY scripts/lc_provo.py --model $OUT/$M.pt \
    --eyetracking assets_provo/Provo_Corpus-Eyetracking_Data.csv \
    --freq-text "$SCRATCH/en_selftrain.txt" --out metrics/en/provo_$M.tsv --beam 10 --expand 16
  $PY scripts/lc_rt_analysis.py --merged metrics/en/provo_$M.tsv \
    2>&1 | tee metrics/en/provo_${M}_fit.txt
done
