#!/usr/bin/env bash
# Driver for the generic parser v2: UPOS + FEATS + four typological features, no lexical channel.
#
#   scripts/train_generic_v2.sh [stage ...]
#
# Stages, in order, all run if none is named:
#   fetch      the SUD 2.18 release            (once; ~560 MB)
#   split      carve the 16 unsplit corpora
#   inventory  per-corpus stats and exclusions
#   typology   treebank profiles, external profiles, and their agreement
#   prep       the balanced train corpus and the disjoint test corpus
#   convert    CoNLL-U -> .spacy at -n 10
#   configs    one per arm
#   baseline   the five trivial baselines -- MUST precede any arm being quoted
#   check      the go/no-go assertions
#   train      the arms
#   eval       score each arm on the held-out languages
#
# ⚠ ARMS RUN SEQUENTIALLY. spaCy on CPU already saturates the cores through AppleOps, and a v1
# sweep died twice on a fork limit when evaluations ran alongside training.
#
# ⚠ NEVER PIPE A TRAINING COMMAND TO `head`. SIGPIPE truncates the run at its best checkpoint and
# the result reads as converged; `model-best == model-last` is the tell, because a
# patience-terminated run always trains past its best.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CODE="--code scripts/generic_code_v2.py"

CORPUS=${CORPUS:-corpus_generic_v2}
CONLLU=${CONLLU:-assets_generic_v2}
TYP=${TYP:-assets_typ/typology_v2.json}
METRICS=${METRICS:-metrics/generic_v2}

#: The four that carry the claim get three seeds. NEGATIVE-RESULTS.md: one seed once reported
#: +0.46 LAS on a channel whose three-seed mean was +0.04.
# `${VAR-default}`, NOT `${VAR:-default}`: the colon form substitutes the default when
# the variable is unset OR EMPTY, so `DIAG_ARMS=""` silently ran the full diagnostic
# list after it had been explicitly deferred. Four arms trained that nobody asked for.
CLAIM_ARMS=${CLAIM_ARMS-"g2_base g2_typ g2_typ_ctl g2_typ_der"}
CLAIM_SEEDS=${CLAIM_SEEDS:-"0 1 2"}
#: Diagnostics, one seed, labelled single-seed wherever they are reported.
DIAG_ARMS=${DIAG_ARMS-"g2_nofeats g2_langid g2_typ12 g2_feats_all"}
DIAG_SEEDS=${DIAG_SEEDS:-0}

STAGES=${*:-"fetch split inventory typology prep convert configs baseline check train eval"}
mkdir -p "$METRICS"

has () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
run () { echo "+ $*"; "$@"; }

has fetch     && run scripts/fetch_sud_release.sh
has split     && run $PY -u scripts/split_unsplit_sud.py
has inventory && run $PY -u scripts/build_tb_inventory.py
if has typology; then
  run $PY -u scripts/build_typology_v2.py --sensitivity
  run $PY -u scripts/typology_external.py
  # Agreement between the two paths, measured on the languages where both exist. This is the only
  # error bar available on the test-side profiles, so it is part of the pipeline, not an extra.
  run $PY -u scripts/compare_typology.py
fi
has prep && run $PY -u scripts/prep_generic_v2.py

if has convert; then
  # Clear first: a .spacy left over from a previous split is a language silently un-held-out, and
  # the reader discovers languages by which files exist.
  rm -f "$CORPUS"/*.spacy
  mkdir -p "$CORPUS"
  for f in "$CONLLU"/*.conllu; do
    run $PY -m spacy convert "$f" "$CORPUS"/ --converter conllu -n 10 > /dev/null
  done
  echo "converted $(ls "$CORPUS"/*.spacy | wc -l) files"
fi

has configs  && run $PY -u scripts/make_generic_config_v2.py
has baseline && run $PY -u scripts/baseline_generic.py --corpus "$CORPUS" --json "$METRICS/baseline.json"

if has check; then
  run $PY -u scripts/check_generic_inputs_v2.py --corpus "$CORPUS" --conllu "$CONLLU" \
      --typology "$TYP" --baseline "$METRICS/baseline.json" || {
    echo "############## go/no-go FAILED -- not training ##############"; exit 1; }
fi

train_one () {   # arm seed
  local arm=$1 seed=$2 out="training_v2_$1_s$2"
  # A `model-best` directory is NOT a finished run -- spaCy rewrites it at every evaluation, so a
  # killed or crashed arm leaves one behind and the next sweep skips it. The completion marker is
  # in the LOG. (Same shape as "a directory is not a release", CLAUDE.md hazard 1.)
  if grep -q "Saved pipeline" "train_${arm}_s${seed}.log" 2>/dev/null \
     && [ -d "$out/model-best" ] && [ "${SKIP_EXISTING:-1}" = 1 ]; then
    echo "############## SKIP $arm seed $seed (exists; SKIP_EXISTING=0 to force) ##############"
    return
  fi
  echo "############## TRAIN $arm seed $seed -> $out ##############"
  # `python -u`: spaCy buffers when redirected, and two runs in this repo's history looked stalled
  # for hours. model-last's mtime is the reliable progress signal.
  run $PY -u -m spacy train "configs/config_${arm}.cfg" $CODE \
      --output "$out/" --system.seed "$seed" --training.seed "$seed" \
      --corpora.train.seed "$seed" --corpora.dev.seed "$seed" \
      --paths.corpus "$CORPUS" --paths.typology "$TYP" \
      > "train_${arm}_s${seed}.log" 2>&1
}

if has train; then
  for arm in $CLAIM_ARMS; do for seed in $CLAIM_SEEDS; do train_one "$arm" "$seed"; done; done
  for arm in $DIAG_ARMS;  do for seed in $DIAG_SEEDS;  do train_one "$arm" "$seed"; done; done
fi

if has eval; then
  # Iterate the explicit arm list, never a glob: `metrics_g2_typ*` would match `_ctl` and `_der`
  # and average three different arms into one column, which is exactly what happened in v1.
  for arm in $CLAIM_ARMS $DIAG_ARMS; do
    for seed in $CLAIM_SEEDS $DIAG_SEEDS; do
      out="training_v2_${arm}_s${seed}"
      [ -d "$out/model-best" ] || continue
      [ -f "$METRICS/metrics_${arm}_s${seed}.json" ] && [ "${SKIP_EXISTING:-1}" = 1 ] && continue
      run $PY -u scripts/eval_generic_v2.py "$out/model-best" --corpus "$CORPUS" \
          --typology "$TYP" --baseline "$METRICS/baseline.json" --held-in \
          --json "$METRICS/metrics_${arm}_s${seed}.json" \
          > "eval_${arm}_s${seed}.log" 2>&1
      tail -12 "eval_${arm}_s${seed}.log"
    done
  done
fi
echo DONE
