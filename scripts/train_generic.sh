#!/usr/bin/env bash
# Build, train and score the GENERIC (language-agnostic) parser end to end.
#
# The arm reads only UPOS, decomposed FEATS and a cross-lingually aligned 128-d vector -- no
# wordform, no affix, no script, no language id -- over a typologically balanced sample of all
# thirteen SUD treebanks. See docs/generic-parser.md.
#
#   bash scripts/train_generic.sh                  # everything: prep, convert, vectors, train, eval
#   bash scripts/train_generic.sh prep convert     # just rebuild the corpus
#   ARMS="generic generic_ctl" bash scripts/train_generic.sh train eval
#   SEEDS="0 1 2" bash scripts/train_generic.sh train eval     # multi-seed or don't claim it
#   bash scripts/train_generic.sh lolo             # the four zero-shot (leave-one-language-out) arms
#
# ⚠ SEEDS DEFAULTS TO ONE AND THAT IS NOT ENOUGH TO CLAIM A DIFFERENCE. NEGATIVE-RESULTS.md records
# a single seed reporting +0.46 LAS on a channel whose three-seed mean was +0.04, and a Sanskrit
# retrain where one seed said -3 LAS on a change that did nothing. Any comparison between the arm
# and its controls needs `SEEDS="0 1 2"`; a single seed is for checking the pipeline runs.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CODE="--code scripts/generic_code.py"

BUDGET=${BUDGET:-60000}                  # train tokens per typological group
DEV_BUDGET=${DEV_BUDGET:-3000}           # dev tokens per language
ARMS=${ARMS:-"generic generic_ctl generic_shuf generic_nofeats generic_langid"}
SEEDS=${SEEDS:-0}
CORPUS=${CORPUS:-corpus_generic}
CONLLU=${CONLLU:-assets_generic}
TABLE=${TABLE:-assets_vec/generic_vec.npz}
# The four held out one at a time, chosen for typological spread rather than convenience: ja (OV,
# agglutinative, FEATS on 4 % of tokens), ar (VSO-ish, rich FEATS, the only Semitic treebank), ta
# (OV, rich FEATS, and small enough that transfer is the only thing that could help it), lzh
# (isolating, no relatives in the sample except zh/yue -- which STAY IN, so this measures whether
# a related language transfers).
LOLO=${LOLO:-"ja ar ta lzh"}

STAGES=${*:-"prep convert vectors configs train eval"}
run() { echo "+ $*"; "$@"; }
have() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if have prep; then
  echo "############## PREP (budget $BUDGET/group) ##############"
  run $PY scripts/prep_generic.py --out "$CONLLU" --budget "$BUDGET" --dev-budget "$DEV_BUDGET" \
    || exit 1
fi

if have convert; then
  echo "############## CONVERT ##############"
  # -n 10 must match prep_generic.py's --block: the sample was drawn in whole 10-sentence blocks so
  # that a doc is ten CONSECUTIVE sentences. `-l xx` is spaCy's MultiLanguage; the arm has no
  # tokenizer of its own and the vocab is only a string store here.
  rm -rf "$CORPUS"; mkdir -p "$CORPUS"
  for f in "$CONLLU"/*.conllu; do
    run $PY -m spacy convert "$f" "$CORPUS"/ --converter conllu -n 10 -l xx >/dev/null \
      || { echo "!! convert failed: $f"; exit 1; }
  done
  echo "converted $(ls "$CORPUS" | wc -l | tr -d ' ') files"
fi

if have vectors; then
  echo "############## VECTOR TABLE ##############"
  # Keyed off the FULL treebanks, not the sample, so it survives any --budget or --hold-out.
  run $PY scripts/build_generic_vectors.py --out "$TABLE" || exit 1
fi

if have configs; then
  echo "############## CONFIGS ##############"
  run $PY scripts/make_generic_config.py --corpus-conllu "$CONLLU" --corpus "$CORPUS" \
    --table "$TABLE" || exit 1
fi

if have train; then
  for seed in $SEEDS; do
    for arm in $ARMS; do
      out="training_${arm}_s${seed}"
      # A finished arm is not retrained unless asked. Multi-seed sweeps get interrupted -- the
      # first one here died on a fork limit partway through -- and re-running a completed arm
      # costs ~40 min and produces a DIFFERENT model, which would silently invalidate any metrics
      # already written against the old one.
      if [ -d "$out/model-best" ] && [ "${SKIP_EXISTING:-1}" = 1 ]; then
        echo "############## SKIP $arm seed $seed (exists; SKIP_EXISTING=0 to force) ##############"
        continue
      fi
      echo "############## TRAIN $arm seed $seed -> $out ##############"
      # `python -u`: `spacy train` buffers when redirected, and two runs in this repo's history
      # looked stalled for hours because of it. model-last's mtime is the reliable progress signal.
      run $PY -u -m spacy train "configs/config_${arm}.cfg" $CODE \
        --output "$out/" --system.seed "$seed" --training.seed "$seed" \
        --corpora.train.seed "$seed" --corpora.dev.seed "$seed" \
        --paths.corpus "$CORPUS" > "train_${arm}_s${seed}.log" 2>&1
      if [ ! -d "$out/model-best" ]; then
        echo "!! TRAIN $arm s$seed FAILED"; tail -12 "train_${arm}_s${seed}.log"; continue
      fi
      tail -3 "train_${arm}_s${seed}.log"
    done
  done
fi

if have eval; then
  for seed in $SEEDS; do
    for arm in $ARMS; do
      out="training_${arm}_s${seed}"
      [ -d "$out/model-best" ] || continue
      echo "############## EVAL $arm seed $seed ##############"
      run $PY scripts/eval_generic.py "$out/model-best" --corpus "$CORPUS" \
        --manifest "$CONLLU/manifest.json" --json "metrics_${arm}_s${seed}.json" \
        2>&1 | tail -25
    done
  done
fi

if have lolo; then
  # Zero-shot. Each arm re-preps the corpus WITHOUT one language (so its budget is redistributed
  # inside its own group), trains, and is scored on the held-out language's full test set -- which
  # prep_generic.py writes even for a held-out language, because that is the entire point.
  for lang in $LOLO; do
    cdir="${CONLLU}_lolo_${lang}"; sdir="${CORPUS}_lolo_${lang}"
    echo "############## LOLO $lang: prep ##############"
    run $PY scripts/prep_generic.py --out "$cdir" --budget "$BUDGET" \
      --dev-budget "$DEV_BUDGET" --hold-out "$lang" || continue
    rm -rf "$sdir"; mkdir -p "$sdir"
    for f in "$cdir"/*.conllu; do
      $PY -m spacy convert "$f" "$sdir"/ --converter conllu -n 10 -l xx >/dev/null
    done
    # ⚠ the held-out language must have NO train/dev file in the corpus the reader sees, or it is
    # not held out at all. prep writes none; this deletes any left by an earlier run.
    rm -f "$sdir/${lang}-train.spacy" "$sdir/${lang}-dev.spacy"
    run $PY scripts/make_generic_config.py --corpus-conllu "$cdir" --corpus "$sdir" \
      --table "$TABLE" --arm generic --suffix "_lolo_${lang}"
    out="training_generic_lolo_${lang}"
    echo "############## LOLO $lang: train ##############"
    run $PY -u -m spacy train "configs/config_generic_lolo_${lang}.cfg" $CODE \
      --output "$out/" --paths.corpus "$sdir" > "train_generic_lolo_${lang}.log" 2>&1
    if [ ! -d "$out/model-best" ]; then
      echo "!! LOLO $lang FAILED"; tail -12 "train_generic_lolo_${lang}.log"; continue
    fi
    echo "############## LOLO $lang: eval (ZERO-SHOT on $lang, held-in on the rest) ##############"
    run $PY scripts/eval_generic.py "$out/model-best" --corpus "$sdir" \
      --manifest "$cdir/manifest.json" --json "metrics_generic_lolo_${lang}.json" 2>&1 | tail -25
  done
fi

echo "DONE: $STAGES"
