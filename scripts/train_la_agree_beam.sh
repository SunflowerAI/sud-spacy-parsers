#!/bin/bash
# Latin: agreement handed to the parser as an INPUT, and the parser trained with a BEAM.
# The two changes Sanskrit is trying, each here for a reason measured on Latin rather than copied.
#
# AGREEMENT (configs/config_la_agree.cfg). config_la_lemvec.cfg already gives the parser a
# dimension per morphological category, which tells it that a token is accusative. It cannot tell it
# that TWO tokens share a case, because agreement is a relation and a per-token embedding cannot
# hold one. scripts/sud_lemmavec_embed.py computes the comparison where both tokens are in hand and
# hands over twelve dimensions. The bet is quantified: gold agreeing arcs are 93.5 %
# Case/Number/Gender-compatible against 13.6 % for a random nominal within three tokens that is not
# the head, and 60.8 of those 79.9 points survive the frozen morphologiser this arm actually reads
# (scripts/check_la_agreement_signal.py). Sanskrit built the same block on a 24.1-point gap.
#
# BEAM (configs/config_la_beam.cfg). Latin's parser loses on discontinuity: 37.4 % of test sentences
# carry a crossing arc, those arcs are 5.0 % of tokens but 16.0 % of all attachment errors (UAS
# 28.72 on them against 82.74 in a wholly projective sentence, and the gap survives a length control
# at 8-11.5 points in every bucket), and only 28.4 % of them come back non-projective at all --
# 1 082 emitted against 2 726 in the gold. Under-production is what greedy decoding does with
# pseudo-projective labels: `mod||subj` scores worse than `mod` at that step and only pays after
# de-projectivisation, which is a bet no greedy decoder can take.
#
# ⚠ SANSKRIT'S BEAM ARM LOST. train_sa_beam_s1.log: patience at 8 600 steps, dev LAS 54.08 against
# the greedy arm's 57.14, behind at every matched step. Latin has the stronger premise (37.4 %
# non-projective sentences against 23.97 %) but that is a reason to run it, not to expect it.
#
# ORDER MATTERS: `agree` is ~2.5 h and is the likelier win, `beam` is 15-30 h (eight states through
# the transition system). `combined` is deliberately gated on both winning separately, so that a
# null result is attributable.
#
# Phases: check | agree | beam | combined | eval | why
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
P=la_ittbproiel-sud
S=corpus_la_eval_slices
CODE="--code scripts/seg_code.py"

MACRON_TRAIN=corpus_la_ext_macron/$P-train.relabeled_ext.macron.spacy
DEV=corpus_la_ext_union/dev
TEST=corpus_la_ext/$P-test.relabeled_ext.spacy

phase=${1:-all}

train_one() {  # name config
  local name=$1 cfg=$2
  echo "### TRAIN $name  ($cfg)"
  $PY -u -m spacy train "$cfg" $CODE --output training_la_$name/ \
    --paths.train "$MACRON_TRAIN" --paths.dev "$DEV" > train_la_$name.log 2>&1
  [ -d training_la_$name/model-best ] \
    || { echo "!! FAILED $name"; tail -25 train_la_$name.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_$name.log | tail -1
}

do_check() {
  echo "### CHECK: the agreement block is populated on the PREDICTED doc, not merely configured"
  # A channel that arrives empty does not raise -- it trains, logs normally, and scores exactly like
  # its own capacity control, which is indistinguishable from having been measured and found
  # worthless. The synthetic half runs first because a per-dimension mean cannot tell a correct
  # comparison from a plausible-looking wrong one.
  $PY scripts/check_la_agree_channel.py configs/config_la_agree.cfg --train "$MACRON_TRAIN"
  $PY scripts/check_la_lemvec_inputs.py configs/config_la_agree.cfg --train "$MACRON_TRAIN"
}

do_agree()    { train_one agree      configs/config_la_agree.cfg; }
do_beam()     { train_one beam       configs/config_la_beam.cfg; }
do_combined() { train_one agree_beam configs/config_la_agree_beam.cfg; }

do_eval() {
  echo "### EVAL: lemvec is the baseline -- agree and beam each change exactly one thing about it"
  # Perseus apart: out-of-domain classical verse, and it is where hyperbaton actually lives, so a
  # combined figure would average away the slice both changes are aimed at.
  # model-last as well as model-best, for the reason train_la_lemvec.sh gives: score_weights puts
  # tag_acc at 0.5 against dep_las 0.25, so `model-best` in a PARSING experiment is selected half on
  # the tagger. The weights are left alone to keep selection synchronised with training_la_aug.
  for arm in ${ARMS:-lemvec agree beam agree_beam}; do
   for ck in model-best model-last; do
    d=training_la_$arm/$ck
    [ -d "$d" ] || { echo "== $arm/$ck: MISSING -- skip"; continue; }
    echo "== $arm/$ck"
    for sl in all itp perseus; do
      case "$sl" in
        all) t=$TEST ;;
        *)   t=$S/$sl-test.relabeled_ext.spacy ;;
      esac
      [ -f "$t" ] || { printf "   %-8s MISSING\n" "$sl"; continue; }
      printf "   %-8s " "$sl"
      $PY -m spacy evaluate "$d" "$t" --gold-preproc $CODE \
          --output metrics/la/metrics_la_${arm}_${sl}_${ck}.json 2>/dev/null \
        | grep -E '^(TAG|UAS|LAS) ' | tr -s ' \n' ' '
      echo
    done
   done
  done
}

do_why() {
  # The headline LAS says whether an arm won; these two say whether it won FOR THE REASON IT WAS
  # BUILT. An agreement channel that lifts LAS without touching the agreement-detectable errors, or
  # a beam that lifts LAS without emitting more crossing arcs, is a capacity effect wearing the
  # right name -- and the project has already paid for that once, when morphfirst's entire gain
  # turned out to be its extra embedding rows.
  local args=()
  for arm in ${ARMS:-lemvec agree beam agree_beam}; do
    [ -d training_la_$arm/model-best ] && args+=(--model "$arm=training_la_$arm/model-best")
  done
  [ ${#args[@]} -gt 0 ] || { echo "no arms trained yet"; return; }
  echo "### WHY 1: did the beam recover the non-projective arcs it was chosen for?"
  $PY scripts/analyse_la_nonproj_errors.py "$TEST" "${args[@]}"
  echo; echo "### WHY 2: did the agreement channel fix agreement-detectable errors?"
  $PY scripts/analyse_la_agreement_errors.py "$TEST" "${args[@]}"
}

case "$phase" in
  check)    do_check ;;
  agree)    do_agree ;;
  beam)     do_beam ;;
  combined) do_combined ;;
  eval)     do_eval ;;
  why)      do_why ;;
  all)      do_check; do_agree; do_beam; do_eval; do_why ;;
  *) echo "unknown phase: $phase (check|agree|beam|combined|eval|why)"; exit 1 ;;
esac
echo "DONE: $phase"
