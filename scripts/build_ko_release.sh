#!/bin/bash
# The ko release chain for the analyser-channel arm: seg base -> morphologiser -> lemmatiser,
# then the wheel and the checks that a wheel has to pass.
#
# ⚠ WHY THIS IS NOT `train_ko_analyser.sh seeds` WITH A DIFFERENT NAME. That driver trains the
# MEASUREMENT arm, on `config_ko_eojeol.cfg`'s single-sentence reader, which is what makes the
# channel's contribution single-variable. It cannot ship: fed two sentences it returns ONE, with a
# single self-headed root, because it never sees an example with a boundary in it. Its
# `--gold-preproc` SENT F of 99.70 says nothing — every example there is already one sentence
# (CLAUDE.md hazard 4). `seg` is a BASE recipe, not a stackable layer, so the release arm is the
# same one-block change applied to `config_ko_eojeol_seg.cfg` and trained from scratch.
#
# Phases (run all, or name one): seeds | pick | stack | graft | raw | wheel | verify
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE="--code scripts/seg_code.py"
export MECAB_PATH="${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}"

P=ko_gsd-sud
TRAIN=corpus_ko_eojeol/$P-train.relabeled_ext.spacy
DEV=corpus_ko_eojeol/$P-dev.relabeled_ext.spacy
TEST=corpus_ko_eojeol/$P-test.relabeled_ext.spacy
SEEDS="${SEEDS:-0 1 2}"
#: set by `pick`, overridable. The arm the stack and the wheel are built on.
PICK_FILE=.ko_release_pick
#: the arm that ships — the grafted one, not the lemma arm it was grafted from
ARM="${ARM:-training_ko_an_senter/model-best}"

do_seeds() {
  for s in $SEEDS; do
    out=training_ko_anseg_s$s
    echo "### $out"
    $PY -u -m spacy train configs/config_ko_analyser_seg.cfg $CODE --output "$out/" \
      --paths.train "$TRAIN" --paths.dev "$DEV" --system.seed "$s" \
      > "train_ko_anseg_s$s.log" 2>&1
    [ -d "$out/model-best" ] || { echo "!! FAILED"; tail -20 "train_ko_anseg_s$s.log"; exit 1; }
    grep -E '^[[:space:]]*[0-9]' "train_ko_anseg_s$s.log" | tail -1
  done
}

do_pick() {
  # ⚠ SELECT ON DEV. The test set is reported, never chosen on; spaCy's own `model-best` is already
  # a dev choice, and picking the seed on test would be a second, hidden one.
  #
  # ⚠ NOT FROM THE LOG. Two traps there, both of which produced a wrong pick before this was
  # rewritten: the SCORE column is printed to two decimals, so three seeds a whole LAS point apart
  # all read `0.81` and the tie broke on whichever ran first; and the columns are
  # (E, #, 3 losses, TAG, POS, UAS, LAS, SENTS, SCORE), so the obvious `$8` is UAS, not LAS.
  # Re-evaluating on dev costs seconds and reports full precision.
  echo "### PICK: best dev LAS across seeds, re-evaluated at full precision"
  best=""; bestv=0
  for s in $SEEDS; do
    d=training_ko_anseg_s$s/model-best
    [ -d "$d" ] || continue
    read -r las uas sents <<<"$($PY -m spacy evaluate "$d" "$DEV" $CODE 2>/dev/null \
      | awk '/^LAS/{l=$2} /^UAS/{u=$2} /^SENT F/{s=$3} END{print l, u, s}')"
    printf "   seed %s  dev LAS %s  UAS %s  SENT F %s\n" "$s" "$las" "$uas" "$sents"
    if awk "BEGIN{exit !($las > $bestv)}"; then bestv=$las; best=$s; fi
  done
  [ -n "$best" ] || { echo "!! no arms to pick from"; exit 1; }
  echo "training_ko_anseg_s$best/model-best" > $PICK_FILE
  echo "   -> $(cat $PICK_FILE)  (dev LAS $bestv)"
}

do_stack() {
  base=$(cat $PICK_FILE)
  echo "### STACK: morphologiser then lemmatiser, by the freeze recipe, on $base"
  # The freeze recipe: source the arm's components, FREEZE them, train only the new one with its own
  # small HashEmbedCNN. Frozen components must come out byte-identical, which `verify` asserts.
  $PY scripts/make_ko_stack_configs.py "$base"
  $PY -u -m spacy train configs/config_ko_anseg_morph.cfg $CODE --output training_ko_anseg_morph/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_anseg_morph.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_anseg_morph.log | tail -1
  $PY -u -m spacy train configs/config_ko_anseg_lemma.cfg $CODE --output training_ko_anseg_lemma/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_anseg_lemma.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_anseg_lemma.log | tail -1
}

do_graft() {
  # ⚠ THE STEP THE RELEASED ko ARM NEVER HAD. `ko_sud_gsd-0.2.0` ships a tagger that is a LISTENER
  # sitting BEFORE its morphologiser, so `package_sud.sh ko` refuses to rebuild the wheel that is
  # live — ko was never grafted, and the guard is right to say so. The graft moves the tagger behind
  # the morphologiser and conditions it on predicted UPOS+MORPH through `sud.Tok2VecPlusFeats.v1`,
  # ABOVE the encoder (docs/xpos.md).
  #
  # ⚠ WARM-START FROM THIS ARM'S OWN TAGGER, not the released one. `train_xpos.sh`'s table names
  # `training_ko_eojeol_lemma` for ko, whose tagger reads 72.51 against this arm's 88.60 — starting
  # the conditioned tagger there would throw the channel away and then have to relearn it. This is
  # the same reason `XPOS_SRC_ARM` exists for the vocalisation chains: a warm start has to come from
  # a tagger trained on the same input regime. The label set is the treebank's own either way, so
  # the ORDER matches and `--warm-start` accepts it (standing hazard 7).
  echo "### GRAFT: the conditioned tagger, behind the morphologiser"
  $PY scripts/make_xpos_config.py configs/config_ko_anseg_lemma.cfg \
      training_ko_anseg_lemma/model-best --out configs/config_ko_anseg_xposwarm.cfg \
      --top --warm-start training_ko_anseg_lemma/model-best --force
  # A missing annotating component leaves the new channels constant and NOTHING raises, so the
  # inputs are verified before the run rather than after it.
  $PY scripts/check_xpos_inputs.py configs/config_ko_anseg_xposwarm.cfg \
      --train "$TRAIN" --dev "$DEV" 2>&1 | grep -E "POS |MORPH |order|FAIL" || true
  $PY -u -m spacy train configs/config_ko_anseg_xposwarm.cfg $CODE \
    --output training_ko_anseg_xposwarm/ --paths.train "$TRAIN" --paths.dev "$DEV" \
    > train_ko_anseg_xposwarm.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_anseg_xposwarm.log | tail -1
}

do_altstack() {
  # THE ALTERNATIVE CHAIN: keep the parser trained on single sentences — the one that reads 74.45
  # LAS under gold sentences, 1.57 above the seg recipe — and buy segmentation from a SEPARATE
  # component instead of from the parser. Everything else is the same recipe on a different base.
  local base=${ALT_BASE:-training_ko_analyser_s2/model-best}
  echo "### ALTSTACK: morphologiser, lemmatiser and the graft on $base"
  $PY scripts/make_ko_stack_configs.py "$base" --prefix an
  $PY -u -m spacy train configs/config_ko_an_morph.cfg $CODE --output training_ko_an_morph/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_an_morph.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_an_morph.log | tail -1
  $PY -u -m spacy train configs/config_ko_an_lemma.cfg $CODE --output training_ko_an_lemma/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_an_lemma.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_an_lemma.log | tail -1
  $PY scripts/make_xpos_config.py configs/config_ko_an_lemma.cfg training_ko_an_lemma/model-best \
      --out configs/config_ko_an_xposwarm.cfg --top \
      --warm-start training_ko_an_lemma/model-best --force
  $PY -u -m spacy train configs/config_ko_an_xposwarm.cfg $CODE \
    --output training_ko_an_xposwarm/ --paths.train "$TRAIN" --paths.dev "$DEV" \
    > train_ko_an_xposwarm.log 2>&1
  grep -E '^[[:space:]]*[0-9]' train_ko_an_xposwarm.log | tail -1
}

do_senter() {
  echo "### SENTER: a standalone sentenciser, grafted in front of the parser"
  $PY scripts/make_ko_senter_config.py training_ko_an_xposwarm/model-best \
      --out configs/config_ko_an_senter.cfg
  $PY -u -m spacy train configs/config_ko_an_senter.cfg $CODE --output training_ko_an_senter/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_ko_an_senter.log 2>&1
  [ -d training_ko_an_senter/model-best ] \
    || { echo "!! FAILED"; tail -20 train_ko_an_senter.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_ko_an_senter.log | tail -1
}

do_compare() {
  # The only comparison that decides anything: RAW end to end, both chains, one command.
  echo "### COMPARE: raw end-to-end, the model finding its own sentences"
  printf "%-30s %7s %7s %7s %7s\n" arm TAG UAS LAS SENT_F
  for arm in eojeol_lemma anseg_xposwarm an_senter; do
    d=training_ko_$arm/model-best
    [ -d "$d" ] || { printf "%-30s MISSING\n" "$arm"; continue; }
    printf "%-30s " "$arm"
    $PY -m spacy evaluate "$d" "$TEST" $CODE --output metrics/ko/metrics_ko_${arm}_raw.json 2>/dev/null \
      | awk '/^TAG/{t=$2} /^UAS/{u=$2} /^LAS/{l=$2} /^SENT F/{s=$3} END{printf "%7s %7s %7s %7s\n", t, u, l, s}'
  done
}

do_wheel() {
  # 0.3.0, not a re-clobber of 0.2.0: the 0.2.0 set has been re-clobbered in place as layers landed,
  # so `pip install -U` is inert for it (CLAUDE.md). ja, la and sa took the same bump for the same
  # reason. A change of this size has to be one users can actually pull.
  local version="${VERSION:-0.3.0}"
  echo "### WHEEL: package $ARM as ko_sud_gsd $version"
  # KO_BASE names the arm; the --code list in package_sud.sh already carries sud_ko_embed.py and
  # ko_analyser.py, and pkg() refuses to build without them.
  KO_BASE="$ARM" VERSION="$version" bash scripts/package_sud.sh ko
}

do_verify() {
  echo "### VERIFY: the frozen layers, the segmentation, then the INSTALLED wheel"
  bash scripts/verify_ko_release.sh
}

do_raw() {
  echo "### RAW: end-to-end, the model finding its own sentences"
  $PY -m spacy evaluate "$ARM" "$TEST" $CODE \
      --output metrics/ko/metrics_ko_anseg_raw.json | grep -E '^(TOK|TAG|POS|MORPH|LEMMA|UAS|LAS|SENT)'
  echo "--- and with gold sentences, for comparison with everything else in docs/korean.md"
  $PY -m spacy evaluate "$ARM" "$TEST" --gold-preproc $CODE \
      --output metrics/ko/metrics_ko_anseg_gp.json | grep -E '^(TAG|UAS|LAS|SENT)'
}

phase=${1:-all}
case "$phase" in
  seeds)  do_seeds ;;
  graft)  do_graft ;;
  altstack) do_altstack ;;
  senter) do_senter ;;
  compare) do_compare ;;
  pick)   do_pick ;;
  stack)  do_stack ;;
  wheel)  do_wheel ;;
  verify) do_verify ;;
  raw)    do_raw ;;
  all)    do_seeds; do_pick; do_stack; do_graft; do_raw; do_wheel; do_verify ;;
  *) echo "unknown phase: $phase (seeds|pick|stack|graft|altstack|senter|compare|wheel|verify|raw)"; exit 1 ;;
esac
echo "DONE: $phase"
