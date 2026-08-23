#!/usr/bin/env bash
# Does conditioning ja's XPOS tagger on the tokeniser's `Inflection` help -- or is it just capacity?
#
# THREE ARMS, all the ordinary freeze recipe (source + freeze everything, train one fresh tagger
# behind the morphologiser, warm-started from the released one):
#
#   training_ja_xposwarm     BASELINE   side channel = POS + ExtPos            (1 feature channel)
#   training_ja_infl         FEATURE    side channel = POS + ExtPos + Inflection
#   training_ja_infl_ctl     CONTROL    side channel = POS + ExtPos + CtlZero
#
# CtlZero is a FEATS key no token carries, so its 512-row table is allocated, concatenated and
# back-propagated exactly as Inflection's is while carrying no information -- the zeroed-channel
# control this repo used for the zh lexicon and the sa affix arms. Verified: both arms initialise
# to 47,012 tagger parameters. A gain that survives the CONTROL is the feature; a gain that only
# beats the BASELINE could be the extra width. Both comparisons are reported, since going 1 -> 2
# channels is itself known to cost ~0.5-1.4 F here (NEGATIVE-RESULTS.md).
#
#   bash scripts/train_ja_infl.sh
set -u

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
SRC=training_ja_lemma/model-best
SRC_CORPUS=corpus_ja_ext
CORPUS=corpus_ja_infl

# The stamper hard-codes split_mode A. If a released arm ever moves off it, the corpus would be
# stamped with one tokeniser and read by another -- silently, and in the direction that looks fine.
mode=$(grep -A2 'spacy.ja.JapaneseTokenizer' "$SRC/config.cfg" | sed -n 's/^split_mode *= *"\(.*\)"/\1/p')
[ "$mode" = "A" ] || { echo "FATAL: $SRC uses split_mode=$mode, stamper assumes A"; exit 1; }

# 1) stamp the tokeniser's Inflection into the corpus as SudInfl
if [ ! -f "$CORPUS/train.spacy" ]; then
  mkdir -p "$CORPUS"
  $PY scripts/stamp_ja_inflection.py \
    "$SRC_CORPUS/ja_gsd-sud-train.relabeled_ext.spacy" "$CORPUS/train.spacy" \
    "$SRC_CORPUS/ja_gsd-sud-dev.relabeled_ext.spacy"   "$CORPUS/dev.spacy" \
    "$SRC_CORPUS/ja_gsd-sud-test.relabeled_ext.spacy"  "$CORPUS/test.spacy" || exit 1
fi

# 2) configs: identical but for the second feature's NAME, so the comparison is single-variable
for v in "infl Inflection" "infl_ctl CtlZero"; do
  set -- $v
  $PY scripts/make_xpos_config.py configs/config_ja_lemma.cfg "$SRC" \
      --out "configs/config_ja_$1.cfg" --top --warm-start "$SRC" \
      --feats "ExtPos,$2" --feat-rows 32,512 --force >/dev/null || exit 1
  $PY - "configs/config_ja_$1.cfg" <<'EOF'
import sys
from thinc.api import Config
p = sys.argv[1]
c = Config().from_disk(p, interpolate=False)   # interpolate=False: see CLAUDE.md (E913)
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflCorpus.v1"
c.to_disk(p)
EOF
done

# 3) prove the channels reach the training docs BEFORE burning a run. A missing annotating
#    component leaves them constant and nothing raises.
for v in infl infl_ctl; do
  echo "===== check $v ====="
  $PY scripts/check_xpos_inputs.py "configs/config_ja_$v.cfg" \
      --train "$CORPUS/train.spacy" --dev "$CORPUS/dev.spacy" 2>&1 \
    | grep -E "POS |MORPH |order|ExtPos|Inflection|CtlZero|FAIL"
done

# 4) train, sequentially -- two spaCy runs on one machine contend for the same AMX units and the
#    wall-clock comparison stops meaning anything.
for v in infl infl_ctl; do
  echo "########## ja -> training_ja_$v ##########"
  $PY -u -m spacy train "configs/config_ja_$v.cfg" --code scripts/seg_code.py \
      --output "training_ja_$v/" \
      --paths.train "$CORPUS/train.spacy" --paths.dev "$CORPUS/dev.spacy" \
      > "train_ja_$v.log" 2>&1
  if [ -d "training_ja_$v/model-best" ]; then
    $PY -c "import json;p=json.load(open('training_ja_$v/model-best/meta.json'))['performance'];print(f'  $v OK  dev tag_acc {p[\"tag_acc\"]:.4f}')"
  else
    echo "  $v FAILED:"; tail -15 "train_ja_$v.log"
  fi
done

# 5) score every arm through the reader it was TRAINED through. The baseline has no Inflection
#    channel, so the stock gold-preproc reader is the matched one for it; the two new arms must go
#    through InflEvalCorpus or they are scored with an input deleted.
echo "########## test (gold-preproc) ##########"
$PY scripts/eval_ja_infl.py training_ja_xposwarm/model-best "$CORPUS/test.spacy" --plain \
    --label "BASELINE  ExtPos only" --out metrics/ja/metrics_ja_xposwarm_test.json
for v in "infl FEATURE   ExtPos+Inflection" "infl_ctl CONTROL   ExtPos+CtlZero"; do
  set -- $v
  n=$1; shift
  $PY scripts/eval_ja_infl.py "training_ja_$n/model-best" "$CORPUS/test.spacy" \
      --label "$*" --out "metrics/ja/metrics_ja_${n}_test.json"
done

# 6) and the counterfactual: what an unwitting `spacy evaluate --gold-preproc` would have reported
#    for the feature arm, with its channel silently deleted.
echo "########## feature arm, channel DELETED (what plain spacy evaluate would report) ##########"
$PY scripts/eval_ja_infl.py training_ja_infl/model-best "$CORPUS/test.spacy" --plain \
    --label "FEATURE arm, Inflection deleted"
