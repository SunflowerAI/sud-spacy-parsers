#!/usr/bin/env bash
# Does a WIDER CONTEXT WINDOW help the lzh arms?
#
# The base tok2vec is MaxoutWindowEncoder width 96 / depth 4 / window 1 -> a ±4-token receptive
# field, and the parser and tagger both read it. Widening the window is the one lever left that
# plausibly moves the metric after three lexical channels failed (NEGATIVE-RESULTS.md).
#
# ⚠ THE DEPTH-MATCHED CONTROL IS NOT OPTIONAL. `MaxoutWindowEncoder` concatenates 2*window+1
# vectors before each Maxout, so a wider window is also MORE PARAMETERS; `d8ctl` reaches the same
# ±8 receptive field by doubling depth instead, so "more context" and "more capacity" can be told
# apart. Without it a win at w2 is uninterpretable.
# ⚠ seg is a BASE recipe: the morphologiser, the tagger and every sud_* layer have to be rebuilt on
# whichever base wins. This script trains the BASES only.
set -u
PY=.venv/bin/python
ARMS="${ARMS:-w2 w3 d8ctl}"
SUF=relabeled_ext.udep_ruled.punct.rulemerged

for arm in $ARMS; do
  out="training_lzh_seg_${arm}"
  [ -d "$out/model-best" ] && { echo "  $out exists — skip"; continue; }
  echo "=== $out ==="
  $PY -u -m spacy train "configs/config_lzh_seg_${arm}.cfg" --output "$out" \
      --code scripts/seg_code.py > "train_lzh_seg_${arm}.log" 2>&1 \
    || { echo "  $out FAILED — see train_lzh_seg_${arm}.log"; continue; }
done

mkdir -p metrics/lzh
echo; echo "=== test, --gold-preproc (window 1 = the shipped training_lzh_seg) ==="
for arm in seg $(for a in $ARMS; do echo "seg_$a"; done); do
  d="training_lzh_${arm}/model-best"
  [ -d "$d" ] || continue
  $PY -m spacy evaluate "$d" corpus_lzh_trad/lzh_kyoto-sud-test.${SUF}.spacy --gold-preproc \
      --code scripts/seg_code.py --output "metrics/lzh/metrics_lzh_${arm}_gp.json" >/dev/null 2>&1
  $PY - "$arm" <<'PYEOF'
import json,sys,pathlib
p=pathlib.Path(f"metrics/lzh/metrics_lzh_{sys.argv[1]}_gp.json")
if p.exists():
    m=json.loads(p.read_text())
    print(f"  {sys.argv[1]:<14}TAG {100*(m.get('tag_acc') or 0):6.2f}  "
          f"UAS {100*(m.get('dep_uas') or 0):6.2f}  LAS {100*(m.get('dep_las') or 0):6.2f}  "
          f"SENTS_F {100*(m.get('sents_f') or 0):6.2f}")
PYEOF
done
