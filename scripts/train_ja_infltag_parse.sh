#!/usr/bin/env bash
# Does the TOKENISER'S XPOS, on top of Inflection, buy anything more for parsing?
#
# XPOS is `"-".join(part_of_speech()[:4])` and Inflection is `";".join(part_of_speech()[4:])` --
# disjoint halves of one UniDic analysis. The parser already has the second half (+0.94 LAS over
# its control). This asks whether the first half adds to it. The tokeniser's tag is free (set at
# tokenisation, before any component runs) but NOISY: 0.7673 against gold XPOS, where the trained
# tagger reaches 0.9457. Handed over raw, uncorrected -- pre-correcting a noisy source has
# destroyed the signal every time this repo tried it.
#
#   training_ja_seg_infltag       FEATURE   attrs +TAG(128)   feats [Inflection](512)
#   training_ja_seg_infltag_ctl   CONTROL   attrs             feats [Inflection](512), CtlZero(128)
#
# THE CONTROL IS SHAPED THIS WAY ON PURPOSE. A dead channel cannot be made from a fake ATTR name
# (spaCy attrs are a fixed enum), only from a fake FEATS key. MultiHashEmbedFeats builds one table
# per entry of attrs+feats and concatenates them all, so a 128-row table costs the same wherever it
# sits in that list: both arms get six tables with rows {5000,1000,2500,2500,512,128} and the same
# Maxout input width. Asserted below, not argued.
#
# Both arms read the SAME corpus through the SAME reader, so the control's predicted docs carry the
# tag too -- its embed simply does not look at it. That keeps the data path identical and the
# comparison single-variable.
#
#   bash scripts/train_ja_infltag_parse.sh [seed ...]      (default 0 1 2)
set -u
PY=.venv/bin/python
CORPUS=corpus_ja_infltag
SEEDS=${*:-0 1 2}

[ -f "$CORPUS/train.spacy" ] || { echo "FATAL: stamp first: stamp_ja_inflection.py --tag ..."; exit 1; }

$PY - <<'EOF' || exit 1
from thinc.api import Config
BASE = "configs/config_ja_seg.cfg"
ARMS = {
    "infltag":     dict(attrs=["NORM","PREFIX","SUFFIX","SHAPE","TAG"],
                        rows=[5000,1000,2500,2500,128],
                        feats=["Inflection"], feat_rows=[512]),
    "infltag_ctl": dict(attrs=["NORM","PREFIX","SUFFIX","SHAPE"],
                        rows=[5000,1000,2500,2500],
                        feats=["Inflection","CtlZero"], feat_rows=[512,128]),
}
for name, spec in ARMS.items():
    c = Config().from_disk(BASE, interpolate=False)      # interpolate=False: CLAUDE.md (E913)
    for s in ("train", "dev"):
        c["corpora"][s]["@readers"] = "sud.InflTagCorpus.v1"
    e = c["components"]["tok2vec"]["model"]["embed"]
    e["@architectures"] = "sud.MultiHashEmbedFeats.v1"
    e.update(spec)
    out = f"configs/config_ja_seg_{name}.cfg"
    c.to_disk(out)
    print(f"wrote {out}: {len(spec['attrs'])} attrs + {len(spec['feats'])} feats, "
          f"rows {sorted(spec['rows'] + spec['feat_rows'])}")
EOF

# The capacity claim is CHECKED before any run: an unequal control proves nothing.
$PY - <<'EOF' || exit 1
import sys
sys.path.insert(0, "scripts"); import seg_code  # noqa: F401
import spacy
from spacy.training.initialize import init_nlp
def params(p):
    cfg = spacy.util.load_config(p, overrides={
        "paths.train": "corpus_ja_infltag/train.spacy",
        "paths.dev": "corpus_ja_infltag/dev.spacy"}, interpolate=True)
    nlp = init_nlp(cfg)
    m = nlp.get_pipe("tok2vec").model
    return sum(n.get_param(x).size for n in m.walk() for x in n.param_names if n.has_param(x))
a = params("configs/config_ja_seg_infltag.cfg")
b = params("configs/config_ja_seg_infltag_ctl.cfg")
print(f"tok2vec params  FEATURE {a:,}   CONTROL {b:,}   {'IDENTICAL' if a == b else 'DIFFER'}")
sys.exit(0 if a == b else 1)
EOF

for seed in $SEEDS; do
  for v in infltag infltag_ctl; do
    arm="training_ja_seg_${v}_s${seed}"
    [ -d "$arm/model-best" ] && { echo "  $v s$seed exists -- skip"; continue; }
    echo "########## ja base -> $arm ##########"
    $PY -u -m spacy train "configs/config_ja_seg_$v.cfg" --code scripts/seg_code.py \
        --output "$arm/" --system.seed "$seed" \
        --paths.train "$CORPUS/train.spacy" --paths.dev "$CORPUS/dev.spacy" \
        > "train_ja_seg_${v}_s${seed}.log" 2>&1
    [ -d "$arm/model-best" ] || { echo "  FAILED:"; tail -12 "train_ja_seg_${v}_s${seed}.log"; continue; }
    $PY scripts/eval_ja_infl.py "$arm/model-best" "$CORPUS/test.spacy" --reader infltag \
        --label "seed $seed  $v" --out "metrics_ja_seg_${v}_s${seed}.json" \
      | grep -E 'seed|dep_|tag_acc'
  done
done
