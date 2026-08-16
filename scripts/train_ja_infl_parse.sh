#!/usr/bin/env bash
# Does the tokeniser's `Inflection` improve PARSING -- or only XPOS tagging?
#
# The tagger experiment (train_ja_infl.sh) put the channel in the tagger's OWN small encoder, under
# the freeze recipe, so it could not touch the parser: the parser listens to the shared `tok2vec`
# and never saw the feature. Testing parsing means putting the channel in the SHARED encoder, which
# is a BASE retrain -- tok2vec, tagger and parser together, from scratch -- not a stackable layer.
#
# WHY IT MIGHT WORK, mechanically. spaCy's ja tokeniser splits one UniDic analysis in two:
#   XPOS        = "-".join(part_of_speech()[:4])     動詞-非自立可能
#   Inflection  = ";".join(part_of_speech()[4:])     五段-カ行;未然形-一般
# They are DISJOINT slices, so the inflectional form is information the XPOS tagset structurally
# cannot carry -- and it is the part that predicts attachment in Japanese: 連体形 takes a nominal
# head, 連用形 an adverbial/conjunctive one, 終止形 ends a clause. The parser reads tok2vec
# (NORM/PREFIX/SUFFIX/SHAPE), not TAG, so all of this is new to it.
#
#   training_ja_seg_infl       FEATURE   shared embed + Inflection (512 rows)
#   training_ja_seg_infl_ctl   CONTROL   shared embed + CtlZero    (512 rows, dead)
#
# Same zeroed-channel control as the tagger run. The released base arm (training_ja_seg) is context
# only: it was trained through a different reader, and comparing across harnesses is how the lzh
# merge got misreported.
#
#   bash scripts/train_ja_infl_parse.sh [seed ...]     (default seed 0)
set -u
PY=.venv/bin/python
CORPUS=corpus_ja_infl
SEEDS=${*:-0}

[ -f "$CORPUS/train.spacy" ] || { echo "FATAL: run train_ja_infl.sh first (stamps the corpus)"; exit 1; }

for v in "infl Inflection" "infl_ctl CtlZero"; do
  set -- $v
  out="configs/config_ja_seg_$1.cfg"
  $PY - "$out" "$2" <<'EOF'
import sys
from thinc.api import Config
out, feat = sys.argv[1], sys.argv[2]
# interpolate=False, or ${paths.train} resolves to null and the CLI overrides break (E913).
c = Config().from_disk("configs/config_ja_seg.cfg", interpolate=False)
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflCorpus.v1"
e = c["components"]["tok2vec"]["model"]["embed"]
# MultiHashEmbedFeats seeds its first len(attrs) tables identically to MultiHashEmbed, so with the
# same attrs/rows the arm differs from the released base in the new channel and nothing else.
e["@architectures"] = "sud.MultiHashEmbedFeats.v1"
e["feats"] = [feat]
e["feat_rows"] = [512]
c.to_disk(out)
print(f"wrote {out}: attrs={e['attrs']} rows={e['rows']} feats={e['feats']} feat_rows={e['feat_rows']}")
EOF
done

for seed in $SEEDS; do
  for v in infl infl_ctl; do
    arm="training_ja_seg_${v}"; [ "$seed" = 0 ] || arm="${arm}_s${seed}"
    [ -d "$arm/model-best" ] && { echo "  $v s$seed exists -- skip"; continue; }
    echo "########## ja base -> $arm (seed $seed) ##########"
    $PY -u -m spacy train "configs/config_ja_seg_$v.cfg" --code scripts/seg_code.py \
        --output "$arm/" --system.seed "$seed" \
        --paths.train "$CORPUS/train.spacy" --paths.dev "$CORPUS/dev.spacy" \
        > "train_ja_seg_${v}_s${seed}.log" 2>&1
    [ -d "$arm/model-best" ] || { echo "  FAILED:"; tail -15 "train_ja_seg_${v}_s${seed}.log"; continue; }
    # Scored through the reader it was TRAINED through: gold-preproc alone would delete the channel.
    $PY scripts/eval_ja_infl.py "$arm/model-best" "$CORPUS/test.spacy" \
        --label "seed $seed  $v" --out "metrics_ja_seg_${v}_s${seed}.json" \
      | grep -E 'seed|dep_|tag_acc'
  done
done
