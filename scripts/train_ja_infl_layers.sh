#!/usr/bin/env bash
# Does the tokeniser's `Inflection` help the MORPHOLOGISER and the LEMMATISER?
#
# Both sit above the base under the ordinary freeze recipe, each with its OWN small encoder --
# and that encoder is `spacy.HashEmbedCNN.v2`, which HARD-CODES NORM/PREFIX/SUFFIX/SHAPE and cannot
# express an extra channel (the same reason make_xpos_config.py has to write the stack out by
# hand). So each arm swaps it for the explicit Tok2Vec.v2 stack HashEmbedCNN builds internally --
# MultiHashEmbed rows [E, E/2, E/2, E/2] over MaxoutWindowEncoder -- with MultiHashEmbedFeats in
# the embed slot. With the same attrs/rows that stack is byte-equivalent to the HashEmbedCNN it
# replaces, so the arms differ from the released ones in the new channel and nothing else.
#
#   training_ja_morph_infl / _ctl     morphologiser, own encoder + Inflection / CtlZero
#   training_ja_lemma_infl / _ctl     lemmatiser,    own encoder + Inflection / CtlZero
#
# STACKED ON THE RELEASED BASE, deliberately -- not on training_ja_seg_infl. Stacking on the new
# base would confound "does this component's encoder benefit" with "does the base change help",
# which are separate questions and the base one is already answered.
#
#   bash scripts/train_ja_infl_layers.sh [seed ...]      (default 0 1 2)
set -u
PY=.venv/bin/python
CORPUS=corpus_ja_infl
SEEDS=${*:-0 1 2}

[ -f "$CORPUS/train.spacy" ] || { echo "FATAL: run train_ja_infl.sh first (stamps the corpus)"; exit 1; }

gen() {  # $1=layer (morph|lemma)  $2=component  $3=feature name  $4=out
  $PY - "$1" "$2" "$3" "$4" <<'EOF'
import sys
from thinc.api import Config
layer, comp, feat, out = sys.argv[1:5]
c = Config().from_disk(f"configs/config_ja_{layer}.cfg", interpolate=False)  # E913: no interpolation
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflCorpus.v1"
old = c["components"][comp]["model"]["tok2vec"]
assert old["@architectures"] == "spacy.HashEmbedCNN.v2", old["@architectures"]
W, D, E = old["width"], old["depth"], old["embed_size"]
# exactly what HashEmbedCNN builds internally, plus the one extra table
c["components"][comp]["model"]["tok2vec"] = {
    "@architectures": "spacy.Tok2Vec.v2",
    "embed": {"@architectures": "sud.MultiHashEmbedFeats.v1", "width": W,
              "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
              "rows": [E, E // 2, E // 2, E // 2],
              "include_static_vectors": False,
              "feats": [feat], "feat_rows": [512]},
    "encode": {"@architectures": "spacy.MaxoutWindowEncoder.v2", "width": W, "depth": D,
               "window_size": old["window_size"], "maxout_pieces": old["maxout_pieces"]},
}
c.to_disk(out)
print(f"wrote {out}: {comp} width={W} depth={D} rows=[{E},{E//2},{E//2},{E//2}] feats=[{feat}]")
EOF
}

for spec in "morph morphologizer morph_acc" "lemma lemmatizer lemma_acc"; do
  set -- $spec
  layer=$1 comp=$2 metric=$3
  gen "$layer" "$comp" Inflection "configs/config_ja_${layer}_infl.cfg"     || exit 1
  gen "$layer" "$comp" CtlZero    "configs/config_ja_${layer}_infl_ctl.cfg" || exit 1
  for seed in $SEEDS; do
    for v in infl infl_ctl; do
      arm="training_ja_${layer}_${v}_s${seed}"
      [ -d "$arm/model-best" ] && { echo "  $layer $v s$seed exists -- skip"; continue; }
      echo "########## ja $layer $v seed $seed ##########"
      $PY -u -m spacy train "configs/config_ja_${layer}_${v}.cfg" --code scripts/seg_code.py \
          --output "$arm/" --system.seed "$seed" \
          --paths.train "$CORPUS/train.spacy" --paths.dev "$CORPUS/dev.spacy" \
          > "train_ja_${layer}_${v}_s${seed}.log" 2>&1
      [ -d "$arm/model-best" ] || { echo "  FAILED:"; tail -12 "train_ja_${layer}_${v}_s${seed}.log"; continue; }
      # matched reader: gold-preproc alone deletes the channel these arms were trained with
      $PY scripts/eval_ja_infl.py "$arm/model-best" "$CORPUS/test.spacy" \
          --label "$layer seed $seed $v" | grep -E "seed|$metric"
    done
  done
done
