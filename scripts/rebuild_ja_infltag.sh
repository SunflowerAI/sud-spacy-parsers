#!/usr/bin/env bash
# Rebuild the whole ja chain on the Inflection+TAG base.
#
# WHAT CHANGED AND WHY, layer by layer -- each decision is a measured one, three seeds, against a
# zeroed-channel capacity control:
#   base         shared tok2vec gains TAG(128) + Inflection(512)   +1.56 LAS over no channels
#   morphologiser  NO channel: Inflection measured -0.03, ja FEATS is at ceiling (96 % of tokens
#                  carry none), so the encoder stays the stock HashEmbedCNN
#   lemmatiser   own encoder gains Inflection(512)                 +0.65 lemma_acc
#   tagger       warm-started conditioned tagger, grafted behind the morphologiser as every
#                released arm has it -- AND overwrite=true, without which it is a no-op at
#                inference because spacy.ja.JapaneseTokenizer pre-sets every tag
#
# ⚠ EVERY DOWNSTREAM CONFIG MUST USE sud.InflTagCorpus.v1. The base's tok2vec is frozen into each
# layer above it and READS those two channels; a reader that does not stamp them feeds the frozen
# encoder a constant, and nothing raises -- the training loss looks normal and the arm is quietly
# trained out of its own regime. This is the same class of failure as a missing annotating
# component, and the check below is what stops it reaching a wheel.
#
#   bash scripts/rebuild_ja_infltag.sh
set -eu
PY=.venv/bin/python
BASE=training_ja_seg_infltag_s2/model-best      # chosen on DEV (0.9136 LAS), not on test
CORPUS=corpus_ja_infltag
TR=$CORPUS/train.spacy; DV=$CORPUS/dev.spacy; TE=$CORPUS/test.spacy

[ -d "$BASE" ] || { echo "FATAL: $BASE missing"; exit 1; }

# ---- 1. morphologiser -------------------------------------------------------------------------
$PY - <<EOF
from thinc.api import Config
c = Config().from_disk("configs/config_ja_morph.cfg", interpolate=False)  # E913: no interpolation
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflTagCorpus.v1"
for name in ("tok2vec", "tagger", "parser"):
    c["components"][name] = {"source": "$BASE"}
c.to_disk("configs/config_ja_it_morph.cfg")
print("wrote configs/config_ja_it_morph.cfg")
EOF
$PY -u -m spacy train configs/config_ja_it_morph.cfg --code scripts/seg_code.py \
    --output training_ja_it_morph/ --paths.train "$TR" --paths.dev "$DV" \
    > train_ja_it_morph.log 2>&1
echo "morph done"

# ---- 2. lemmatiser (own encoder + Inflection) --------------------------------------------------
$PY - <<'EOF'
from thinc.api import Config
c = Config().from_disk("configs/config_ja_lemma.cfg", interpolate=False)
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflTagCorpus.v1"
for name in ("tok2vec", "tagger", "parser", "morphologizer"):
    c["components"][name] = {"source": "training_ja_it_morph/model-best"}
old = c["components"]["lemmatizer"]["model"]["tok2vec"]
W, D, E = old["width"], old["depth"], old["embed_size"]
c["components"]["lemmatizer"]["model"]["tok2vec"] = {
    "@architectures": "spacy.Tok2Vec.v2",
    "embed": {"@architectures": "sud.MultiHashEmbedFeats.v1", "width": W,
              "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
              "rows": [E, E // 2, E // 2, E // 2], "include_static_vectors": False,
              "feats": ["Inflection"], "feat_rows": [512]},
    "encode": {"@architectures": "spacy.MaxoutWindowEncoder.v2", "width": W, "depth": D,
               "window_size": old["window_size"], "maxout_pieces": old["maxout_pieces"]},
}
c.to_disk("configs/config_ja_it_lemma.cfg")
print("wrote configs/config_ja_it_lemma.cfg")
EOF
$PY -u -m spacy train configs/config_ja_it_lemma.cfg --code scripts/seg_code.py \
    --output training_ja_it_lemma/ --paths.train "$TR" --paths.dev "$DV" \
    > train_ja_it_lemma.log 2>&1
echo "lemma done"

# ---- 3. conditioned tagger, then graft it behind the morphologiser ------------------------------
$PY scripts/make_xpos_config.py configs/config_ja_it_lemma.cfg training_ja_it_lemma/model-best \
    --out configs/config_ja_it_xposwarm.cfg --top \
    --warm-start training_ja_it_lemma/model-best \
    --feats ExtPos --feat-rows 32 --force >/dev/null
$PY - <<'EOF'
from thinc.api import Config
p = "configs/config_ja_it_xposwarm.cfg"
c = Config().from_disk(p, interpolate=False)
for s in ("train", "dev"):
    c["corpora"][s]["@readers"] = "sud.InflTagCorpus.v1"
c["components"]["tagger"]["overwrite"] = True     # the ja no-op fix; see fix_tagger_overwrite.py
c.to_disk(p)
EOF
# prove the conditioning inputs reach the training docs before burning the run
$PY scripts/check_xpos_inputs.py configs/config_ja_it_xposwarm.cfg --train "$TR" --dev "$DV" 2>&1 \
  | grep -E "POS |MORPH |order|ExtPos|FAIL"
$PY -u -m spacy train configs/config_ja_it_xposwarm.cfg --code scripts/seg_code.py \
    --output training_ja_it_xposwarm/ --paths.train "$TR" --paths.dev "$DV" \
    > train_ja_it_xposwarm.log 2>&1
$PY scripts/graft_xpos_tagger.py training_ja_it_lemma/model-best \
    training_ja_it_xposwarm/model-best training_ja_it_graft --corpus "$TE"
echo "graft done"

# ---- 4. the SUD MISC layer, then the fixes that must survive it ---------------------------------
$PY scripts/add_sud_idiom.py training_ja_it_graft training_ja_it_idiom
# graft/add_idiom rebuild the pipeline from pieces, so re-assert the flag on the FINAL arm rather
# than trusting that it travelled. Verified by reloading, never in memory.
$PY scripts/fix_tagger_overwrite.py training_ja_it_idiom
echo "chain rebuilt -> training_ja_it_idiom"
