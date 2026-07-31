#!/bin/bash
# Build the Latin macronisation lookup table LOCALLY and attach it to a local copy of the model.
#
# WHY THIS IS A SCRIPT AND NOT A SHIPPED DATA FILE
# ------------------------------------------------
# The vowel-length data ultimately comes from **Morpheus** (Perseus Project, CC BY-SA 3.0 US),
# reached through Johan Winge's **latin-macronizer** (GPL-3.0). CC BY-SA permits commercial use but
# forbids imposing further restrictions on the work -- and the released Latin model is CC BY-NC-SA,
# because SUD_Latin-ITTB / PROIEL / Perseus are all NonCommercial. Bundling Morpheus-derived
# content into an NC wheel would add exactly the restriction BY-SA rules out.
#
# So the repository ships the BUILDER (ours, MIT) and never the TABLE. You generate the table on
# your own machine from your own macroniser run; nothing Morpheus-derived is ever redistributed.
# `scripts/la_macron_lut.json.gz` and `build_la_macron/` are gitignored for this reason, and
# `package_lemma.sh` deliberately does NOT add `la_macronise` to the released wheel.
#
# NB the macroniser's own tagger is RFTagger, which is licensed for non-commercial use only. It is
# used here only to LABEL your treebank offline; the resulting component uses this project's own
# morphologiser at inference, so RFTagger is not a dependency of anything that runs later.
#
# Usage:
#   bash scripts/build_la_macron.sh [SRC_MODEL] [OUT_MODEL]
# Defaults: training_la_lemma/model-best -> build_la_macron/model
set -e
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
P=la_ittbproiel-sud
SRC=${1:-training_la_lemma/model-best}
OUT=${2:-build_la_macron/model}
LUT=scripts/la_macron_lut.json.gz

echo "### 1/4  what still needs macronising?"
# Work this out BEFORE touching Docker: the macroniser image is amd64-only and takes minutes to
# boot under emulation, so starting it when every output already exists is pure waste.
TODO=""
for s in train dev test; do
  [ -f "assets_la/$P-$s.macron.conllu" ] || TODO="$TODO $s"
done
if [ -z "$TODO" ]; then
  echo "  nothing to do -- all assets_la/$P-*.macron.conllu already present"
else
  echo " needed:$TODO"
  echo "### ---  starting the macroniser container (amd64 emulation -- slow to boot)"
  if ! curl -s -o /dev/null -m 5 http://localhost:51234/macronize 2>/dev/null; then
    docker rm -f macronizer >/dev/null 2>&1 || true
    docker run -d --name macronizer --platform linux/amd64 \
      -p 51234:105 -e PYTHONUNBUFFERED=1 vedph2020/macronizer:0.1.3 >/dev/null
    echo -n "  waiting for the API"
    for _ in $(seq 1 60); do
      curl -s -o /dev/null -m 3 http://localhost:51234/macronize && break
      echo -n "."; sleep 5
    done; echo
    curl -s -o /dev/null -m 3 http://localhost:51234/macronize \
      || { echo "  !! macroniser API never came up"; exit 1; }
  fi
fi

echo "### 2/4  macronise the treebank"
for s in $TODO; do
  in=assets_la/$P-$s.conllu
  [ -f "$in" ] || { echo "  !! missing $in -- run scripts/add_perseus_la.sh first"; exit 1; }
  $PY scripts/macronise_la.py "$in" "assets_la/$P-$s.macron.conllu"
done

echo "### 3/4  harvest the lookup table"
$PY scripts/build_la_macron_lut.py \
  assets_la/$P-train.conllu assets_la/$P-train.macron.conllu "$LUT"

echo "### 4/4  attach la_macronise to a LOCAL copy of the model"
[ -d "$SRC" ] || { echo "  !! missing $SRC"; exit 1; }
mkdir -p "$(dirname "$OUT")"
$PY scripts/add_la_macronise.py "$SRC" "$OUT" --lut "$LUT"

cat <<EOF

Done. Use it with:
    import spacy
    nlp = spacy.load("$OUT")        # needs --code scripts/la_macronise.py if packaged
    print(nlp("Gallia est omnis divisa in partes tres.")._.macron)

Score it against the macroniser with:
    $PY scripts/eval_la_macronise.py $SRC \\
        assets_la/$P-test.conllu assets_la/$P-test.macron.conllu

$OUT contains Morpheus-derived data -- keep it local, do not redistribute it.
EOF
