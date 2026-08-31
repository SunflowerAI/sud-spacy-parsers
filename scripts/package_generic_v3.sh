#!/usr/bin/env bash
# Package the v3 arm: guard + morphologiser + tok2vec + parser, plus the ENGLISH gloss table and the
# tool that fits a language row. You supply UPOS, a gloss per token, and Doc._.tb_lang.
#
# ⚠ THIS ARM REFUSES A LANGUAGE IT HAS NO ROW FOR. That is not a defect: the per-language embedding
# is worth +9.45 LAS on held-out languages once fitted on FIFTY annotated sentences, which is more
# than twice what the gloss channel is worth on its own (+5.12). `adapt_lang_embed` ships inside the
# wheel and is the first thing a user runs.
#
# ⚠ THE TABLE IS ENGLISH-ONLY, ON PURPOSE. Source-language rows would be dead weight -- no user's
# language is among the 32 the training table covered -- and dropping them takes 455 MB to 49 MB.
# The shipped fill is therefore `gloss`; `assemble_generic_v3.py` sets it and refuses without it.
#
# ⚠ CC BY-NC-SA 4.0. 24 of the 80 training treebanks are NonCommercial. Not negotiable by
# relabelling the wheel; it is what the training data permits.
set -u
cd "$(dirname "$0")/.."
PY=$(pwd)/.venv/bin/python
SRC=${SRC:-training_v3_bundle}
OUT=${OUT:-build_generic_v3}
NAME=${NAME:-sud_generic}
VERSION=${VERSION:-0.2.0}

[ -d "$SRC" ] || { echo "SRC $SRC missing -- run scripts/assemble_generic_v3.py"; exit 1; }
$PY - "$SRC" <<'CHK' || exit 1
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
cfg = (d / "config.cfg").read_text()
if 'vectors_fill = "gloss"' not in cfg:
    sys.exit("REFUSING: the bundle's fill is not `gloss`; its English-only table would be all-OOV.")
if not list(d.glob("*.npz")):
    sys.exit("REFUSING: no vector table in the bundle -- the channel would fail to load.")
print("bundle checks passed")
CHK

CODE="scripts/sud_feats_embed.py,scripts/sud_generic_embed_v2.py,scripts/sud_generic_embed_v3.py"
CODE="$CODE,scripts/generic_corpus.py,scripts/aligned_vectors.py,scripts/adapt_lang_embed.py"

rm -rf "$OUT"; mkdir -p "$OUT"
META=$OUT/meta_override.json
$PY - "$SRC" "$META" "$VERSION" <<'METAPY'
import json, pathlib, sys
src, out, ver = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
m = json.loads((src / "meta.json").read_text(encoding="utf-8"))
m.update({
    "lang": "xx", "version": ver, "license": "CC BY-NC-SA 4.0", "author": "Siva Kalyan",
    "url": "https://github.com/skalyan91/SUD-spaCy",
    "description": (
        "A language-agnostic SUD parser for a language with no treebank. YOU SUPPLY: gold tokens, "
        "UPOS, an English GLOSS per token (Token._.gloss), and Doc._.tb_lang. The wheel supplies "
        "FEATS and the parse. For a new language, fit one of the 32 spare language-embedding rows "
        "with the bundled `adapt_lang_embed` on ~50 annotated sentences -- worth +9.45 LAS on "
        "held-out languages, with the gloss channel adding +0.70 on top. Trained on 80 SUD 2.18 "
        "treebanks. No tokenizer and no lemmatiser."),
    "sources": [{"name": "SUD 2.18, 80 treebanks", "url": "https://surfacesyntacticud.org/data",
                 "license": "union: CC BY-NC-SA 4.0 (24 of 80 treebanks are NonCommercial)"},
                {"name": "fastText aligned vectors (English rows)",
                 "url": "https://fasttext.cc/docs/en/aligned-vectors.html",
                 "license": "CC BY-SA 3.0"}],
})
out.write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"licence {m['license']}, version {m['version']}")
METAPY
# ⚠ RUN `spacy package` FROM INSIDE THE BUNDLE. It LOADS the model to build the package README, and
# the config names the vector table by BARE FILENAME (so an installed wheel does not chase an
# absolute build-time path). A bare name resolves against the CWD, so packaging from the repo root
# fails to find a table sitting in the bundle -- and `spacy package` then exits having written no
# wheel at all, while still printing "Building package artifacts".
ROOT=$(pwd)
ABS_OUT=$ROOT/$OUT
ABS_META=$ROOT/$META
ABS_CODE=$(echo "$CODE" | tr ',' '\n' | sed "s|^|$ROOT/|" | paste -sd, -)
( cd "$SRC" && $PY -m spacy package . "$ABS_OUT" --name "$NAME" --version "$VERSION" \
    --code "$ABS_CODE" --meta-path "$ABS_META" --force --build wheel ) || exit 1
W=$(find "$OUT" -name '*.whl' | head -1)
echo "built $W"; du -h "$W" | cut -f1
