#!/usr/bin/env bash
# Package the generic arm as ONE wheel: a morphologiser that predicts FEATS from UPOS, feeding a
# parser that reads UPOS + FEATS + a per-language embedding, plus the tool that fits that embedding
# for a language it has never seen. The user supplies UPOS; the wheel supplies everything else.
#
# ⚠ PIPELINE ORDER IS LOAD-BEARING TWICE. The morphologiser must precede the parser because the
# parser READS FEATS, and the parser's own tok2vec must sit between them because it reads the FEATS
# the morphologiser has just written: [morphologizer, tok2vec, parser]. The morphologiser also
# carries an INLINED copy of its own encoder (`replace_listeners`), because two listeners in one
# pipeline would otherwise both resolve to whichever tok2vec is present and silently read the wrong
# one -- which produced empty FEATS on the in-memory assembly and was only caught on reload.
#
# ⚠ THERE IS NO LEMMATISER, DELIBERATELY. Across six held-out languages and two architectures an
# edit-tree lemmatiser never deviated from copying the wordform by more than +0.31 points.
#
# ⚠ THIS WHEEL IS CC BY-NC-SA 4.0. 24 of the 80 training treebanks are NonCommercial -- 276 891 of
# 880 919 tokens, 31 % -- so the union of the corpus licences is NonCommercial and ShareAlike. That
# is not negotiable by relabelling the wheel; it is what the training data permits.
#
# ⚠ THE MODEL IS USELESS ON A NEW LANGUAGE UNTIL ITS EMBEDDING ROW IS FITTED. An unfitted spare row
# is not neutral -- on Georgian it cost 4 LAS against having no channel at all. `adapt_lang_embed`
# ships inside the wheel for exactly this reason, and ten annotated sentences is enough.
#
# ⚠ UPOS IS AN INPUT AND MUST NOT COME BACK OUT. spaCy's morphologiser predicts a joint
# `POS=X|Feat=Val` label and writes BOTH halves when `overwrite` is on, so 0.1.0 as first shipped
# replaced the user's UPOS with its own guess -- and clobbered supplied FEATS with it (40 % of
# gold-FEATS tokens in Latin). `prepare_generic_bundle.py` sets `overwrite = false` and inserts the
# `sud_require_upos` guard, and this script REFUSES to package a bundle where it has not run: a
# default plus a refusal, because a comment telling the next person is not the fix.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

SRC=${SRC:-training_v2_g2_bundle}
OUT=${OUT:-build_generic_v2}
NAME=${NAME:-sud_generic}
VERSION=${VERSION:-0.1.0}

[ -d "$SRC" ] || { echo "SRC $SRC missing"; exit 1; }

# Make the arm shippable, then refuse if it still is not. Idempotent: a bundle already carrying
# `overwrite = false` and the guard is left untouched, and the parse digest is checked either way.
$PY scripts/prepare_generic_bundle.py "$SRC" || exit 1
$PY scripts/prepare_generic_bundle.py "$SRC" --check || {
    echo "REFUSING to package: $SRC still writes the UPOS it was given"; exit 1; }

# The layer, the FEATS decomposition and the reader must all travel, or the wheel will not load.
# `adapt_lang_embed` travels too: a model that cannot be given a new language is only half of what
# this release is.
CODE="scripts/sud_feats_embed.py,scripts/sud_generic_embed_v2.py,scripts/generic_tag_corpus.py,scripts/adapt_lang_embed.py"

# ⚠ This wipes $OUT. Keep nothing there you want to survive a rebuild -- the release notes live in
# docs/release-notes-generic-v0.1.0.md for exactly that reason, having once been deleted mid-release.
rm -rf "$OUT"
mkdir -p "$OUT"

# ⚠ `spacy package` copies the licence straight out of the source meta.json, and a training run
# leaves that field EMPTY. The repo has shipped eleven wheels with a blank `License:` once already;
# write it explicitly rather than inherit a blank.
META=$OUT/meta_override.json
$PY - "$SRC" "$META" <<'METAPY'
import json, pathlib, sys
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
m = json.loads((src / "meta.json").read_text(encoding="utf-8"))
m.update({
    "lang": "xx",
    "license": "CC BY-NC-SA 4.0",
    "author": "Siva Kalyan",
    "url": "https://github.com/skalyan91/SUD-spaCy",
    "description": (
        "A language-agnostic SUD pipeline: a morphologiser that predicts FEATS from UPOS, and a "
        "dependency parser that reads UPOS + FEATS + a trainable per-language embedding. Trained "
        "on 80 SUD 2.18 treebanks. YOU SUPPLY UPOS and `Doc._.tb_lang`; the wheel supplies FEATS "
        "and the parse. For a language it has not seen, assign one of the 32 spare embedding rows "
        "and fit it with the bundled `adapt_lang_embed` -- ten annotated sentences is usually "
        "enough. There is no tokenizer and no lemmatiser: the first is your business, and the "
        "second was measured to be inert on unseen languages."),
    # 24 of the 80 training treebanks are NonCommercial -- 31 % of tokens -- so the union of the
    # corpus licences is NC and ShareAlike, and no relabelling of the wheel changes that.
    "sources": [{"name": "SUD 2.18, 80 treebanks",
                 "url": "https://surfacesyntacticud.org/data",
                 "license": "union: CC BY-NC-SA 4.0 (24 of 80 treebanks are NonCommercial)"}],
})
out.write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"licence set to {m['license']}")
METAPY
$PY -m spacy package "$SRC" "$OUT" --name "$NAME" --version "$VERSION" \
    --code "$CODE" --meta-path "$META" --force --build wheel || exit 1

W=$(find "$OUT" -name '*.whl' | head -1)
echo "built $W"
echo "size: $(du -h "$W" | cut -f1)"
