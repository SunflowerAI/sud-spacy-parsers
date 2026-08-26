#!/usr/bin/env bash
# Package the generic parser v2 as a wheel: a parser that reads UPOS + FEATS + a per-language
# embedding, plus the tool that fits that embedding for a language it has never seen.
#
# ⚠ THIS WHEEL IS CC BY-NC-SA 4.0. 24 of the 80 training treebanks are NonCommercial -- 276 891 of
# 880 919 tokens, 31 % -- so the union of the corpus licences is NonCommercial and ShareAlike. That
# is not negotiable by relabelling the wheel; it is what the training data permits.
#
# ⚠ THE MODEL IS USELESS ON A NEW LANGUAGE UNTIL ITS EMBEDDING ROW IS FITTED. An unfitted spare row
# is not neutral -- on Georgian it cost 4 LAS against having no channel at all. `adapt_lang_embed`
# ships inside the wheel for exactly this reason, and ten annotated sentences is enough.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

SRC=${SRC:-training_v2_g2_langemb_s0/model-best}
OUT=${OUT:-build_generic_v2}
NAME=${NAME:-sud_generic}
VERSION=${VERSION:-0.1.0}

[ -d "$SRC" ] || { echo "SRC $SRC missing"; exit 1; }

# The layer, the FEATS decomposition and the reader must all travel, or the wheel will not load.
# `adapt_lang_embed` travels too: a model that cannot be given a new language is only half of what
# this release is.
CODE="scripts/sud_feats_embed.py,scripts/sud_generic_embed_v2.py,scripts/adapt_lang_embed.py"

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
        "A language-agnostic SUD dependency parser. Reads UPOS, decomposed FEATS and a trainable "
        "per-language embedding -- no wordform, no script, no vectors. Trained on 80 SUD 2.18 "
        "treebanks. For a language it has not seen, assign one of the 32 spare embedding rows and "
        "fit it with the bundled `adapt_lang_embed`: ten annotated sentences is usually enough. "
        "NOTE the parser expects GOLD (or predicted) UPOS and FEATS on the input Doc and "
        "`Doc._.tb_lang` set to the language; it has no tokenizer and no morphologiser of its own."),
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
