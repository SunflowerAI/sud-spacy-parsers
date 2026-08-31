#!/usr/bin/env python3
"""Assemble the shippable v3 arm: guard + morphologiser + tok2vec + parser, in that order.

The trained v3 arms are `[tok2vec, parser]` only. A wheel also needs the morphologiser that predicts
FEATS from UPOS, and the guard that refuses an untagged doc -- both already exist in the v2 bundle
and neither was retrained, so they are SOURCED rather than rebuilt.

⚠ PIPELINE ORDER IS LOAD-BEARING TWICE, inherited from v2. The morphologiser must precede the parser
because the parser READS FEATS, and the parser's own tok2vec must sit between them because it reads
the FEATS the morphologiser has just written. The morphologiser carries an INLINED copy of its own
encoder, so it does not compete with the parser's tok2vec for a listener -- which is why swapping
the tok2vec and parser underneath it is safe at all.

⚠ THE SHIPPED FILL IS `gloss`, NOT THE TRAINING DEFAULT. The bundled table carries ENGLISH ROWS ONLY,
because a deployed arm fills the channel from an English gloss and no user's language is among the
32 the training table covered. Under `lemma` the channel would therefore be OOV on every token --
which trains, loads, parses and is simply worse. Set here, verified below, and refused if absent.

⚠ AND THE TABLE PATH MUST BE THE BARE FILENAME. spaCy resolves a config path against the CWD, so an
absolute build-time path would send an installed wheel looking for a 49 MB table wherever the user
happens to be standing. The layer falls back to the copy beside its own module inside the package.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                       # noqa: E402
import generic_code_v3             # noqa: E402,F401
from spacy.tokens import Doc       # noqa: E402
from sud_generic_embed_v3 import set_vectors_fill   # noqa: E402

WANT = ["sud_require_upos", "morphologizer", "tok2vec", "parser"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="training_v3_g3_lemb_vec_s0/model-best")
    ap.add_argument("--morph-from", default="training_v2_g2_bundle")
    ap.add_argument("--table", default="assets_vec/generic_vec_v3_en.npz")
    ap.add_argument("--out", default="training_v3_bundle")
    a = ap.parse_args()

    nlp = spacy.load(a.arm)
    if "morphologizer" not in nlp.pipe_names:
        nlp.add_pipe("morphologizer", source=spacy.load(a.morph_from), before="tok2vec")
    if "sud_require_upos" not in nlp.pipe_names:
        nlp.add_pipe("sud_require_upos", first=True)
    if nlp.pipe_names != WANT:
        sys.exit(f"pipeline is {nlp.pipe_names}, want {WANT}")

    embed = nlp.config["components"]["tok2vec"]["model"]["embed"]
    embed["vectors"] = pathlib.Path(a.table).name
    embed["vectors_fill"] = "gloss"
    nlp.config["paths"]["vec_table"] = pathlib.Path(a.table).name
    # ⚠ THE CONFIG IS NOT ENOUGH, AND IT LIES CONVINCINGLY. thinc serialises a Model's `attrs`, so
    # `vt_fill` is baked into tok2vec/model at TRAINING time and restored on load -- overriding
    # whatever the config says. A wheel built without this line ships config.cfg reading
    # `vectors_fill = "gloss"` and a model that runs the LEMMA fill, which on an English-only table
    # is OOV on every token. Set the attr as well, and verify the ATTR, not the config.
    set_vectors_fill(nlp, "gloss")

    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    nlp.to_disk(out)
    shutil.copy(a.table, out / pathlib.Path(a.table).name)

    # Verify from INSIDE the bundle, so the bare table name resolves the way it will inside a
    # wheel (where the layer finds it one level down from the --code modules).
    import os
    cwd = os.getcwd()
    os.chdir(out)
    try:
        _verify(".")
    finally:
        os.chdir(cwd)
    return


def _verify(path):
    # ⚠ VERIFY THE RELOADED MODEL, NEVER THE IN-MEMORY ONE. Assigning to nlp.config does not
    # necessarily survive to_disk the way assigning to a component does, and a wheel that loads and
    # is wrong is this repo's most expensive recurring defect.
    r = spacy.load(path)
    e = r.config["components"]["tok2vec"]["model"]["embed"]
    assert r.pipe_names == WANT, r.pipe_names
    assert e["vectors"].endswith(".npz"), e["vectors"]
    # the RESOLVED attribute, which is what actually runs
    fills = [n.attrs["vt_fill"] for _, pp in r.pipeline if getattr(pp, "model", None) is not None
             for n in pp.model.walk() if n.name == "extract_aligned_vec"]
    assert fills == ["gloss"], f"resolved vt_fill is {fills}, config says {e['vectors_fill']!r}"

    doc = Doc(r.vocab, words=["the", "cat", "sat", "on", "the", "mat"])
    for tok, pos in zip(doc, ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]):
        tok.pos_ = pos
    for tok, g in zip(doc, ["the", "cat", "sit", "on", "the", "mat"]):
        tok._.gloss = g
    doc._.tb_lang = "en"
    out_doc = r(doc)
    assert [t.pos_ for t in out_doc] == ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"], "UPOS moved"
    print(f"reloaded ok: {r.pipe_names}")
    print(f"  config fill={e['vectors_fill']}  RESOLVED fill={fills[0]}  table={e['vectors']}")
    print(f"  parse: {[(t.text, t.head.text, t.dep_) for t in out_doc][:4]}")
    meta = json.loads((pathlib.Path(path) / "meta.json").read_text())
    print(f"verified ({meta.get('lang')})")


if __name__ == "__main__":
    main()
