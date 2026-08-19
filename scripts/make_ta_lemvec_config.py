#!/usr/bin/env python3
"""Derive Tamil's lemma + decomposed-morphology parser config, and its capacity control.

THE RECIPE, and where it necessarily departs from Latin's. `make_la_lemvec_config.py` gives the
parser two channels: a DISTRIBUTIONAL lemma-vector block, and one hash table per morphological
CATEGORY. Tamil gets the second exactly, and the first in its identity form, for a reason that is
about corpus size and not about preference:

    the lemma-VECTOR block is PPMI+SVD over the training treebank's own lemmas. Latin has
    529 809 lemma tokens to build that from. Tamil TTB has 6 329, and TTB+MWTT 8 409. A
    co-occurrence matrix over eight thousand tokens is noise, and `build_lemma_vectors.py`'s
    own `--report` gate exists precisely so a table is looked at rather than trusted.

So the lemma arrives as ITS OWN HASH TABLE (`LEMMA` added to the embed's `attrs`) rather than as a
geometry. That is not a consolation prize: the Sanskrit oracle grid measured gold lemma IDENTITY,
hash-embedded, at **+2.22 LAS**, and the vector block exists to test whether GENERALISATION beyond
identity buys anything further — a question that needs a corpus Tamil does not have. Building the
vectors from external raw Tamil text is the route back to it, and it is blocked on licensing for
the combined arm: TTB is CC BY-NC-SA 3.0, so a table derived from CC BY-SA Wikipedia could not
ship with it, the same conflict that keeps Morpheus out of `la_macronise` and fastText out of la.

WHERE LEMMA AND FEATS COME FROM. The released order runs the parser FIRST, so at inference it has
neither. This moves the already-trained morphologiser and lemmatiser to the FRONT — sourced,
FROZEN, and listed in `annotating_components`, so the parser reads their PREDICTED output during
training as well as at run time. That is what makes the arm shippable rather than an oracle. They
can be moved because the freeze recipe gave each its OWN `HashEmbedCNN`: neither is a listener,
neither reads the parser, so there is no circularity.

WHY DECOMPOSED. spaCy's `MORPH` column is one hash of the WHOLE FEATS bundle, so
`Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` arrive as unrelated symbols and nothing tells the
parser they share a case. Latin measured that as worth exactly nothing over a capacity control
(`config_la_morphfirst.cfg`: LAS 0.7256 against 0.7255). Tamil has more riding on the decomposition
than Latin does: its finite verb agrees with the subject in Person, Number, Gender AND Polite, and
its case system is what distinguishes `comp:obl` from `mod` on the dependents this project's whole
`udep` contribution is about.

THE FEATURE LIST IS READ OFF THE TREEBANK, not written down here — the same reason
`build_feats_inventory.py` exists. Two categories are excluded by name and both exclusions matter:

  * **`Shared`** is not morphology. It is SUD's own coordination annotation, and it is the TARGET
    of the `sud_shared` pipe this project trains as a later layer (`docs/sud-misc-layer.md`).
    Feeding it to the parser as an input would be handing a downstream layer's gold answer to an
    upstream component.
  * **`PunctType`** and **`NumForm`** govern no attachment; a punctuation mark's shape is already
    in NORM and SHAPE.

THE CONTROL (`--control`) is tight, because a loose one is exactly what made the Latin morphfirst
result unreadable for a generation. Same architecture, same NUMBER of hash tables, same rows, same
Maxout input width: every `feats` table is replaced by one more `NORM` table and the `LEMMA` table
by another, all differently seeded, so they add capacity and no information. Any gain over this
control is the two channels and not their parameters.

    make_ta_lemvec_config.py --arm ttb  --out configs/config_ta_ttb_lemvec.cfg
    make_ta_lemvec_config.py --arm both --out configs/config_ta_both_lemvec_ctl.cfg --control
"""
from __future__ import annotations

import argparse
import collections
import pathlib

from thinc.api import Config

#: Not morphology, or no bearing on attachment. See the docstring — `Shared` is the important one.
EXCLUDE = {"Shared", "PunctType", "NumForm"}

#: arm -> (training CoNLL-U used to read the inventory, source arm for the frozen pipes)
ARMS = {
    "ttb": ("assets_ta/ta_ttb-sud-train.conllu", "training_ta_ttb_lemma/model-best"),
    "both": ("assets_ta/ta_ttb_mwtt-sud-train.conllu", "training_ta_both_lemma/model-best"),
}

#: rows per table, as a multiple of the number of values the treebank actually writes (+1 for the
#: "no value" row that `sud_feats_embed` hashes as `Case=`). A table of 8 rows costs 3 kB, so the
#: multiplier is generous on purpose: collisions here are not worth saving bytes over.
ROW_MULTIPLE = 4
MIN_ROWS = 8


def inventory(path):
    """(feature -> number of distinct values, number of distinct lemmas) from a CoNLL-U file."""
    values = collections.defaultdict(set)
    lemmas = set()
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) != 10 or not cols[0].isdigit():
            continue
        if cols[2] != "_":
            lemmas.add(cols[2])
        if cols[5] == "_":
            continue
        for item in cols[5].split("|"):
            if "=" in item:
                key, val = item.split("=", 1)
                values[key].add(val)
    return values, lemmas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--base", default="configs/config_ta_seg.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true",
                    help="capacity control: same tables and Maxout, no information in them")
    ap.add_argument("--source", default=None, help="override the arm to freeze the pipes from")
    args = ap.parse_args()

    train_conllu, default_source = ARMS[args.arm]
    source = args.source or default_source

    values, lemmas = inventory(train_conllu)
    feats = sorted(k for k in values if k not in EXCLUDE)
    feat_rows = [max(MIN_ROWS, ROW_MULTIPLE * (len(values[f]) + 1)) for f in feats]
    # The lemma table is sized off the type inventory, not the token count, and generously: an
    # agglutinative language's lemma list keeps growing with the corpus, so a table sized exactly
    # to train would collide on every unseen lemma the frozen lemmatiser produces.
    lemma_rows = max(2048, ROW_MULTIPLE * len(lemmas))

    cfg = Config().from_disk(args.base, interpolate=False)

    # 1. the two frozen pipes, in FRONT of the parser and annotating during training
    for name in ("morphologizer", "lemmatizer"):
        cfg["components"][name] = {"source": source}
    cfg["nlp"]["pipeline"] = ["morphologizer", "lemmatizer", "tok2vec", "tagger", "parser"]
    cfg["training"]["frozen_components"] = ["morphologizer", "lemmatizer"]
    cfg["training"]["annotating_components"] = ["morphologizer", "lemmatizer"]
    # A sourced component brings its own labels; an `initialize` block pointed at a labels file it
    # never reads is how a stale path survives a rename unnoticed.
    for name in ("morphologizer", "lemmatizer"):
        cfg["initialize"].get("components", {}).pop(name, None)
    # ⚠ NO COLLECTED-LABELS BLOCK HERE, unlike `make_la_lemvec_config.py`. Latin needs one because
    # its arm runs the orthographic augmenter under `max_epochs = -1`, where `init_nlp` initialises
    # from `islice(train_corpus(nlp), 100)` and silently truncates the label set. This config has no
    # augmenter and `max_epochs = 0`, so spaCy lists the WHOLE training corpus and collects every
    # label itself. Copying the block across anyway is what made the first run of this arm die with
    # `ValueError: Can't read file: labels_ta_ttb_lemvec/tagger.json` — a `require = True` pointed
    # at a directory no phase ever built. The order-on-lemvec arm DOES need collected labels, and
    # `make_dravidian_order_config.py` adds them there, where `max_epochs` really is -1.

    # 2. the embed
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    attrs = list(embed["attrs"])
    rows = list(embed["rows"])
    new = {"@architectures": "sud.MultiHashEmbedFeats.v1",
           "width": embed["width"],
           "include_static_vectors": False}
    if args.control:
        # One extra NORM table for the LEMMA channel and one per feature. Differently seeded by
        # construction (`sud_feats_embed` increments the seed per table), so they are capacity
        # without information — which is the whole point.
        new["attrs"] = attrs + ["NORM"] * (1 + len(feats))
        new["rows"] = rows + [lemma_rows] + feat_rows
        new["feats"] = []
        new["feat_rows"] = []
    else:
        new["attrs"] = attrs + ["LEMMA"]
        new["rows"] = rows + [lemma_rows]
        new["feats"] = feats
        new["feat_rows"] = feat_rows
    cfg["components"]["tok2vec"]["model"]["embed"] = new

    cfg.to_disk(args.out)
    what = "capacity control" if args.control else "lemma hash + per-feature morphology"
    print(f"wrote {args.out}  ({what}; frozen+annotating from {source})")
    print(f"  lemma table {lemma_rows} rows over {len(lemmas)} training types")
    print(f"  {len(feats)} feature tables: "
          + ", ".join(f"{f}({len(values[f])}v/{r}r)" for f, r in zip(feats, feat_rows)))


if __name__ == "__main__":
    main()
