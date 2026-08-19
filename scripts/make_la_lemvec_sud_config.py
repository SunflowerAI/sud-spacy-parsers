#!/usr/bin/env python3
"""Retrain ONLY `sud_shared` on the lemma-vector base, freezing everything else.

WHY IT NEEDS RETRAINING AT ALL. Standing hazard 5: every MISC pipe reads the base's own
predictions, so a base retrain invalidates the layer above it. Re-measured after the lemvec base
landed, two of the three pipes were fine and one was not:

    Subject, trained   67.02 -> 67.02     unchanged; it reads NORM/PREFIX/SUFFIX/SHAPE, not the parse
    Idiom  / InIdiom   32.00 -> 42.31 / 52.05 -> 56.76   improved, being a CONJUNCTION of upstream
    Shared, trained    38.11 -> 37.67     REGRESSED -- `sud.HeadDepsTagger.v1` pools over the head
                                          and the head's other dependents, so a changed parse is a
                                          changed INPUT, and the old weights were fitted to the old
                                          parser's mistakes

So `sud_shared` alone is rebuilt: sourcing the other seven components and freezing them keeps this a
one-variable change, and keeps `sud_subject` byte-identical rather than re-rolling a pipe that was
already right.

TRAINED THROUGH THE ORTHOGRAPHIC AUGMENTER, like the released layer, because the base underneath is
an augmented arm. `sud_shared` is the least surface-exposed of the three (it reads structure, not
spelling), but the released arm trained it that way and changing two things at once would make the
comparison worthless.

⚠ `annotating_components` MUST list `tok2vec` (docs/sud-misc-layer.md). Without it the frozen base
runs but its predictions never reach the doc the new pipe reads, and the pipe trains against blanks.

    make_la_lemvec_sud_config.py --out configs/config_la_lemvec_sud.cfg
"""
from __future__ import annotations

import argparse

from thinc.api import Config

BASE = ["morphologizer", "lemmatizer", "tok2vec", "parser", "tagger"]
#: sourced and frozen alongside the base -- present so the arm comes out complete, not retrained
KEEP = ["sud_subject", "sud_reported"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/config_la_aug_sud.cfg")
    ap.add_argument("--source", default="training_la_lemvec_misc",
                    help="the arm every frozen component is taken from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config().from_disk(args.template, interpolate=False)

    # the lemvec pipeline order: morphologiser and lemmatiser run BEFORE the parser, because the
    # parser reads their predictions through sud.LemmaVecFeatsEmbed.v1.
    cfg["nlp"]["pipeline"] = BASE + KEEP + ["sud_shared"]

    for name in BASE + KEEP:
        # replace the block wholesale: a sourced component may not also carry `factory` and its
        # settings, and leaving them is how a stale value survives a repoint unnoticed.
        cfg["components"][name] = {"source": args.source}

    cfg["training"]["frozen_components"] = BASE + KEEP
    cfg["training"]["annotating_components"] = list(BASE)

    # sourced components bring their own labels; an initialize block pointed at a labels file they
    # never read is exactly how a stale path survives a rename.
    for name in BASE + KEEP:
        cfg["initialize"].get("components", {}).pop(name, None)

    cfg.to_disk(args.out)
    print(f"wrote {args.out}")
    print(f"  pipeline           {cfg['nlp']['pipeline']}")
    print(f"  frozen             {cfg['training']['frozen_components']}")
    print(f"  annotating         {cfg['training']['annotating_components']}")
    print(f"  trained fresh      ['sud_shared']  (all others sourced from {args.source})")


if __name__ == "__main__":
    main()
