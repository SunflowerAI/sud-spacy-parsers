#!/usr/bin/env python3
"""Derive a verb-distance parser config for lzh from its own from-scratch base recipe.

WHY THIS BASE. `configs/config_lzh_resplit_ctl.cfg` trains tok2vec+tagger+parser TOGETHER from
scratch (`frozen_components = []`, `pipeline = ["tok2vec", "tagger", "parser"]`) -- it is the
config that actually produced `training_lzh_resplit_ctl/model-best`, which
`configs/config_lzh_depmorph_resplit.cfg` (and every later lzh layer) SOURCES its own tok2vec/
tagger/parser from. There is no frozen upstream to front-load here the way la's `--verbdist` port
needed (morphologizer/lemmatizer moved to the front) -- lzh's own deployed parser reads nothing
but NORM/PREFIX/SUFFIX/SHAPE, which need no annotation pass before tok2vec runs at all.

WHAT IT CHANGES, and it is one thing: the tok2vec's EMBED, from plain `spacy.MultiHashEmbed.v2` to
`sud.MultiHashEmbedVerbDistEmbed.v1` (sud_lemmavec_embed.py) -- the SAME `VerbDistExtractor`
mechanism la's own `sud.LemmaVecFeatsVerbDistEmbed.v1` uses, but without the lemma-vector/per-
feature-morphology machinery la's port piggybacked on, since lzh's deployed parser has neither.

WHY THIS IS WORTH TRYING FOR LZH SPECIFICALLY, checked directly rather than assumed to transfer
from la: lzh's own `conj:coord` (38.85% arc-factored accuracy, the SUD relation's second-worst
label) AND `parataxis` (42.91%, the single worst) both show the identical verb-crossing accuracy
drop la's `conj:coord` did, on BOTH the arc-factored decoder (conj:coord 43.00% -> 16.00%, parataxis
46.98% -> 27.00%, crossing vs not) AND the TRANSITION PARSER (conj:coord 57.97% -> 45.33%, parataxis
60.91% -> 46.41%) -- real, shared headroom on la's own model, confirmed here rather than guessed.

    make_lzh_verbdist_config.py --out configs/config_lzh_verbdist.cfg
    make_lzh_verbdist_config.py --out configs/config_lzh_verbdist_ctl.cfg --control
"""
from __future__ import annotations

import argparse

from thinc.api import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/config_lzh_resplit_ctl.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true",
                    help="capacity control: same Linear, same parameter count, POS never read")
    args = ap.parse_args()

    cfg = Config().from_disk(args.base, interpolate=False)
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    cfg["components"]["tok2vec"]["model"]["embed"] = {
        "@architectures": "sud.MultiHashEmbedVerbDistEmbed.v1",
        "width": embed["width"],
        "attrs": list(embed["attrs"]),
        "rows": list(embed["rows"]),
        "include_static_vectors": bool(embed["include_static_vectors"]),
        "verbdist_constant": bool(args.control),
    }

    cfg.to_disk(args.out)
    bit = "capacity control (zeros)" if args.control else "verb-distance block"
    print(f"wrote {args.out}  ({bit}, base={args.base})")


if __name__ == "__main__":
    main()
