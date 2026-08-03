#!/usr/bin/env python3
"""Derive a *morphology-informed* base config from a released arm's (seg) config.

The released recipe trains ``[tok2vec, tagger, parser]`` and only afterwards bolts a
``morphologizer`` on the end, so the parser is committed BEFORE any morphology exists and never
sees a single feature of it — in a case-based language, where the SUD relation on a nominal is
largely a function of its Case, that is the obvious thing to try changing.

Reordering the released pipeline does NOT achieve it. ``Tok2Vec.predict`` runs at the position of
the ``tok2vec`` PIPE and caches into ``doc.tensor``; a ``Tok2VecListener`` only ever reads that
cache. So a listener-based parser cannot see anything produced by a component that runs after
tok2vec, wherever that component sits. Two ways out:

  * give the parser its OWN encoder placed after the morphologizer (costs the multi-task benefit
    of the shared encoder — measured at 1.7–2.0 LAS on sa), or
  * move the morphologizer to the FRONT, frozen, and let the SHARED encoder read its predictions.

This script builds the second. The morphologizer carries its own ``HashEmbedCNN`` (the freeze
recipe in ``make_morph_config.py``), so it is self-contained and can run first; it is frozen AND
listed in ``annotating_components`` so its predictions are written onto the predicted docs during
training. tok2vec/tagger/parser are then trained FROM SCRATCH exactly as in the baseline, with
``MORPH`` appended to the embed attrs — a single-variable change against the released base.

``POS`` is deliberately NOT fed: the morphologizer also sets ``token.pos_``, and a shared encoder
that reads it hands the tagger a large part of its own target, collapsing its gradient.

CAVEAT (report it with any result): the morphologizer was trained on this same training split, so
its predictions are far more accurate on train than on dev/test. The parser therefore learns to
trust morphology more than it should at inference. Jackknifed (k-fold) morph predictions would be
the clean fix.

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_morphfirst_config.py configs/config_la_seg.cfg training_la_morph/model-best
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="the released arm's (seg) config")
    ap.add_argument("morph_model", help="model-best dir holding the trained morphologizer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--morph-rows", type=int, default=4096,
                    help="embed rows for MORPH (la has 5319 distinct FEATS strings)")
    args = ap.parse_args()

    cfg = Config().from_disk(args.base_config, interpolate=False)

    # 1) morphologizer FIRST, sourced + frozen, and annotating so it writes onto the predicted docs
    pipe = [p for p in cfg["nlp"]["pipeline"] if p != "morphologizer"]
    cfg["nlp"]["pipeline"] = ["morphologizer"] + pipe
    cfg["components"]["morphologizer"] = {"source": args.morph_model}
    cfg["training"]["frozen_components"] = ["morphologizer"]
    cfg["training"]["annotating_components"] = ["morphologizer"]

    # a sourced encoder must not be clobbered by an init_tok2vec bin
    cfg["paths"]["init_tok2vec"] = None
    cfg["initialize"]["init_tok2vec"] = None

    # 2) the single variable: the shared encoder additionally reads MORPH
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    if "MORPH" not in embed["attrs"]:
        embed["attrs"] = list(embed["attrs"]) + ["MORPH"]
        embed["rows"] = list(embed["rows"]) + [args.morph_rows]

    # 3) score the morphologizer at 0 — it is frozen, and its accuracy must not steer model-best
    cfg["training"]["score_weights"]["pos_acc"] = 0.0
    cfg["training"]["score_weights"]["morph_acc"] = 0.0
    cfg["training"]["score_weights"]["morph_per_feat"] = None

    out = args.out or args.base_config.replace("_seg.cfg", "_morphfirst.cfg")
    Config(cfg).to_disk(out)
    print(f"wrote {out}")
    print(f"  pipeline {cfg['nlp']['pipeline']}")
    print(f"  embed attrs {embed['attrs']}  rows {embed['rows']}")


if __name__ == "__main__":
    main()
