#!/usr/bin/env python3
"""Derive a *lemmatiser-only* training config from a released (morphologiser-equipped) arm.

Every released model should emit ``token.lemma_``. The recipe mirrors ``make_morph_config.py``:
source the arm's existing ``tok2vec``/``tagger``/``parser``/``morphologizer``, FREEZE all of them,
and train ONLY a new ``trainable_lemmatizer`` (spaCy's EditTreeLemmatizer) that carries its OWN
standalone ``HashEmbedCNN`` encoder (width 64 / depth 3 / embed 2000). A dedicated encoder keeps
the frozen components byte-identical (no parse/seg/morph re-verification) and makes the lemmatiser
self-contained. The edit-tree lemmatiser is language-agnostic — it learns FORM→LEMMA string edits
from the treebank's LEMMA column, so it works for the custom-script arms (lzh/sa) too.

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_lemma_config.py configs/config_id_morph.cfg training_id_morph/model-best
"""
import argparse
import os
import sys

from thinc.api import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_guard import guard_overwrite                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="the released arm's *_morph config")
    ap.add_argument("source_model", help="path to training_<lang>_morph/model-best to source+freeze")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--embed-size", type=int, default=2000)
    ap.add_argument("--force", action="store_true",
                    help="overwrite a hand-maintained --out (see scripts/config_guard.py)")
    args = ap.parse_args()

    out = (args.out or args.base_config.replace("_morph.cfg", "_lemma.cfg")
                                       .replace(".cfg", "_lemma.cfg"))
    guard_overwrite(out, "lemmatizer", "spacy.HashEmbedCNN.v2", args.force)

    cfg = Config().from_disk(args.base_config, interpolate=False)

    # 1) add the lemmatiser to the end of the pipeline
    pipe = list(cfg["nlp"]["pipeline"])
    if "lemmatizer" not in pipe:
        pipe.append("lemmatizer")
    cfg["nlp"]["pipeline"] = pipe

    # 2) source + freeze every existing trainable component (keeps them byte-identical)
    frozen = []
    for name in ("tok2vec", "tagger", "parser", "morphologizer"):
        if name in cfg["components"]:
            cfg["components"][name] = {"source": args.source_model}
            frozen.append(name)

    # never let init_tok2vec clobber the sourced (frozen) encoder (e.g. yue's Mandarin-init bin)
    if "paths" in cfg and "init_tok2vec" in cfg["paths"]:
        cfg["paths"]["init_tok2vec"] = None
    if "initialize" in cfg and "init_tok2vec" in cfg["initialize"]:
        cfg["initialize"]["init_tok2vec"] = None

    # 3) a self-contained EditTreeLemmatizer: Tagger head over its OWN HashEmbedCNN tok2vec, with an
    # orth backoff (fall back to the surface form when no edit tree applies).
    cfg["components"]["lemmatizer"] = {
        "factory": "trainable_lemmatizer",
        "backoff": "orth",
        "min_tree_freq": 3,
        "overwrite": False,
        "top_k": 1,
        "scorer": {"@scorers": "spacy.lemmatizer_scorer.v1"},
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "normalize": False,
            "tok2vec": {
                "@architectures": "spacy.HashEmbedCNN.v2",
                "pretrained_vectors": None,
                "width": args.width,
                "depth": args.depth,
                "embed_size": args.embed_size,
                "window_size": 1,
                "maxout_pieces": 3,
                "subword_features": True,
            },
        },
    }

    # 4) freeze everything but the lemmatiser
    cfg["training"]["frozen_components"] = frozen
    cfg["training"]["annotating_components"] = []

    # 5) checkpoint selection tracks lemma accuracy (the frozen scores are constant)
    sw = cfg["training"].setdefault("score_weights", {})
    for k in list(sw):
        sw[k] = None if k.endswith("_per_type") or k.endswith("_per_feat") else 0.0
    sw["lemma_acc"] = 1.0

    out = args.out or args.base_config.replace("_morph.cfg", "_lemma.cfg").replace(".cfg", "_lemma.cfg")
    cfg.to_disk(out)
    print(f"wrote {out}  (frozen: {frozen})")


if __name__ == "__main__":
    main()
