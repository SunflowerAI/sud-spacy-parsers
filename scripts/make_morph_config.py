#!/usr/bin/env python3
"""Derive a *morphologiser-only* training config from a released arm.

We want UPOS (``token.pos_``) + morph (``token.morph``) in the output of every released model,
WITHOUT touching the parser/tagger/segmentation. So the recipe is: source the existing
``tok2vec``/``tagger``/``parser`` from the released model, FREEZE them, and train a single new
``morphologizer`` that has its OWN standalone ``tok2vec`` (HashEmbedCNN). Because the morphologiser
does not listen to the shared encoder, it is immune to treebanks (e.g. Indonesian) whose XPOS is
orthogonal to UPOS, and the frozen components stay byte-identical (no parse/seg re-verification).

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_morph_config.py configs/config_id_seg.cfg training_id_seg/model-best
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="the released arm's (seg) config")
    ap.add_argument("source_model", help="path to the released model-best dir to source+freeze")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--embed-size", type=int, default=2000)
    args = ap.parse_args()

    cfg = Config().from_disk(args.base_config, interpolate=False)

    # 1) add the morphologiser to the end of the pipeline
    pipe = list(cfg["nlp"]["pipeline"])
    if "morphologizer" not in pipe:
        pipe.append("morphologizer")
    cfg["nlp"]["pipeline"] = pipe

    # 2) source + freeze the three existing components (keeps them byte-identical)
    frozen = []
    for name in ("tok2vec", "tagger", "parser"):
        if name in cfg["components"]:
            cfg["components"][name] = {"source": args.source_model}
            frozen.append(name)

    # never let init_tok2vec clobber the sourced (frozen) encoder — e.g. yue's Mandarin-init bin,
    # which would overwrite the trained tok2vec with the raw init and break the parser's input.
    if "paths" in cfg and "init_tok2vec" in cfg["paths"]:
        cfg["paths"]["init_tok2vec"] = None
    if "initialize" in cfg and "init_tok2vec" in cfg["initialize"]:
        cfg["initialize"]["init_tok2vec"] = None

    # 3) a self-contained morphologiser: Tagger head over its OWN HashEmbedCNN tok2vec. Keep only the
    # args common to the standard and language-specific factories (ja.morphologizer rejects
    # label_smoothing/overwrite/extend).
    cfg["components"]["morphologizer"] = {
        "factory": "morphologizer",
        "scorer": {"@scorers": "spacy.morphologizer_scorer.v1"},
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

    # 4) freeze; only the morphologiser updates
    cfg["training"]["frozen_components"] = frozen
    cfg["training"]["annotating_components"] = []

    # 5) checkpoint selection tracks UPOS (frozen scores are constant anyway)
    sw = cfg["training"].setdefault("score_weights", {})
    for k in list(sw):
        # dict-valued scores (per_type / per_feat) must be null, not 0.0 (E915)
        sw[k] = None if k.endswith("_per_type") or k.endswith("_per_feat") else 0.0
    sw["pos_acc"] = 0.8
    sw["morph_acc"] = 0.2

    out = args.out or args.base_config.replace(".cfg", "_morph.cfg")
    cfg.to_disk(out)
    print(f"wrote {out}  (frozen: {frozen})")


if __name__ == "__main__":
    main()
