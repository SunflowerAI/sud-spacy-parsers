#!/usr/bin/env python3
"""Derive a *SUD-MISC-only* training config from a released (lemmatiser-equipped) arm.

One storey above ``make_lemma_config.py``, and the same freeze recipe: source the arm's existing
``tok2vec``/``tagger``/``parser``/``morphologizer``/``lemmatizer``, FREEZE all of them, and train
ONLY the new ``sud_tagger`` pipe(s), each carrying its OWN standalone ``HashEmbedCNN`` (width 64 /
depth 3 / embed 2000). The dedicated encoder keeps the frozen components byte-identical -- so
LAS/UAS/TAG/UPOS/morph/lemma need no re-verification -- and makes the new layer self-contained.

The pipes read gold from the FEATS column, where ``hoist_sud_gold.py`` has put it under the ``Sud``
prefix (``spacy convert`` discards MISC); at inference they write to ``Token._.sud_misc``.

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_sud_config.py configs/config_la_lemma.cfg training_la_lemma/model-best --feats Subject
"""
import argparse

from thinc.api import Config

FROZEN = ("tok2vec", "tagger", "parser", "morphologizer", "lemmatizer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="the released arm's *_lemma config")
    ap.add_argument("source_model", help="path to training_<lang>_lemma/model-best to source+freeze")
    ap.add_argument("--feats", nargs="+", default=["Subject"],
                    help="SUD MISC features to train a pipe for (one pipe each)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--embed-size", type=int, default=2000)
    ap.add_argument("--structural", action="store_true",
                    help="read DEP/POS/MORPH/LEMMA/IS_QUOTE and widen the receptive field "
                         "(needed for Reported; see the note below)")
    ap.add_argument("--tree", action="store_true",
                    help="concatenate each token's HEAD and SUBTREE-mean vectors; implies "
                         "--structural (the rule's evidence is tree-shaped, not linear)")
    ap.add_argument("--pool", default="deps",
                    choices=["none", "deps", "closed", "deps2", "closed2"],
                    help="--tree only: what the third slice pools (none = diagnostic)")
    ap.add_argument("--detach", action="store_true",
                    help="--tree only: read head/pool slices without backpropagating through them")
    ap.add_argument("--window-size", type=int, default=None,
                    help="convolution window; --structural defaults it to 3 (receptive field +-12)")
    args = ap.parse_args()
    if args.tree:
        args.structural = True   # the tree layer needs the parse annotated during training

    cfg = Config().from_disk(args.base_config, interpolate=False)

    # 1) add one pipe per feature, named sud_<feat>
    pipe = list(cfg["nlp"]["pipeline"])
    names = []
    for feat in args.feats:
        name = f"sud_{feat.lower()}"
        names.append(name)
        if name not in pipe:
            pipe.append(name)
    cfg["nlp"]["pipeline"] = pipe

    # 2) source + freeze every existing trainable component (keeps them byte-identical)
    frozen = []
    for name in FROZEN:
        if name in cfg["components"]:
            cfg["components"][name] = {"source": args.source_model}
            frozen.append(name)

    # never let init_tok2vec clobber the sourced (frozen) encoder (e.g. yue's Mandarin-init bin)
    if "paths" in cfg and "init_tok2vec" in cfg["paths"]:
        cfg["paths"]["init_tok2vec"] = None
    if "initialize" in cfg and "init_tok2vec" in cfg["initialize"]:
        cfg["initialize"]["init_tok2vec"] = None

    # 3) each new pipe: a Tagger head over its OWN encoder, writing to the MISC slot.
    #
    # Two encoder shapes. The DEFAULT is HashEmbedCNN (NORM/PREFIX/SUFFIX/SHAPE, window 1, depth 3
    # -- a +-3 receptive field), which is right for `Subject`: a raising complement sits next to
    # its control verb, and this reaches F 0.72-0.92.
    #
    # `--structural` is for `Reported`, whose evidence is nowhere near local -- the governing
    # speech verb can be far from the clause head, quotation marks sit at the CLAUSE EDGES, and
    # Latin's diagnostic is the complement's own VerbForm/Mood plus the absence of a subordinator.
    # It swaps in the explicit MultiHashEmbed + MaxoutWindowEncoder pair so the embed can read
    # what actually carries that evidence:
    #   DEP       the parser's relation -- comp:*/parataxis vs anything else
    #   POS/MORPH VerbForm and Mood, i.e. the whole Latin finite-vs-infinitive diagnostic
    #   LEMMA     collapses inflection, so a speech verb is one symbol across its paradigm
    #             (decisive for la/ar/sa, where `dico`/`قَال`/`vac` inflect heavily)
    #   IS_QUOTE  quotation marks as a first-class feature rather than a shape accident
    # plus window 3 / depth 4, a +-12 receptive field that actually reaches the clause edges.
    structural_attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "LEMMA", "POS", "DEP", "MORPH",
                        "IS_QUOTE"]
    structural_rows = [args.embed_size, 1000, 1000, 1000, args.embed_size, 100, 500, 500, 20]
    window = args.window_size if args.window_size is not None else (3 if args.structural else 1)
    depth = args.depth if not args.structural else max(args.depth, 4)

    if args.structural:
        encoder = {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": args.width,
                "attrs": structural_attrs,
                "rows": structural_rows,
                "include_static_vectors": False,
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": args.width,
                "depth": depth,
                "window_size": window,
                "maxout_pieces": 3,
            },
        }
    else:
        encoder = {
            "@architectures": "spacy.HashEmbedCNN.v2",
            "pretrained_vectors": None,
            "width": args.width,
            "depth": depth,
            "embed_size": args.embed_size,
            "window_size": window,
            "maxout_pieces": 3,
            "subword_features": True,
        }

    # `--tree` goes further than `--structural`: the head/subtree concatenation gives the model the
    # SAME neighbourhood the rule reads. A convolution, however wide, mixes over LINEAR neighbours,
    # but the evidence is tree-shaped -- is my HEAD a speech verb, does my SUBTREE contain a quote
    # or a discourse marker. This reads those directly, at any distance.
    arch = "sud.HeadDepsTagger.v1" if args.tree else "spacy.Tagger.v2"

    for feat, name in zip(args.feats, names):
        cfg["components"][name] = {
            "factory": "sud_tagger",
            "feat": feat,
            "overwrite": True,
            "model": {
                "@architectures": arch,
                **({"pool": args.pool, "detach": args.detach} if args.tree else {}),
                "nO": None,
                "normalize": False,
                "tok2vec": dict(encoder),
            },
        }

    # 4) freeze everything but the new pipes.
    #
    # With --structural the frozen components must also ANNOTATE: the corpus readers build the
    # predicted doc from gold words and nothing else, so DEP/POS/MORPH/LEMMA would be absent during
    # training and present at inference -- the model would learn to ignore inputs that then appear
    # from nowhere. Listing them here runs each frozen component over the predicted docs, so the
    # pipe trains on exactly the predictions it will meet at runtime. (Same reasoning as the
    # `Compound=Yes` input feature, and as config_sa_lemma's annotating_components.)
    cfg["training"]["frozen_components"] = frozen
    cfg["training"]["annotating_components"] = (
        [n for n in ("tagger", "parser", "morphologizer", "lemmatizer") if n in frozen]
        if args.structural else []
    )

    # 5) checkpoint selection tracks the new F scores only (the frozen scores are constant).
    # Dict-valued scores (_per_type / _per_feat) must be null, not 0.0 -- spaCy E915.
    sw = cfg["training"].setdefault("score_weights", {})
    for k in list(sw):
        sw[k] = None if k.endswith("_per_type") or k.endswith("_per_feat") else 0.0
    for feat in args.feats:
        key = feat.lower()
        sw[f"sud_{key}_f"] = round(1.0 / len(args.feats), 4)
        sw[f"sud_{key}_p"] = 0.0
        sw[f"sud_{key}_r"] = 0.0

    out = args.out or args.base_config.replace("_lemma.cfg", "_sud.cfg")
    cfg.to_disk(out)
    print(f"wrote {out}  (pipes: {names}; frozen: {frozen})")


if __name__ == "__main__":
    main()
