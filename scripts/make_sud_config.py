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
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from thinc.api import Config  # noqa: E402

from sud_misc import SUD_FEATS_KEYS  # noqa: E402

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
    ap.add_argument("--mask", nargs="+", default=None,
                    help="candidate mask PER FEATURE (see sud_tagger.MASKS); \"\" for none. "
                         "Defaults to `coordination` for Shared and none for the rest.")
    ap.add_argument("--encoder", nargs="+", default=None,
                    choices=["default", "structural", "tree"],
                    help="encoder PER FEATURE, in --feats order (one value broadcasts to all). "
                         "The three SUD features want three different ones -- see below -- so "
                         "an arm training more than one cannot use a single global flag.")
    args = ap.parse_args()

    # THE ENCODER IS A PROPERTY OF THE FEATURE, not of the arm. `Subject` is local (a raising
    # complement sits next to its control verb), `Reported` is not (the speech verb can be far
    # off, quotes sit at the clause edges), and `Shared` is neither -- it is a fact about a
    # COORDINATION, so what it needs is the head and the head's other dependents. en trains all
    # three in one arm, so a global --structural/--tree cannot express it.
    if args.encoder:
        encoders = args.encoder * len(args.feats) if len(args.encoder) == 1 else args.encoder
        if len(encoders) != len(args.feats):
            ap.error(f"--encoder takes 1 or {len(args.feats)} values, got {len(args.encoder)}")
    else:   # legacy: the flags apply to every feature in the call
        encoders = [("tree" if args.tree else "structural" if args.structural else "default")] * \
            len(args.feats)
    # The tree layer reads the parse, so it needs the same annotation the structural embed does.
    any_structural = any(e in ("structural", "tree") for e in encoders)

    # A mask restricts a pipe to the tokens where its feature's question is even asked. Only
    # `Shared` has one, and it reads the parse, so it forces the annotating components on too.
    if args.mask:
        masks = args.mask * len(args.feats) if len(args.mask) == 1 else args.mask
        if len(masks) != len(args.feats):
            ap.error(f"--mask takes 1 or {len(args.feats)} values, got {len(args.mask)}")
    else:
        masks = ["coordination" if f == "Shared" else "" for f in args.feats]
    any_structural = any_structural or any(masks)

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

    # 3) each new pipe: a Tagger head over its OWN encoder, writing to the SUD slot.
    #
    # Three encoder shapes, chosen per feature (--encoder). The DEFAULT is HashEmbedCNN
    # (NORM/PREFIX/SUFFIX/SHAPE, window 1, depth 3 -- a +-3 receptive field), which is right for
    # `Subject`: a raising complement sits next to its control verb, and this reaches F 0.72-0.92.
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
    #
    # `tree` goes further still, and is what `Shared` needs. `Shared` says whether a dependent of
    # a conjunct is shared with the other conjuncts, so its evidence is not linear at ANY width:
    # what matters is which token is my HEAD (is it a conjunct?) and what else hangs off it. The
    # HeadDepsTagger concatenates [own | head | mean of immediate dependents], which is exactly
    # that neighbourhood, read directly rather than approximated by proximity.
    structural_attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "LEMMA", "POS", "DEP", "MORPH",
                        "IS_QUOTE"]
    structural_rows = [args.embed_size, 1000, 1000, 1000, args.embed_size, 100, 500, 500, 20]

    def build_encoder(kind):
        structural = kind in ("structural", "tree")
        window = args.window_size if args.window_size is not None else (3 if structural else 1)
        depth = max(args.depth, 4) if structural else args.depth
        if structural:
            return {
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
        return {
            "@architectures": "spacy.HashEmbedCNN.v2",
            "pretrained_vectors": None,
            "width": args.width,
            "depth": depth,
            "embed_size": args.embed_size,
            "window_size": window,
            "maxout_pieces": 3,
            "subword_features": True,
        }

    # A pipe for a key the treebanks keep in FEATS (only `Shared`) TAKES THAT FEATURE OVER: it
    # deletes the morphologiser's own guess from `token.morph` so the arm has one answer for it
    # rather than two contradictory ones. See `clear_morph` in sud_tagger.py.
    for feat, name, kind, mask in zip(args.feats, names, encoders, masks):
        tree = kind == "tree"
        cfg["components"][name] = {
            "factory": "sud_tagger",
            "feat": feat,
            "overwrite": True,
            "clear_morph": feat in SUD_FEATS_KEYS,
            "mask": mask,
            "model": {
                "@architectures": "sud.HeadDepsTagger.v1" if tree else "spacy.Tagger.v2",
                **({"pool": args.pool, "detach": args.detach} if tree else {}),
                "nO": None,
                "normalize": False,
                "tok2vec": build_encoder(kind),
            },
        }

    # 4) freeze everything but the new pipes.
    #
    # As soon as ANY pipe uses the structural or tree encoder, the frozen components must also
    # ANNOTATE: the corpus readers build the predicted doc from gold words and nothing else, so
    # DEP/POS/MORPH/LEMMA would be absent during training and present at inference -- the model
    # would learn to ignore inputs that then appear from nowhere. Listing them here runs each
    # frozen component over the predicted docs, so the pipe trains on exactly the predictions it
    # will meet at runtime. (Same reasoning as the `Compound=Yes` input feature, and as
    # config_sa_lemma's annotating_components.) The tree encoder needs it for the parse itself,
    # not merely for the DEP embedding: with no heads annotated, every token would be its own
    # head and the head/dependent slices would carry nothing.
    #
    # ⚠ `tok2vec` MUST BE IN THIS LIST, and leaving it out fails SILENTLY. The tagger, parser,
    # morphologizer and lemmatizer here are listeners on the shared encoder: they read whatever
    # `tok2vec` last cached for the batch, so running them without it feeds them a stale buffer.
    # Nothing raises. What comes out is a degenerate parse -- on a 298-token dev doc, three
    # distinct deprels (`ROOT`, `comp:obj`, `goeswith`) and not one `conj` -- against twelve and
    # four once tok2vec runs. So a pipe that reads DEP/POS/MORPH was reading noise, and one that
    # reads the tree (or a mask derived from it) saw no structure at all: the `Shared` pipe's
    # coordination mask came out EMPTY on every training doc, its loss was a flat 0.00, and it
    # learnt nothing. Found here; it applies to every `--structural` arm trained before this fix.
    cfg["training"]["frozen_components"] = frozen
    cfg["training"]["annotating_components"] = (
        [n for n in ("tok2vec", "tagger", "parser", "morphologizer", "lemmatizer") if n in frozen]
        if any_structural else []
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
