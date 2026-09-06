#!/usr/bin/env python3
"""Derive a lemma-vector + decomposed-morphology parser config from the released Latin recipe.

WHAT IT CHANGES, and it is one thing: the tok2vec's EMBED. Everything else — the orthographic
augmenter, ``max_epochs = -1``, the reader, the encoder, the dev corpus — is left exactly as
``configs/config_la_aug.cfg`` has it, so the arm that comes out is comparable with
``training_la_aug`` without an asterisk.

    [components.tok2vec.model.embed]
    @architectures = "sud.LemmaVecFeatsEmbed.v1"
    attrs   = ["NORM","PREFIX","SUFFIX","SHAPE"]      # unchanged
    feats   = ["Case","Number","Gender", …]            # one hash table per category
    vectors = "scripts/la_lemmavec_96.npz"             # PPMI+SVD over the treebank's own lemmas

WHERE LEMMA AND FEATS COME FROM. The released Latin pipeline runs the parser FIRST, so at inference
it has neither. This moves the already-trained morphologiser and lemmatiser to the FRONT, frozen and
listed in ``annotating_components``, so the parser reads their PREDICTED output — during training as
well as at run time, which is what makes the arm shippable rather than an oracle. They can be moved
because the freeze recipe gave each of them its OWN ``HashEmbedCNN``: neither is a listener, and
neither reads the parser, so there is no circularity to unpick.

WHY DECOMPOSED, given this has been tried. ``configs/config_la_morphfirst.cfg`` already put a frozen
morphologiser at the front of this arm and fed its FEATS in as spaCy's ``MORPH`` column — ONE hash
of the whole bundle. Measured on the plain combined test: morphfirst LAS **0.7256**, capacity
control **0.7255**, released base 0.7226. The entire gain was the extra embedding rows; the
morphology contributed nothing measurable. That is the result this config exists to revisit, and
the reason it is worth revisiting is that a single bundle hash makes ``Case=Nom|Number=Sing`` and
``Case=Nom|Number=Plur`` unrelated symbols — so a parser cannot ask whether two tokens share a case,
which is the only question Latin agreement and government ever pose.

THE CONTROL (``--control``) is tight on purpose, because the morphfirst result above is exactly what
a loose one costs. Same architecture, same number of hash tables, same rows, same Maxout input
width: the ``feats`` tables are replaced by that many extra ``NORM`` tables (differently seeded, so
they add capacity and no information), and the lemma block is switched to ``constant = true``, which
keeps its Linear and hands every token the zero vector. Any gain over this control is the two
channels and not their parameters.

    make_la_lemvec_config.py --out configs/config_la_lemvec.cfg
    make_la_lemvec_config.py --out configs/config_la_lemvec_ctl.cfg --control
    make_la_lemvec_config.py --out configs/config_la_agree.cfg --agree
    make_la_lemvec_config.py --out configs/config_la_beam.cfg  --beam
    make_la_lemvec_config.py --out configs/config_la_agree_beam.cfg --agree --beam

--agree ADDS THE AGREEMENT-COMPATIBILITY BLOCK (`sud.LemmaVecFeatsAgreeEmbed.v1`, twelve dims
documented in scripts/sud_lemmavec_embed.py). Latin agreement is far more discriminative than the
Sanskrit signal the same idea was built on: gold agreeing arcs are 93.5 % Case/Number/Gender-
compatible against 13.6 % for a random nominal within three tokens that is not the head, and under
the PREDICTED morphology this arm actually reads, 60.8 points of that 79.9-point gap survive
(scripts/check_la_agreement_signal.py). `--agree --control-agree` is the exact capacity control,
though `config_la_lemvec.cfg` is very nearly one already: the block adds a single Linear(64, 12),
832 parameters, against the ~50 000 extra embedding rows that made the morphfirst control decisive.

--verbdist ADDS THE VERB-DISTANCE BLOCK (`sud.LemmaVecFeatsVerbDistEmbed.v1`, five dims documented
in scripts/sud_lemmavec_embed.py) -- the transition-parser counterpart of the arc-factored research
decoder's `--clausegap` term. Checking actual conj:coord errors under that decoder found the
dominant signal is whether a VERB/AUX sits between a candidate head and dependent (18.68% accuracy
crossing a verb vs 51.15% not), and a follow-up check found the TRANSITION parser has the SAME
qualitative weakness (66.13% vs 34.00%, an almost identical ~32-point relative gap) -- so this is a
shared weakness, not one the transition parser's own stack state already closes. Weaker than a
direct port, though: the transition parser has no clean insertion point for a PAIRWISE "does this
specific arc cross a verb" fact the way the arc-factored decoder's explicit (head, dependent, label)
scoring does, so this instead gives each token its OWN distance to the nearest VERB/AUX in each
direction, from which the parser's state-composition MLP could in principle reconstruct crossing
information across two stack/buffer positions -- real signal, but an indirect bet, not a certainty.
`--verbdist --control-verbdist` is the exact capacity control: identical Linear(64, 5), 325+64
parameters, POS never read.

--beam SWAPS THE GREEDY PARSER FOR `beam_parser`, for one measured reason. Latin discontinuity is
where this arm loses: 37.4 % of test sentences carry a crossing arc, those arcs are 5 % of tokens
but 16 % of all attachment errors, and the parser recovers only 28.4 % of them as non-projective at
all -- it emits 1 082 crossing arcs against 2 726 in the gold (scripts/analyse_la_nonproj_errors.py).
That under-production is exactly what a greedy decoder does with pseudo-projective labels: choosing
`mod||subj` over `mod` looks worse at that step and only pays after de-projectivisation, and a
greedy decoder cannot take a locally-costly bet.

⚠ THE SANSKRIT BEAM ARM CAME BACK NEGATIVE -- `train_sa_beam_s1.log` hit patience at 8 600 steps
with dev LAS 54.08 against the greedy arm's 57.14, and trailed at every matched step. Latin is a
better candidate on the premise (37.4 % non-projective sentences against Sanskrit's 23.97 %), but
"better candidate" is not "will work", and this run should be read as a test of that, not a
formality. It is also SLOW: eight states instead of one, so expect 15-30 h against the greedy
arm's ~2.5 h.
"""
from __future__ import annotations

import argparse

from thinc.api import Config

#: The morphological categories the parser gets a dimension for, and each table's rows.
#:
#: Chosen for a PARSER, which is not the same list ``build_feats_inventory.py`` proposes: that tool
#: ranks by information gain about XPOS GIVEN THE FORM, so it drops ``Mood`` (IG|form 0.004) and
#: ``VerbForm`` (0.017) as things the suffix already reveals. True for a tagger reading one token;
#: irrelevant here, because a parser needs the category to compare TWO tokens, and a table of 8 rows
#: costs 3 kB. So the list is everything that governs Latin attachment, with rows ~4x the value
#: count (``Case`` has 9 values, ``InflClass`` 14, ``PronType`` 10).
FEATS = ["Case", "Number", "Gender", "VerbForm", "Mood", "Tense",
         "Voice", "Person", "PronType", "Degree", "InflClass", "Aspect"]
FEAT_ROWS = [64, 16, 32, 32, 16, 32, 16, 16, 64, 32, 64, 16]

#: The arm the frozen morphologiser and lemmatiser are taken from — the RELEASED chain's own, so
#: their predictions are the ones a user of the wheel would actually get.
SOURCE_ARM = "training_la_aug_lemma/model-best"

VECTORS = "scripts/la_lemmavec_96.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/config_la_aug.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true",
                    help="capacity control: same tables and Maxout, no information in them")
    ap.add_argument("--agree", action="store_true",
                    help="add the agreement-compatibility block")
    ap.add_argument("--control-agree", action="store_true",
                    help="with --agree: keep the block's Linear, hand every token twelve zeros")
    ap.add_argument("--agree-near", type=int, default=20,
                    help="how far the any-left/any-right/n-compat dims reach. 20 is roughly a "
                         "Latin sentence, and it has to exceed the +-4 offsets or hyperbaton -- "
                         "the construction the block exists for -- falls outside the window")
    ap.add_argument("--verbdist", action="store_true",
                    help="add the verb-distance block (transition-parser counterpart of the "
                         "arc-factored decoder's --clausegap)")
    ap.add_argument("--control-verbdist", action="store_true",
                    help="with --verbdist: keep the block's Linear, hand every token five zeros")
    ap.add_argument("--beam", action="store_true",
                    help="train the parser as spacy's `beam_parser`")
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--beam-update-prob", type=float, default=0.5)
    ap.add_argument("--source", default=SOURCE_ARM)
    ap.add_argument("--vectors", default=VECTORS)
    ap.add_argument("--labels-dir", default="labels_la_lemvec",
                    help="collected fresh by scripts/init_aug_labels.py. Do NOT point this at "
                         "labels_la_aug/: that directory holds 1 952 tagger labels, the tagset "
                         "from BEFORE normalise_la_xpos.py, against the corpus's current 2 342, "
                         "and a missing tagger label is not a silent loss but a KeyError on the "
                         "first batch that carries one.")
    args = ap.parse_args()

    cfg = Config().from_disk(args.base, interpolate=False)

    # 1. the two frozen pipes, in FRONT of the parser and annotating during training
    for name in ("morphologizer", "lemmatizer"):
        cfg["components"][name] = {"source": args.source}
    cfg["nlp"]["pipeline"] = ["morphologizer", "lemmatizer", "tok2vec", "tagger", "parser"]
    cfg["training"]["frozen_components"] = ["morphologizer", "lemmatizer"]
    cfg["training"]["annotating_components"] = ["morphologizer", "lemmatizer"]
    # A sourced component brings its own labels; leaving an `initialize` block pointed at a labels
    # file it never reads is how a stale path survives a rename unnoticed.
    for name in ("morphologizer", "lemmatizer"):
        cfg["initialize"].get("components", {}).pop(name, None)
    for name in ("tagger", "parser"):
        block = cfg["initialize"].setdefault("components", {}).setdefault(name, {})
        block["labels"] = {"@readers": "spacy.read_labels.v1",
                           "path": f"{args.labels_dir.rstrip('/')}/{name}.json", "require": True}

    # 2. the embed
    if args.agree and args.verbdist:
        raise SystemExit("--agree and --verbdist together need a THIRD registered architecture "
                          "combining both blocks, which does not exist yet -- one at a time")
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    attrs = list(embed["attrs"])
    rows = list(embed["rows"])
    new = {
        "@architectures": ("sud.LemmaVecFeatsAgreeEmbed.v1" if args.agree
                           else "sud.LemmaVecFeatsVerbDistEmbed.v1" if args.verbdist
                           else "sud.LemmaVecFeatsEmbed.v1"),
        "width": embed["width"],
        "include_static_vectors": False,
        "vectors": args.vectors,
    }
    if args.control:
        new["attrs"] = attrs + ["NORM"] * len(FEATS)
        new["rows"] = rows + FEAT_ROWS
        new["feats"] = []
        new["feat_rows"] = []
        new["constant"] = True
    else:
        new["attrs"] = attrs
        new["rows"] = rows
        new["feats"] = FEATS
        new["feat_rows"] = FEAT_ROWS
        new["constant"] = False
    if args.agree:
        new["agree_near"] = args.agree_near
        new["agree_constant"] = bool(args.control_agree)
    elif args.control_agree:
        raise SystemExit("--control-agree only means anything with --agree")
    if args.verbdist:
        new["verbdist_constant"] = bool(args.control_verbdist)
    elif args.control_verbdist:
        raise SystemExit("--control-verbdist only means anything with --verbdist")
    cfg["components"]["tok2vec"]["model"]["embed"] = new

    # 3. the decoder. `beam_parser` takes the same model and the same labels; it changes how the
    #    action sequence is SEARCHED, and -- because beam_update_prob > 0 -- how the scores are
    #    trained to compose into a sequence score. Switching the factory without training it that
    #    way is measured useless: the sa note records rank-0 coming out 13 LAS below greedy.
    if args.beam:
        p = cfg["components"]["parser"]
        p["factory"] = "beam_parser"
        p["beam_width"] = args.beam_width
        p["beam_density"] = 0.0001
        p["beam_update_prob"] = args.beam_update_prob

    cfg.to_disk(args.out)
    bits = ["capacity control" if args.control else "lemma vectors + per-feature morphology"]
    if args.agree:
        bits.append("agreement block (zeros)" if args.control_agree
                    else f"agreement block (near={args.agree_near})")
    if args.verbdist:
        bits.append("verb-distance block (zeros)" if args.control_verbdist
                    else "verb-distance block")
    if args.beam:
        bits.append(f"beam_parser width={args.beam_width} update_prob={args.beam_update_prob}")
    print(f"wrote {args.out}  ({'; '.join(bits)}; frozen+annotating from {args.source})")


if __name__ == "__main__":
    main()
