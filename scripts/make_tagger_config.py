#!/usr/bin/env python
"""Derive a *tagger-only* retraining config on a frozen released arm.

Normalising PROIEL's and Perseus's XPOS onto ITTB's tagset (normalise_la_xpos.py) changes
exactly one thing a model can learn: the TAG target.  Nothing in the Latin pipeline reads TAG
as an input feature -- the base encoder embeds NORM/PREFIX/SUFFIX/SHAPE, and the SUD layer's
structural encoder reads POS (that is UPOS), DEP, MORPH and LEMMA -- so the parser, the
morphologiser, the lemmatiser and the three `sud_*` pipes are all answering the same question
they were trained on and have no business being retrained.

So this is the ordinary freeze recipe with the new component being a REPLACEMENT rather than an
addition: source `tok2vec` and `parser` from the released base arm and freeze them, and train a
single fresh `tagger` over its OWN encoder.  The frozen weights come out byte-identical, which
is what lets every published Latin figure -- LAS, UAS, comp:obl F, the orthography table, the
whole MISC layer -- stand without re-measurement.  The trained tagger is then grafted into the
released arm (graft_pipe.py), so the only thing that moves in the wheel is the tagger.

Two departures from make_morph_config.py, both because of what is being predicted:

  * The encoder is the base arm's size (width 96, depth 4), not the 64/3 the morphologiser and
    lemmatiser get.  Those predict a UPOS or an edit tree; this predicts one of ~2 340 Index
    Thomisticus composite codes, and the old tagger read a width-96 depth-4 encoder as a
    listener.  Under-sizing it here would confound the tagset change with a capacity cut.
  * `model-best` is selected on `tag_acc` ALONE.  In the arm this is derived from, the
    checkpoint is chosen on a weighted mean of tag/LAS/UAS/sents; with every other component
    frozen those terms are constant, but leaving them in the weighting would still let a
    constant dominate the mean and blur which epoch was best FOR THE TAGGER.  (Same hazard as
    the multi-feature SUD arms, where Latin's `Shared` was checkpointed at an epoch that suited
    its neighbours.)

Whatever the base config sets is inherited, which matters for Latin: the augmenter,
`max_epochs = -1` and `shuffle = true` must stay together -- at `max_epochs = 0` spaCy lists the corpus once and a corpus-level
augmenter then samples ONE style per document for the whole run.  Streaming in turn means
`init_nlp` sees only the first 100 examples, so the label set has to be handed in; unlike the
lemmatiser's edit trees, XPOS labels are properties of the TREES and of a folded spelling
(build_la_xpos_map.fold), so one clean pass over the corpus is exact and `--passes 1` suffices.  A base config that does NOT stream (English, `max_epochs = 0`)
initialises from the whole corpus and collects its own labels, so `--labels-dir` is omitted
there and no label file is written or read.

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_la_tagger_config.py configs/config_la_aug.cfg training_la_aug/model-best \
        --out configs/config_la_aug_xpos.cfg --labels-dir labels_la_aug_xpos
"""
import argparse
import os
import sys

from thinc.api import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_guard import guard_overwrite                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="the augmented base arm's config")
    ap.add_argument("source_model", help="model-best dir to source + freeze tok2vec/parser from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-dir", default=None,
                    help="only for a STREAMING base config, whose init sees 100 examples")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--embed-size", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    guard_overwrite(a.out, "tagger", "spacy.HashEmbedCNN.v2", a.force)
    cfg = Config().from_disk(a.base_config, interpolate=False)

    # 1) source + freeze everything that is not the tagger
    frozen = []
    for name in cfg["nlp"]["pipeline"]:
        if name == "tagger":
            continue
        cfg["components"][name] = {"source": a.source_model}
        frozen.append(name)
    cfg["training"]["frozen_components"] = frozen
    cfg["training"]["annotating_components"] = []

    # never let init_tok2vec clobber the sourced (frozen) encoder
    for section in ("paths", "initialize"):
        if section in cfg and "init_tok2vec" in cfg[section]:
            cfg[section]["init_tok2vec"] = None

    # 2) a self-contained tagger: Tagger head over its OWN HashEmbedCNN
    cfg["components"]["tagger"] = {
        "factory": "tagger",
        "neg_prefix": "!",
        "overwrite": False,
        "scorer": {"@scorers": "spacy.tagger_scorer.v1"},
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "normalize": False,
            "tok2vec": {
                "@architectures": "spacy.HashEmbedCNN.v2",
                "pretrained_vectors": None,
                "width": a.width,
                "depth": a.depth,
                "embed_size": a.embed_size,
                "window_size": 1,
                "maxout_pieces": 3,
                "subword_features": True,
            },
        },
    }

    # 3) the tagger is the only thing being trained, so it is the only thing selecting model-best
    cfg["training"]["score_weights"] = {
        "tag_acc": 1.0, "pos_acc": 0.0, "dep_uas": 0.0, "dep_las": 0.0, "sents_f": 0.0,
        "tag_micro_p": None, "tag_micro_r": None, "tag_micro_f": None,
        "dep_las_per_type": None, "sents_p": None, "sents_r": None,
    }

    # 4) labels come from a file (streaming init sees only 100 examples)
    if a.labels_dir:
        cfg["initialize"]["components"] = {
            "tagger": {"labels": {"@readers": "spacy.read_labels.v1",
                                  "path": f"{a.labels_dir}/tagger.json",
                                  "require": True}}
        }
    else:
        cfg["initialize"]["components"] = {}

    Config(cfg).to_disk(a.out)
    print(f"{a.out}: frozen={frozen} tagger=HashEmbedCNN "
          f"w{a.width}/d{a.depth}/e{a.embed_size}, "
          f"labels={a.labels_dir + '/tagger.json' if a.labels_dir else 'collected at init'}")


if __name__ == "__main__":
    main()
