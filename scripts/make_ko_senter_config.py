#!/usr/bin/env python3
"""A standalone SENTENCISER for ko, trained by the freeze recipe and grafted in front of the parser.

WHY, AND WHAT IT REPLACES. Korean sentence boundaries can be learned two ways here, and they are not
equally priced:

  * THE SEG RECIPE. Change the reader so each example is a multi-sentence document and let the
    PARSER learn to emit a boundary. It works — raw SENT F 81.37 → 90.14 — but it is a trade, not a
    free win: the same channel arm trained this way reads 72.88 LAS under gold sentences against
    74.45 for the single-sentence recipe. **1.57 LAS is the price of teaching the parser to
    segment**, and the lzh write-up records the same shape of trade.
  * A SEPARATE SENTENCISER. Keep the parser that scored 74.45 and put a component in front of it
    that does nothing else.

⚠ THE SECOND ONLY WORKS BECAUSE spaCy's PARSER HONOURS PRESET BOUNDARIES AS A HARD CONSTRAINT, and
that was verified rather than assumed: fed a two-sentence string with no boundaries the plain arm
returns ONE sentence with one root, and with `is_sent_start` preset it returns TWO with two roots,
same weights. `ArcEager` reads `sent_start` off the doc, so the `senter` in front is not advisory.

⚠ IT READS THE ANALYSER CHANNEL, which is the point. A Korean sentence ends on a final ending —
`EF` — and that is precisely what `sud.KoAnalyserEmbed.v1`'s last-morpheme columns carry. The
sentenciser is a tagger over its OWN encoder (the freeze recipe: width 64, depth 3, embed 2000)
whose embed is the channel, not the tiny width-12 HashEmbedCNN spaCy defaults to.

⚠ THE READER MUST BE `sud.GoldTokCorpus.v1`. Under `gold_preproc` every example is already exactly
one sentence, so a sentenciser trained on it sees no boundary to find and scores 100 on a task it
never learned — the same mirage that hid the parser's inability to segment (CLAUDE.md hazard 4).

    .venv/bin/python scripts/make_ko_senter_config.py training_ko_an_xposwarm/model-best \
        --out configs/config_ko_an_senter.cfg
"""
from __future__ import annotations

import argparse

from thinc.api import Config

#: The senter goes SECOND: after `tok2vec` (whose listeners need it built first) and before the
#: parser, which is the whole point.
PIPELINE = ["tok2vec", "senter", "parser", "morphologizer", "lemmatizer", "tagger"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="the grafted arm to source every other component from")
    ap.add_argument("--out", default="configs/config_ko_an_senter.cfg")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--embed-size", type=int, default=2000)
    args = ap.parse_args()

    cfg = Config().from_disk("configs/config_ko_eojeol_seg.cfg", interpolate=False)
    cfg["nlp"]["pipeline"] = list(PIPELINE)
    cfg["training"]["frozen_components"] = [c for c in PIPELINE if c != "senter"]
    # Nothing needs annotating: the senter's own encoder reads the raw doc, and every other
    # component is frozen and not run.
    cfg["training"]["annotating_components"] = []
    cfg["training"]["score_weights"] = {"sents_f": 1.0, "sents_p": 0.0, "sents_r": 0.0,
                                        "tag_acc": 0.0, "dep_uas": 0.0, "dep_las": 0.0}

    cfg["components"] = {}
    for comp in PIPELINE:
        if comp == "senter":
            continue
        cfg["components"][comp] = {"source": args.base}
    cfg["components"]["senter"] = {
        "factory": "senter",
        "overwrite": False,
        "scorer": {"@scorers": "spacy.senter_scorer.v1"},
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "normalize": False,
            "tok2vec": {
                "@architectures": "spacy.Tok2Vec.v2",
                "embed": {
                    "@architectures": "sud.KoAnalyserEmbed.v1",
                    "width": args.width,
                    "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                    "rows": [2000, 1000, 2500, 1000],
                    "morph_rows": [2000, 2000],
                    "feats": ["First", "Last", "Bag"],
                    "constant": False,
                    "include_static_vectors": False,
                },
                "encode": {
                    "@architectures": "spacy.MaxoutWindowEncoder.v2",
                    "width": args.width,
                    "depth": args.depth,
                    "window_size": 1,
                    "maxout_pieces": 3,
                },
            },
        },
    }
    cfg.to_disk(args.out)
    print(f"wrote {args.out}")
    print(f"  pipeline {PIPELINE}")
    print(f"  training only `senter`, sourcing the rest from {args.base}")


if __name__ == "__main__":
    main()
