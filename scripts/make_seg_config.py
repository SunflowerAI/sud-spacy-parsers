#!/usr/bin/env python3
"""Derive a sentence-segmentation training config from a base config.

Two edits to ``configs/config_<lang>.cfg`` → ``configs/config_<lang>_seg.cfg``:
  1. Swap both corpus readers from ``spacy.Corpus.v1`` (gold_preproc) to
     ``sud.GoldTokCorpus.v1`` — whole multi-sentence docs with gold tokenisation, so the parser
     learns sentence boundaries without the tokeniser-mismatch cost (see scripts/gold_tok_corpus.py).
  2. Give ``sents_f`` a small score weight (default 0.05) so checkpoint selection rewards a model
     that actually segments (the base configs weight it 0.0).

Loads/saves with interpolation OFF so ``${paths.train}`` etc. survive (CLAUDE.md gotcha).

    make_seg_config.py configs/config_ar.cfg [--sents-weight 0.05]
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("--sents-weight", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config().from_disk(args.base_config, interpolate=False)
    for side in ("train", "dev"):
        c = cfg["corpora"][side]
        c.pop("gold_preproc", None)
        c["@readers"] = "sud.GoldTokCorpus.v1"
    cfg["training"]["score_weights"]["sents_f"] = args.sents_weight

    out = args.out or args.base_config.replace(".cfg", "_seg.cfg")
    cfg.to_disk(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
