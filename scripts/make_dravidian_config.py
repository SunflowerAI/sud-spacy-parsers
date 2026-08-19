#!/usr/bin/env python3
"""Derive the ta / te base configs from Latin's, with the two changes a 6 000-token treebank forces.

Everything else is left exactly as `configs/config_la.cfg` has it — encoder, embed, optimiser,
batcher, patience — so the arms stay comparable with the rest of the project. The two deviations
are both consequences of SIZE, and each of them is silent if it is not made.

1. ⚠ **`min_action_freq = 30` DELETES MOST OF THE LABEL INVENTORY HERE.** spaCy's default drops any
   parser action seen fewer than 30 times, which is sized for treebanks like Latin's 586 604
   tokens. Measured on these three training corpora it removes:

       ta TTB (6 329 tokens)          7 of 19 labels
       ta TTB+MWTT (8 409 tokens)    19 of 33 labels
       te MTG (5 082 tokens)         14 of 29 labels

   — `mod@relcl`, `parataxis`, `discourse`, `vocative`, `mod@cond`, `compound@redup` and the rest.
   The parser cannot then EMIT them, so their recall is exactly zero and no error message says so;
   the label simply never appears in the output. Set to 1: at this scale a label seen three times
   is not learnable either way, but excluding it guarantees the zero, whereas including it costs
   one dead action in the transition system.

2. **`tag_acc` drops out of checkpoint selection** (0.5 -> 0.0, redistributed to `dep_las`). Two
   reasons, and the first alone would be enough. Telugu's XPOS column is a VERBATIM COPY of UPOS —
   zero mismatches in 6 465 tokens — so `tag_acc` there is UPOS accuracy wearing a different name,
   and weighting it twice `dep_las` would select a parser's checkpoints on a tagger's score for a
   task with no content. On the combined Tamil arm 60.7 % of the MWTT half's tags are ones TTB
   never wrote (`normalise_ta_xpos.py` faithfully renders MWTT's sparser FEATS, so a noun with no
   gold Gender gets `-` in the gender slot), so selection would be riding the projection's
   artefacts. The real XPOS tagger is a LATER layer in this project's stack anyway — it is grafted
   by the freeze recipe reading UPOS+FEATS (`docs/xpos.md`), which is why the released component
   order puts `tagger` behind `morphologizer`.

`--seg` additionally swaps the readers to `sud.GoldTokCorpus.v1` and gives `sents_f` a weight,
exactly as `make_seg_config.py` does — the arms are trained through the seg recipe, since a base
recipe that never learned to START a sentence is standing hazard 4.

    make_dravidian_config.py --lang ta --out configs/config_ta.cfg
    make_dravidian_config.py --lang ta --out configs/config_ta_seg.cfg --seg
"""
from __future__ import annotations

import argparse

from thinc.api import Config

#: Redistributed from `tag_acc`. `sents_f` is set only by `--seg`, as in `make_seg_config.py`.
SCORE_WEIGHTS = {"tag_acc": 0.0, "dep_uas": 0.25, "dep_las": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=("ta", "te"))
    ap.add_argument("--base", default="configs/config_la.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--sents-weight", type=float, default=0.05)
    ap.add_argument("--min-action-freq", type=int, default=1)
    args = ap.parse_args()

    cfg = Config().from_disk(args.base, interpolate=False)
    cfg["nlp"]["lang"] = args.lang
    cfg["components"]["parser"]["min_action_freq"] = args.min_action_freq
    cfg["training"]["score_weights"].update(SCORE_WEIGHTS)

    if args.seg:
        for side in ("train", "dev"):
            corpus = cfg["corpora"][side]
            corpus.pop("gold_preproc", None)
            corpus["@readers"] = "sud.GoldTokCorpus.v1"
        cfg["training"]["score_weights"]["sents_f"] = args.sents_weight

    cfg.to_disk(args.out)
    print(f"wrote {args.out}  (lang={args.lang}, min_action_freq={args.min_action_freq}, "
          f"tag_acc=0.0{', GoldTokCorpus' if args.seg else ''})")


if __name__ == "__main__":
    main()
