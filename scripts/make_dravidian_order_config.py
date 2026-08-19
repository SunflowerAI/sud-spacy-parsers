#!/usr/bin/env python3
"""Graft the ta/te word-order augmenter onto an existing arm's config.

It grafts onto EITHER config, and both matter:

    config_ta_seg.cfg          -> the order arm, measured against the plain base
    config_ta_<arm>_lemvec.cfg -> order AND lemma+morphology together, the full recipe

⚠ THE AUGMENTER NEEDS THREE EDITS, NOT ONE, and the two companions are the silent ones.

1. ``max_epochs = -1``. At the project's usual ``0``, spaCy's ``create_train_batches`` does
   ``examples = list(corpus(nlp))`` ONCE and reshuffles that same list every epoch, so a
   corpus-level augmenter samples a single linearisation per document for the WHOLE RUN. The run
   looks completely normal and trains on one fixed permutation. This is `docs/latin.md`'s hazard 9.
2. ``shuffle = true`` on the reader. ``-1`` streams the corpus, which also turns off the training
   loop's own shuffling — so the reader has to do it. Harmless: under ``sud.GoldTokCorpus.v1`` a
   document IS an example, so it is the same shuffle by another name.
3. **Collected labels, with ``require = true``.** Under ``-1``, ``init_nlp`` initialises from
   ``islice(train_corpus(nlp), 100)`` and takes whatever label set that slice happens to contain.
   For Latin that truncated the tagger to 639 of 1 952 labels. Here the exposure is different and
   worse in kind: **the parser's own labels are a property of the ORDER**, because a non-projective
   gold tree is pseudo-projectivised and the lifted arc picks up a ``||`` suffix naming what it was
   lifted over. So the label set genuinely MOVES between augmented passes, and a missing parser
   label is not a silent loss but a `KeyError` on the first batch that carries one. Collect with
   ``scripts/init_aug_labels.py --passes N`` and read the coverage it prints.

    make_dravidian_order_config.py configs/config_ta_seg.cfg --lang ta \\
        --out configs/config_ta_ttb_order.cfg --labels-dir labels_ta_ttb_order
"""
from __future__ import annotations

import argparse

from thinc.api import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("--lang", required=True, choices=("ta", "te"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--p-sentence", type=float, default=0.5)
    ap.add_argument("--p-hyperbaton", type=float, default=-1.0,
                    help="-1 takes the language's calibrated default: 0.08 for ta (18.0 %% of its "
                         "training sentences carry a crossing arc), 0.0 for te (0.1 %%)")
    ap.add_argument("--clause-only", type=int, default=1,
                    help="1 scrambles only under VERB/AUX heads; 0 under every head. See the "
                         "dravidian_order docstring -- the corpora do not settle this, so it is a "
                         "knob to be measured rather than a decision to be made here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config().from_disk(args.base_config, interpolate=False)

    train = cfg["corpora"]["train"]
    if train.get("@readers") != "sud.GoldTokCorpus.v1":
        raise SystemExit(f"{args.base_config}: the augmenter needs the GoldTokCorpus reader, "
                         f"found {train.get('@readers')!r}. Graft onto a *_seg config.")
    train["shuffle"] = True
    train["augmenter"] = {"@augmenters": "sud.dravidian_order_variants.v1",
                          "lang": args.lang,
                          "p_sentence": args.p_sentence,
                          "p_hyperbaton": args.p_hyperbaton,
                          "clause_only": bool(args.clause_only),
                          "seed": args.seed}
    cfg["training"]["max_epochs"] = -1

    for name in ("tagger", "parser"):
        if name not in cfg["components"]:
            continue
        block = cfg["initialize"].setdefault("components", {}).setdefault(name, {})
        block["labels"] = {"@readers": "spacy.read_labels.v1",
                           "path": f"{args.labels_dir.rstrip('/')}/{name}.json", "require": True}

    cfg.to_disk(args.out)
    print(f"wrote {args.out}  (lang={args.lang}, p_sentence={args.p_sentence}, "
          f"p_hyperbaton={args.p_hyperbaton}, clause_only={bool(args.clause_only)}, "
          f"max_epochs=-1, shuffle=true, labels={args.labels_dir})")


if __name__ == "__main__":
    main()
