#!/usr/bin/env python3
"""Derive an orthography-augmented training config from a base config.

Three edits to ``configs/config_la_<layer>.cfg`` -> ``configs/config_la_aug<_layer>.cfg``:

  1. Hang ``sud.la_orth_variants.v1`` off the TRAIN corpus only, so every epoch rewrites each
     document into a freshly sampled edition style (see scripts/la_augment.py). Dev is left alone:
     checkpoint selection must stay comparable with the union arm it is being measured against.
  2. ``max_epochs = -1``. Non-negotiable, and silent if missed -- at ``0`` spaCy materialises the
     corpus once and reshuffles that same list forever, so the augmenter samples one style per
     document for the whole run.
  3. ``shuffle = true`` on the train reader, because ``-1`` also turns off the training loop's own
     shuffling. Under ``sud.GoldTokCorpus.v1`` a document IS an example, so this is the same
     shuffle by another name.

``--retarget`` rewrites the ``source`` of every component, which is what makes the morph and lemma
layers stack on the augmented base rather than the union one.

Loads/saves with interpolation OFF so ``${paths.train}`` survives (CLAUDE.md gotcha).

    make_la_aug_config.py configs/config_la_seg.cfg --out configs/config_la_aug.cfg
    make_la_aug_config.py configs/config_la_morph.cfg --out configs/config_la_aug_morph.cfg \
        --retarget training_la_seg=training_la_aug
"""
import argparse

from thinc.api import Config

#: Rates the augmenter is created with; see la_orth.OrthPolicy for what each one means. Written
#: into the config explicitly rather than left to defaults, so a run's orthography policy is
#: readable from the config it was trained with.
#: components whose labels must be supplied explicitly under a streamed corpus
LABEL_FACTORIES = {"tagger", "parser", "morphologizer", "trainable_lemmatizer", "senter", "ner"}

#: Word-order rates, added on top of the orthographic ones by `--order`. `p_hyperbaton` is set by
#: scripts/calibrate_la_order.py against the corpus's own 37.75 % crossing rate; `p_sentence` is
#: deliberately below 1 so half the epoch still shows the treebank's own linearisation.
ORDER_RATES = {
    "p_sentence": 0.5,
    "p_hyperbaton": 0.08,
    "p_rise": 0.4,
}

DEFAULT_RATES = {
    "p_v": 0.5,
    "p_j": 0.5,
    "p_lig": 0.25,
    "p_capital": 0.5,
    "p_length": 0.5,
    "p_uniform_length": 0.5,
    "p_breve_doc": 0.3,
    "max_breve_rate": 0.5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retarget", default=None,
                    help="OLD=NEW: rewrite every component's `source` directory")
    ap.add_argument("--labels-dir", default=None,
                    help="directory of label JSONs for the trained components; REQUIRED in "
                         "practice, since streaming initialises from 100 examples "
                         "(see scripts/init_aug_labels.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", action="store_true",
                    help="also re-linearise the word order (sud.la_variants.v1 instead of "
                         "sud.la_orth_variants.v1); see scripts/la_order.py")
    for name, value in {**DEFAULT_RATES, **ORDER_RATES}.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = ap.parse_args()

    cfg = Config().from_disk(args.base_config, interpolate=False)

    train = cfg["corpora"]["train"]
    if train.get("@readers") != "sud.GoldTokCorpus.v1":
        raise SystemExit(f"expected the gold-token reader, found {train.get('@readers')!r} -- the "
                         "augmenter assumes one document per example (see make_la_aug_config.py)")
    train["shuffle"] = True
    rates = {**DEFAULT_RATES, **ORDER_RATES} if args.order else DEFAULT_RATES
    train["augmenter"] = {
        "@augmenters": "sud.la_variants.v1" if args.order else "sud.la_orth_variants.v1",
        "seed": args.seed, **{k: getattr(args, k) for k in rates}}
    cfg["training"]["max_epochs"] = -1

    if args.retarget:
        old, new = args.retarget.split("=", 1)
        for name, block in cfg["components"].items():
            if isinstance(block, dict) and "source" in block:
                block["source"] = block["source"].replace(old, new)

    if args.labels_dir:
        init = cfg["initialize"].setdefault("components", {})
        for name, block in cfg["components"].items():
            if not isinstance(block, dict) or "source" in block:
                continue                       # sourced components are frozen; they bring theirs
            if block.get("factory") in LABEL_FACTORIES:
                init.setdefault(name, {})["labels"] = {
                    "@readers": "spacy.read_labels.v1",
                    "path": f"{args.labels_dir.rstrip('/')}/{name}.json",
                    "require": True,
                }

    cfg.to_disk(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
