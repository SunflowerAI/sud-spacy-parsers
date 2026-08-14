#!/usr/bin/env python3
"""Derive a vocalisation-augmented training config from an arm's ordinary one.

The `make_la_aug_config.py` recipe, for `sud.ar_vocal_variants.v1` / `sud.fa_vocal_variants.v1`.
Three edits, and all three are load-bearing -- the augmentation silently does nothing without the
first two, and training dies on the third:

  1. `max_epochs = -1`. At `0` spaCy lists the corpus ONCE and reshuffles that same list every
     epoch, so a corpus-level augmenter samples one style per document for the entire run. The run
     looks normal. It is training on a single fixed perturbation.
  2. `shuffle = true` on the reader, because `-1` turns off the training loop's own shuffling.
     Harmless: under `sud.GoldTokCorpus.v1` a document IS an example, so it is the same shuffle.
  3. `[initialize.components.*.labels]` pointed at a pre-collected inventory, because `init_nlp`
     initialises from `islice(train_corpus(nlp), 100)` and will otherwise truncate.

⚠ Load with `Config().from_disk(p, interpolate=False)`. The default interpolation resolves
`${paths.train}` to null and silently breaks CLI path overrides -- this project's E913.
"""
import argparse
from pathlib import Path

from spacy import util
from thinc.api import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=("ar", "fa"), required=True)
    ap.add_argument("--labels-dir", default=None,
                    help="write [initialize.components.*.labels] pointing here")
    ap.add_argument("--retarget", default=None,
                    help="OLD=NEW: rewrite every component's `source` directory, which is what "
                         "makes the morph and lemma layers stack on the AUGMENTED base rather "
                         "than the released one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-bare", type=float, default=0.40)
    ap.add_argument("--p-full", type=float, default=0.15)
    a = ap.parse_args()

    cfg = Config().from_disk(a.base_config, interpolate=False)
    cfg["training"]["max_epochs"] = -1
    cfg["corpora"]["train"]["shuffle"] = True
    cfg["corpora"]["train"]["augmenter"] = {
        "@augmenters": f"sud.{a.lang}_vocal_variants.v1",
        "seed": a.seed, "p_bare": a.p_bare, "p_full": a.p_full,
    }
    # The DEV corpus is deliberately left un-augmented and BARE. Undiacritised text is what this
    # arm is judged on, so `model-best` must be chosen on it -- otherwise the checkpoint drifts
    # toward the spellings the augmenter happens to have sampled, and the headline regresses to buy
    # a robustness nobody measured.
    cfg["corpora"]["dev"]["augmenter"] = None
    if a.retarget:
        old, new = a.retarget.split("=", 1)
        for comp in cfg.get("components", {}).values():
            if isinstance(comp, dict) and isinstance(comp.get("source"), str):
                comp["source"] = comp["source"].replace(old, new)
    # Only the component this layer ADDS is trained; everything sourced is frozen, so its labels
    # are the only ones that need collecting. The edit-tree lemmatiser is the one that really
    # needs it: its labels are properties of the FORM, so كتاب and كِتاب are different trees and a
    # missing one does not raise -- `get_loss` quietly maps it to label 0.
    if a.labels_dir:
        for comp in ("tagger", "parser", "morphologizer", "lemmatizer"):
            block = cfg.get("components", {}).get(comp)
            if block is not None and "source" not in block:
                cfg.setdefault("initialize", {}).setdefault("components", {})[comp] = {
                    "labels": {"@readers": "spacy.read_labels.v1",
                               "path": f"{a.labels_dir}/{comp}.json", "require": True}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cfg.to_disk(a.out)
    print(f"wrote {a.out}")
    print(f"  max_epochs={cfg['training']['max_epochs']}  "
          f"shuffle={cfg['corpora']['train']['shuffle']}  "
          f"augmenter={cfg['corpora']['train']['augmenter']['@augmenters']}")


if __name__ == "__main__":
    main()
