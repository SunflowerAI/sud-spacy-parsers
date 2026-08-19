#!/usr/bin/env python3
"""Write the three Korean channel configs off `configs/config_ko_eojeol.cfg`, single-variable.

    config_ko_analyser.cfg      the analyser channel
    config_ko_analyser_ctl.cfg  the same, `constant = true` -- the capacity control
    config_ko_order.cfg         the analyser channel plus the constrained scrambler

The point of generating them rather than hand-editing three files is that the diff against the base
config stays readable: the first two differ from it in ONE block, and a reader can see that the
control differs from the channel in one boolean.

⚠ `interpolate=False`. The default resolves `${paths.train}` to null and silently breaks the CLI
path overrides (CLAUDE.md; it caused E913).

⚠ Three edits the AUGMENTED config needs, each of which has bitten this repo before (hazard 9):

  * `max_epochs = -1`, not 0. With 0, spaCy lists the corpus ONCE, so one linearisation is sampled
    per document for the whole run -- and the run looks entirely normal.
  * `shuffle = true` on the reader, because `create_train_batches` shuffles the example list only on
    the `max_epochs >= 0` branch, which an augmented run does not take. `spacy.Corpus.v1` does not
    expose the argument, hence `sud.ShuffledCorpus.v1`.
  * labels collected over several augmented passes (`scripts/init_aug_labels.py`), because a label
    missing from the single pass spaCy would otherwise use does not raise -- it teaches label 0.

    .venv/bin/python scripts/make_ko_configs.py
"""
from thinc.api import Config

BASE = "configs/config_ko_eojeol.cfg"

EMBED = {
    "@architectures": "sud.KoAnalyserEmbed.v1",
    "width": "${components.tok2vec.model.encode.width}",
    "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
    "rows": [5000, 1000, 2500, 2500],
    # [first morpheme, last morpheme] -- the lexical key and the functional one. Sized off the
    # treebank's own type counts: 10 569 first-morpheme and 5 888 last-morpheme keys in train.
    "morph_rows": [5000, 2000],
    "feats": ["First", "Last", "Bag"],
    "constant": False,
    "include_static_vectors": False,
}


def channel(constant: bool) -> Config:
    cfg = Config().from_disk(BASE, interpolate=False)
    embed = dict(EMBED)
    embed["constant"] = constant
    cfg["components"]["tok2vec"]["model"]["embed"] = embed
    return cfg


def main() -> None:
    for out, constant in (("configs/config_ko_analyser.cfg", False),
                          ("configs/config_ko_analyser_ctl.cfg", True)):
        channel(constant).to_disk(out)
        print(f"wrote {out}  (constant = {str(constant).lower()})")

    cfg = channel(False)
    cfg["training"]["max_epochs"] = -1
    cfg["corpora"]["train"]["@readers"] = "sud.ShuffledCorpus.v1"
    cfg["corpora"]["train"]["shuffle"] = True
    cfg["corpora"]["train"]["augmenter"] = {
        "@augmenters": "sud.ko_order_variants.v1",
        "p_order": 0.5,
        "p_head": 0.5,
        "table": "scripts/ko_order_bigrams.json",
        "seed": 0,
    }
    for comp in ("tagger", "parser"):
        cfg["initialize"]["components"][comp] = {
            "labels": {"@readers": "spacy.read_labels.v1",
                       "path": f"labels_ko_order/{comp}.json", "require": True}}
    cfg.to_disk("configs/config_ko_order.cfg")
    print("wrote configs/config_ko_order.cfg")


if __name__ == "__main__":
    main()
