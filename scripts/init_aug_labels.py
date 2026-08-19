#!/usr/bin/env python3
"""Collect a component's label set over SEVERAL augmented passes, for a streamed training run.

``spacy init labels`` cannot do this, and the reason is a trap worth stating plainly. An augmented
config sets ``max_epochs = -1`` so the corpus is re-read and re-augmented every epoch (see
scripts/la_augment.py); but ``init_nlp`` reacts to that setting by initialising from
``islice(train_corpus(nlp), 100)`` -- the first hundred examples only. The tagger then starts with
639 of the treebank's 1 952 XPOS tags and training dies on the first batch carrying one of the
missing ones. ``spacy init labels`` runs through the same ``init_nlp``, so pointing it at the
augmented config reproduces the truncation instead of fixing it.

The second reason is subtler and does NOT announce itself. Tagger, parser and morphologiser labels
are properties of the TREES, which augmentation never touches, so one clean pass would do. The
**edit-tree lemmatiser's labels are properties of the FORMS**: each label is a string edit from a
word to its lemma, so ``uītae`` -> ``uita``, ``vitae`` -> ``uita`` and ``vītæ`` -> ``uita`` are
three different labels. A tree missing from the initial set does not raise -- ``get_loss`` maps it
to ``self.tree2label.get(tree_id, 0)``, so the token is quietly taught label 0 instead. Hence
``--passes``: enough independent style draws that the trees the augmenter can produce are actually
in the set, and a coverage figure at the end saying how well that worked.

    .venv/bin/python scripts/init_aug_labels.py configs/config_la_aug.cfg labels_la \
        --code scripts/seg_code.py --passes 3 \
        --paths.train corpus_la_ext_macron/...spacy --paths.dev corpus_la_ext_union/dev
"""
from __future__ import annotations

import argparse
from pathlib import Path

import srsly
from spacy import util
from spacy.training.initialize import init_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--code", type=Path, default=None)
    ap.add_argument("--passes", type=int, default=3,
                    help="independent augmented passes to union labels over")
    ap.add_argument("--paths.train", dest="train", default=None)
    ap.add_argument("--paths.dev", dest="dev", default=None)
    args = ap.parse_args()

    if args.code:
        util.import_file("cli_code", args.code)
    overrides = {k: v for k, v in (("paths.train", args.train), ("paths.dev", args.dev)) if v}
    config = util.load_config(args.config, overrides=overrides, interpolate=False)
    # The config being read is the one that will POINT AT the files this script is about to write,
    # so drop those references before initializing -- otherwise the first run of a new arm fails
    # on its own not-yet-existent labels.
    for block in config.get("initialize", {}).get("components", {}).values():
        block.pop("labels", None)

    # Read the sourced components off the RAW config. `load_model_from_config` replaces a sourced
    # component's block with the source model's own factory config, so the `source` key is gone by
    # the time `nlp.config` is interpolated and `get_sourced_components` finds none -- which
    # silently re-initialises pipes that were supposed to be left alone, and then reports a
    # coverage figure for them. That is how this printed "morphologizer 2 labels" and an edit-tree
    # coverage of 0.0000 for a lemmatiser that was not being trained at all.
    sourced_raw = [name for name, block in config.get("components", {}).items()
                   if isinstance(block, dict) and "source" in block]

    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    resolved = nlp.config.interpolate()
    train_corpus = util.resolve_dot_names(resolved, [resolved["training"]["train_corpus"]])[0]
    # The same reader with the augmenter removed, so a coverage figure has something to be compared
    # WITH. `min_action_freq` prunes rare parser labels on ANY corpus, so an absolute miss rate on
    # the augmented pass says nothing on its own.
    plain_cfg = resolved.copy()
    plain_cfg["corpora"]["train"] = {**resolved["corpora"]["train"], "augmenter": None}
    plain_corpus = util.resolve_dot_names(plain_cfg, [plain_cfg["training"]["train_corpus"]])[0]
    training = resolved["training"]
    init_vocab(nlp, data=resolved["initialize"]["vocab_data"],
               lookups=resolved["initialize"]["lookups"])

    def get_examples():
        for _ in range(args.passes):
            yield from train_corpus(nlp)

    # Same set spaCy would train: everything frozen or sourced-and-resumed stays out of the way,
    # so only the component this layer is actually adding gets initialized.
    sourced = set(util.get_sourced_components(resolved)) | set(sourced_raw)
    frozen = training["frozen_components"]
    skip = [*frozen, *[p for p in sourced if p not in frozen]]
    nlp._link_components()
    with nlp.select_pipes(disable=skip):
        nlp.initialize(get_examples)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, component in nlp.pipeline:
        labels = getattr(component, "label_data", None)
        if labels is None or name in skip:
            continue
        out = args.outdir / f"{name}.json"
        srsly.write_json(out, labels)
        n = len(labels) if not isinstance(labels, dict) else len(labels.get("labels", labels))
        print(f"  {name:20s} {n:6d} labels -> {out}")

    # PARSER labels used to need no passes at all: they are properties of the TREES, and the
    # orthographic augmenter never touches a tree. The WORD-ORDER augmenter does not touch one
    # either -- but spaCy's parser is projective, so a non-projective gold tree is
    # pseudo-projectivised and the deprel picks up a `||` suffix naming the arc that was lifted.
    # Those labels are therefore properties of the ORDER, which is exactly what is being resampled,
    # and a missing one is as silent here as a missing edit tree: `init_gold` maps an unknown label
    # to 0. So it gets the same coverage figure.
    parser = next((p for n, p in nlp.pipeline if hasattr(p, "moves") and n not in skip), None)
    if parser is not None:
        from spacy.pipeline._parser_internals import nonproj
        # `moves.labels` is keyed by MOVE ID, not by move name, and each value is that move's own
        # label dict -- so the label set is the union, not one entry of it.
        known = {lab for per_move in parser.moves.labels.values() for lab in per_move}

        def uncovered(augment: bool) -> tuple[int, int]:
            seen = miss = 0
            for eg in train_corpus(nlp) if augment else plain_corpus(nlp):
                ref = eg.reference
                _, deco = nonproj.projectivize([t.head.i for t in ref], [t.dep_ for t in ref])
                for lab, t in zip(deco, ref):
                    if t.head.i == t.i:
                        continue
                    seen += 1
                    miss += lab not in known
            return miss, seen

        miss, seen = uncovered(True)
        base = ""
        if plain_corpus is not None:
            b_miss, b_seen = uncovered(False)
            base = f"; un-augmented corpus {b_miss / max(b_seen, 1):.4f}"
        print(f"  parser-label coverage on a fresh pass: {1 - miss / max(seen, 1):.4f} "
              f"({miss} of {seen} arcs would train against label 0{base})")

    # How much of a FRESH augmented pass the collected labels cover. Only the lemmatiser can miss
    # (its labels are form-derived), and a miss there is silent, so it is worth a number.
    # by capability, not by name: the factory is `trainable_lemmatizer` but the pipe is called
    # `lemmatizer` in this project's configs.
    lemmatizer = next((p for n, p in nlp.pipeline
                       if hasattr(p, "trees") and hasattr(p, "tree2label") and n not in skip), None)
    if lemmatizer is not None:
        seen = miss = 0
        for eg in train_corpus(nlp):
            for token in eg.reference:
                if not token.lemma_:
                    continue
                seen += 1
                tree_id = lemmatizer.trees.add(token.text, token.lemma_)
                if tree_id not in lemmatizer.tree2label:
                    miss += 1
        print(f"  edit-tree coverage on a fresh pass: {1 - miss / max(seen, 1):.4f} "
              f"({miss} of {seen} tokens would train against label 0)")


if __name__ == "__main__":
    main()
