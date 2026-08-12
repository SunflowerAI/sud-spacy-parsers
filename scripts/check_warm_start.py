#!/usr/bin/env python3
"""Verify a `--warm-start` arm begins life AS the released tagger, before any training.

The whole claim of `sud.WarmStartTagger.v1` is that the new columns are inert at step 0, so the
untrained conditioned model predicts EXACTLY what the released tagger predicts. If that holds, any
later difference is something training chose; if it does not -- a label-order mismatch, a
side-channel column that is not actually zero, an encoder copied into the wrong sub-model -- the
arm is silently starting somewhere else, and its result would be uninterpretable rather than wrong
in any visible way.

    check_warm_start.py configs/config_ar_xposwarm.cfg training_ar_lemma/model-best \\
        --train corpus_ar_ext/ar_padt-sud-train.relabeled_ext.spacy --limit 50
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_code  # noqa: F401,E402

import spacy                                            # noqa: E402
from spacy.tokens import Doc, DocBin                    # noqa: E402
from spacy.training.initialize import init_nlp          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("released", help="the arm the config warm-starts from")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()

    cfg = spacy.util.load_config(a.config, overrides={"paths.train": a.train,
                                                      "paths.dev": a.dev or a.train})
    nlp = init_nlp(cfg)                      # runs after_init, i.e. the warm start
    rel = spacy.load(a.released)

    new_t, rel_t = nlp.get_pipe("tagger"), rel.get_pipe("tagger")
    print(f"labels           new {len(new_t.labels)}  released {len(rel_t.labels)}  "
          f"same order: {list(new_t.labels) == list(rel_t.labels)}")
    W = new_t.model.get_ref("output_layer").get_param("W")
    Wr = rel_t.model.get_ref("output_layer").get_param("W")
    side = W[:, Wr.shape[1]:]
    print(f"head             {Wr.shape} -> {W.shape}; "
          f"first {Wr.shape[1]} cols identical: {bool((W[:, :Wr.shape[1]] == Wr).all())}; "
          f"new {side.shape[1]} cols all zero: {bool((side == 0).all())}")

    # la's --paths.train is a DIRECTORY (plain + macron copies), not a single .spacy file
    tp = pathlib.Path(a.train)
    files = sorted(tp.glob("*.spacy")) if tp.is_dir() else [tp]
    docs = []
    for f in files:
        docs.extend(DocBin().from_disk(f).get_docs(nlp.vocab))
        if len(docs) >= a.limit:
            break
    docs = docs[:a.limit]
    n = same = 0
    for g in docs:
        words = [t.text for t in g]
        spaces = [t.whitespace_ == " " for t in g]
        a_doc = nlp(Doc(nlp.vocab, words=words, spaces=spaces))
        b_doc = rel(Doc(rel.vocab, words=words, spaces=spaces))
        for x, y in zip(a_doc, b_doc):
            n += 1
            same += x.tag_ == y.tag_
    print(f"predictions      {same}/{n} tokens identical to the released tagger "
          f"({100*same/max(n,1):.2f} %)")
    if same != n:
        raise SystemExit("\nFAIL: the warm-started model does NOT start as the released tagger")
    print("\nOK -- starts as the released tagger, to the token")


if __name__ == "__main__":
    main()
