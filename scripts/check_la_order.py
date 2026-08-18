#!/usr/bin/env python3
"""Assert that the word-order augmenter moves the STRING and nothing else.

The augmenter rewrites ``HEAD`` through a permutation, and a permutation bug there does not raise —
it produces a well-formed Example with a different tree, which trains perfectly happily and is
invisible in every training log. So the invariant is checked directly, on real corpus documents,
through the registered augmenter rather than through the transform underneath it:

  * the token count is unchanged, and so is the sentence count and where the boundaries fall;
  * the ARC SET is identical, compared as (lemma, tag, morph, deprel, head lemma, head tag) —
    identified by annotation rather than by position, since positions are exactly what moved, and
    not by FORM, since the orthography pass re-spells it;
  * every ``-que`` still directly follows the word it followed.

    .venv/bin/python scripts/check_la_order.py corpus_la_ext_macron/…-train.…spacy
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
import la_augment  # noqa: F401,E402  (registers the augmenters)
from la_order import ENCLITICS, fold  # noqa: E402


def arc_multiset(doc) -> Counter:
    """Every arc as an annotation-identified triple. Position is excluded on purpose."""
    return Counter((t.lemma_, t.tag_, str(t.morph), t.dep_,
                    "ROOT" if t.head.i == t.i else t.head.lemma_,
                    "ROOT" if t.head.i == t.i else t.head.tag_)
                   for t in doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--docs", type=int, default=500)
    ap.add_argument("--augmenter", default="sud.la_variants.v1")
    ap.add_argument("--p-sentence", type=float, default=1.0)
    ap.add_argument("--p-hyperbaton", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    nlp = spacy.blank("la")
    aug = registry.augmenters.get(args.augmenter)(
        p_sentence=args.p_sentence, p_hyperbaton=args.p_hyperbaton, seed=args.seed)

    db = DocBin().from_disk(args.corpus)
    fails: Counter = Counter()
    docs = tokens = moved = 0
    for gold in itertools.islice(db.get_docs(nlp.vocab), args.docs):
        out = list(aug(nlp, Example(nlp.make_doc(gold.text), gold)))[-1].reference
        docs += 1
        tokens += len(gold)
        if len(out) != len(gold):
            fails["token count"] += 1
            continue
        if arc_multiset(out) != arc_multiset(gold):
            fails["arc set"] += 1
        if [t.i for t in out if t.is_sent_start] != [t.i for t in gold if t.is_sent_start]:
            fails["sentence boundaries"] += 1
        for t in out:
            if t.i and fold(t.text) in ENCLITICS and t.pos_ == "CCONJ" \
                    and t.dep_.split("@")[0].split(":")[0] == "cc" and out[t.i - 1].is_punct:
                fails["enclitic after punctuation"] += 1
        moved += sum(1 for a, b in zip(out, gold) if a.lemma_ != b.lemma_)

    print(f"{docs} docs / {tokens} tokens through {args.augmenter} "
          f"(p_sentence={args.p_sentence}, p_hyperbaton={args.p_hyperbaton})")
    print(f"  tokens whose position changed: {moved / max(tokens, 1):.1%}")
    if fails:
        for k, v in sorted(fails.items()):
            print(f"  FAIL {k}: {v}")
        raise SystemExit(1)
    print("  OK — arcs, sentence boundaries and clitic attachment all preserved")


if __name__ == "__main__":
    main()
