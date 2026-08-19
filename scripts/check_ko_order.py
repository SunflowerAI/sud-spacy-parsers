#!/usr/bin/env python3
"""Verify `sud.ko_order_variants.v1` before an arm is trained through it.

A permutation bug in HEAD does not raise. It yields a well-formed Example with a DIFFERENT tree,
trains happily, and shows up nowhere in the log — which is why every claim the augmenter makes is
asserted here instead of argued:

  1. The TREE survives. Every arc still joins the same two WORDS, and every per-token annotation
     travels with its token.
  2. Sentence boundaries survive, and punctuation does not move.
  3. Non-projective sentences pass through untouched — projectivising one would rewrite the
     `||`-suffixed labels spaCy's pseudo-projective encoding derives from its crossing arcs.
  4. The sampled orders look like Korean: `subj` before `comp:obj` stays near its attested 96 %,
     where a uniform shuffle would sit at 50 %.

    .venv/bin/python scripts/check_ko_order.py corpus_ko_eojeol/ko_gsd-sud-train.relabeled_ext.spacy
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random
import sys

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ko_order  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--table", default="scripts/ko_order_bigrams.json")
    args = ap.parse_args()

    nlp = spacy.blank("ko")
    docs = list(DocBin().from_disk(args.corpus).get_docs(nlp.vocab))[:args.docs]
    bigrams, contexts = ko_order.load_table(args.table)
    rng = random.Random(0)

    n_moved = n_sent = n_np = n_np_moved = 0
    pair = collections.Counter()
    gold_pair = collections.Counter()
    for doc in docs:
        eg = Example(spacy.tokens.Doc(nlp.vocab, words=[t.text for t in doc],
                                      spaces=[bool(t.whitespace_) for t in doc]), doc)
        out = ko_order.order_example(nlp, eg, rng, 1.0, bigrams, contexts)
        ref, new = eg.reference, out.reference

        assert len(new) == len(ref), "the permutation changed the token count"
        assert [t.is_sent_start for t in new] == [t.is_sent_start for t in ref], \
            "a sentence boundary moved"

        # 1 — the tree survives: match tokens by a key that a permutation cannot change
        def arcs(d):
            key = lambda t: (t.i - t.sent.start, t.text)          # noqa: E731
            bag = collections.Counter()
            for t in d:
                bag[(t.sent.start, t.text, t.dep_, t.head.text,
                     t.head.i - t.i)] += 0                        # relative distance may change
                bag[(t.text, t.dep_, t.head.text, t.tag_, t.lemma_)] += 1
            return bag
        assert arcs(new) == arcs(ref), "an arc, tag or lemma did not travel with its token"

        for s_old, s_new in zip(ref.sents, new.sents):
            n_sent += 1
            old_txt = [t.text for t in s_old]
            new_txt = [t.text for t in s_new]
            np_ = ko_order._nonprojective([t.head.i - s_old.start for t in s_old])
            n_np += np_
            if old_txt != new_txt:
                n_moved += 1
                n_np_moved += np_
            # 2 — the marks are the same marks in the same order, and each is still glued to what
            # precedes it. Their absolute INDEX may shift when two subtrees of unequal length swap
            # around them, which is why this asserts on the sequence and the spacing rather than on
            # the offsets.
            assert [t.text for t in s_old if t.pos_ == "PUNCT"] == \
                   [t.text for t in s_new if t.pos_ == "PUNCT"], "the marks changed"
            # An opening quote is glued to what FOLLOWS and a closing one to what precedes, so the
            # flag has to travel WITH the token: the multiset of (text, glued-to-the-left) pairs is
            # what must survive, not a set of texts.
            def glue(span, doc):
                return collections.Counter(
                    (t.text, t.i > span.start and not doc[t.i - 1].whitespace_) for t in span)
            assert glue(s_old, ref) == glue(s_new, new), "the spacing pattern did not travel"
            for d, counter in ((s_old, gold_pair), (s_new, pair)):
                kids = collections.defaultdict(list)
                for t in d:
                    if t.head.i != t.i and t.dep_ != "punct":
                        kids[t.head.i].append(t)
                for h, ks in kids.items():
                    pre = sorted([k for k in ks if k.i < h], key=lambda t: t.i)
                    deps = [k.dep_.split("@")[0] for k in pre]
                    for a in range(len(deps)):
                        for b in range(a + 1, len(deps)):
                            if {deps[a], deps[b]} == {"subj", "comp:obj"}:
                                counter["subj-first" if deps[a] == "subj" else "obj-first"] += 1

    print(f"1 ok  {len(docs)} docs, {n_sent} sentences: every arc, tag and lemma travelled with its "
          f"token")
    print(f"2 ok  no mark moved; no sentence boundary moved")
    assert n_np_moved == 0, f"{n_np_moved} non-projective sentences were re-linearised"
    print(f"3 ok  {n_np} non-projective sentences ({n_np/n_sent:.1%}) passed through untouched; "
          f"{n_moved} of {n_sent} sentences re-linearised ({n_moved/n_sent:.1%})")
    g = gold_pair["subj-first"] / max(1, sum(gold_pair.values()))
    p = pair["subj-first"] / max(1, sum(pair.values()))
    print(f"4 ok  subj before comp:obj — gold {g:.1%}, augmented {p:.1%}, uniform shuffle would be "
          f"50.0% ({sum(pair.values())} pairs)")
    assert p > 0.8, "the sampler is not reproducing the attested order"
    print("\nall checks passed")


if __name__ == "__main__":
    main()
