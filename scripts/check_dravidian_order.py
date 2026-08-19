#!/usr/bin/env python3
"""Assert that the Dravidian order augmenter moves the STRING and nothing else.

⚠ THIS IS NOT A FORMALITY. A permutation bug in HEAD does not raise: it yields a perfectly
well-formed `Example` carrying a DIFFERENT tree, trains happily, and shows up in no log and no
metric. `train_la_order.sh` says the same thing about the Latin augmenter and for the same reason.

What is checked, per document, against the augmented example:

  1. **The arc set is identical up to the permutation.** Every gold arc is re-expressed as a pair
     of TOKEN IDENTITIES (which original token heads which original token) and the two multisets
     must be equal. This is the check that a wrong `HEAD` re-index cannot survive.
  2. **Every per-token annotation travelled with its token** — LEMMA, POS, TAG, MORPH, DEP.
  3. **It is a permutation**: each original token appears exactly once.
  4. **Sentence boundaries did not move**, so `SENT_START` is still valid.

And then reported, because a transform that is correct but does nothing is also a bug:

  * how many sentences actually changed
  * the crossing-arc rate before and after — the number `p_hyperbaton` is calibrated against
  * head-finality, before and after: the share of dependents preceding their head, which must be
    essentially UNCHANGED, since the side of the head is read off the data and never assigned

    check_dravidian_order.py corpus_ta/ta_ttb_mwtt-sud-train.spacy --lang ta --docs 200
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import spacy
from spacy.tokens import DocBin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dravidian_augment import reorder_example, sentence_starts  # noqa: E402
from dravidian_order import POLICIES, OrderPolicy, Tok, crossing_arcs, reorder_doc  # noqa: E402


def arc_identities(doc):
    """Arcs as (head token index, dep token index) in the ORIGINAL numbering."""
    return sorted((t.head.i, t.i, t.dep_) for t in doc)


def head_final_rate(doc):
    pre = tot = 0
    for t in doc:
        if t.head.i == t.i or t.pos_ == "PUNCT":
            continue
        tot += 1
        pre += t.i < t.head.i
    return pre, tot


def sentence_bounds(doc):
    return [i for i, t in enumerate(doc) if t.is_sent_start]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--lang", default="ta", choices=sorted(POLICIES))
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-sentence", type=float, default=1.0,
                    help="1.0 for checking: every sentence is exercised, not half of them")
    ap.add_argument("--clause-only", type=int, default=1)
    args = ap.parse_args()

    nlp = spacy.blank(args.lang)
    docs = list(DocBin().from_disk(args.corpus).get_docs(nlp.vocab))
    if args.docs:
        docs = docs[:args.docs]
    base = POLICIES[args.lang]
    policy = OrderPolicy(p_sentence=args.p_sentence, p_hyperbaton=base.p_hyperbaton,
                         clause_only=bool(args.clause_only))
    rng = random.Random(args.seed)

    from spacy.training import Example
    changed = toks = 0
    cross_before = cross_after = arcs_total = 0
    pre_b = tot_b = pre_a = tot_a = 0
    failures: list[str] = []

    # The permutation is COMPUTED, not recovered from the output. Recovering it by matching forms
    # cannot distinguish a genuine arc bug from two identical tokens swapping places, and Tamil
    # corpora are full of repeated function words -- the first version of this check reported 13
    # false failures for exactly that reason. Two RNGs with the same seed, driven through the same
    # call sequence, give the checker the same permutation the augmenter used.
    rng_plan = random.Random(args.seed)

    for d, doc in enumerate(docs):
        ex = Example(nlp.make_doc(doc.text), doc)
        ref = ex.reference
        plan = reorder_doc(
            [Tok(form=t.text, lemma=t.lemma_, upos=t.pos_, deprel=t.dep_,
                 head=t.head.i if t.head.i != t.i else -1,
                 feats=str(t.morph), space_after=bool(t.whitespace_)) for t in ref],
            sentence_starts(ref), rng_plan, policy)
        out = reorder_example(nlp, ex, rng, policy)
        new = out.reference
        toks += len(ref)

        if len(new) != len(ref):
            failures.append(f"doc {d}: length {len(ref)} -> {len(new)}")
            continue

        order = plan.order
        if sorted(order) != list(range(len(ref))):
            failures.append(f"doc {d}: not a permutation")
            continue
        where = {old: k for k, old in enumerate(order)}
        want = sorted((where[h], where[i], dep) for h, i, dep in arc_identities(ref))
        got = arc_identities(new)
        if want != got:
            bad = [a for a in got if a not in want][:3]
            failures.append(f"doc {d}: ARC SET CHANGED, e.g. {bad}")
            continue
        for k, old in enumerate(order):
            a, b = ref[old], new[k]
            if (a.lemma_, a.pos_, a.tag_, str(a.morph), a.dep_) != \
               (b.lemma_, b.pos_, b.tag_, str(b.morph), b.dep_):
                failures.append(f"doc {d}: annotation did not travel with token {old}")
                break

        # 4. sentence boundaries
        if sentence_bounds(ref) != sentence_bounds(new):
            failures.append(f"doc {d}: sentence boundaries moved")

        changed += order != list(range(len(ref)))
        for s, chunk in ((0, ref), (1, new)):
            starts = sentence_starts(chunk) + [len(chunk)]
            for a, b in zip(starts, starts[1:]):
                heads = [(t.head.i - a if a <= t.head.i < b and t.head.i != t.i else -1)
                         for t in chunk[a:b]]
                c = crossing_arcs(heads)
                if s:
                    cross_after += c
                else:
                    cross_before += c
                    arcs_total += 1
        p, t = head_final_rate(ref)
        pre_b += p
        tot_b += t
        p, t = head_final_rate(new)
        pre_a += p
        tot_a += t

    print(f"{len(docs)} documents, {toks} tokens, lang={args.lang}, "
          f"clause_only={bool(args.clause_only)}, p_hyperbaton={policy.p_hyperbaton}")
    print(f"  documents whose order changed : {changed}/{len(docs)}")
    print(f"  crossing arc pairs  before {cross_before}  after {cross_after}")
    print(f"  dependent precedes head  before {pre_b/max(tot_b,1):.3%}  "
          f"after {pre_a/max(tot_a,1):.3%}   (must be ~unchanged)")
    if failures:
        print(f"\n!! {len(failures)} FAILURES")
        for line in failures[:10]:
            print("   " + line)
        raise SystemExit(1)
    print("\nOK: the tree, every per-token annotation, and every sentence boundary survived.")




if __name__ == "__main__":
    main()
