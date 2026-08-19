#!/usr/bin/env python3
"""Set ``p_hyperbaton`` against the corpus's own discontinuity, and check the constraints hold.

A recursive re-linearisation is projective by construction, so an un-tuned word-order augmenter
hands the parser a Latin with no hyperbaton in it at all while 37.75 % of the test set has some.
This sweeps the displacement rate and reports, per setting, how much of the corpus's own
discontinuity the augmented copy reproduces — and, in the same pass, whether the closed-class
constraints survived, because a shuffle that quietly separates ``-que`` from its host would look
fine in every aggregate here.

The last column is the one that decides whether this is affordable at all. spaCy's parser is
projective, so a non-projective gold tree is PSEUDO-PROJECTIVISED — the deprel gets a ``||``
suffix recording the arc that was lifted. New displacement patterns therefore mint new parser
labels, and a label the parser was not initialised with is a training crash, not a silent loss.
``NEW`` counts labels the original corpus does not already contain.

    .venv/bin/python scripts/calibrate_la_order.py assets_la/…-train.…conllu
    .venv/bin/python scripts/calibrate_la_order.py …conllu --rates 0.10,0.12 --show 3
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from la_order import (CONNECTIVE_INITIAL, ENCLITICS, OrderPolicy, POSTPOSITIONS,  # noqa: E402
                      Tok, WACKERNAGEL, crossing_arcs, fold, reorder_sentence)

from spacy.pipeline._parser_internals import nonproj  # noqa: E402


def read_conllu(path: Path) -> list[list[list[str]]]:
    sents, rows = [], []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if rows:
                sents.append(rows)
                rows = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0].isdigit():
            rows.append(f)
    if rows:
        sents.append(rows)
    return sents


def to_toks(rows: list[list[str]]) -> list[Tok]:
    return [Tok(form=f[1], lemma=f[2] if f[2] != "_" else "", upos=f[3], deprel=f[7],
                head=int(f[6]) - 1, feats="" if f[5] == "_" else f[5],
                space_after="SpaceAfter=No" not in f[9].split("|"))
            for f in rows]


def render(toks: list[Tok], order: list[int], forms: list[str], spaces: list[bool]) -> str:
    out = []
    for k, _ in enumerate(order):
        out.append(forms[k])
        if spaces[k]:
            out.append(" ")
    return "".join(out).strip()


def violations(toks: list[Tok], order: list[int]) -> Counter:
    """Every constraint the module claims, checked on the OUTPUT rather than trusted."""
    bad = Counter()
    pos = {i: k for k, i in enumerate(order)}
    n = len(toks)
    kids: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, t in enumerate(toks):
        if 0 <= t.head < n and t.head != i:
            kids[t.head].append(i)

    for i, t in enumerate(toks):
        # an enclitic still directly follows the word it followed before
        if i > 0 and fold(t.form) in ENCLITICS and t.upos == "CCONJ" \
                and t.deprel.split("@")[0].split(":")[0] == "cc" \
                and not kids[i] and toks[i - 1].upos != "PUNCT":
            if pos[i] != pos[i - 1] + 1:
                bad["enclitic detached"] += 1
        # A preposition still precedes everything it governs -- everything, that is, but the
        # things that are allowed in front of one: the coordinator that marks the whole conjunct
        # (`sed secundum ...`) and punctuation, neither of which the constraint is about.
        if t.upos == "ADP" and kids[i] and fold(t.lemma) not in POSTPOSITIONS:
            span = [j for j in _subtree(i, kids) if j != i and toks[j].upos != "PUNCT"
                    and toks[j].deprel.split("@")[0].split(":")[0] != "cc"
                    and fold(toks[j].lemma) not in CONNECTIVE_INITIAL]
            if any(pos[j] < pos[i] for j in span):
                bad["preposition split (hyperbaton)"] += 1
        # A Wackernagel particle sits second in its clause, counted in WORDS: punctuation is not
        # one, which is the same convention the corpus table in la_order.py was measured under.
        if fold(t.lemma) in WACKERNAGEL and not kids[i] and 0 <= t.head < n:
            span = sorted(pos[j] for j in _subtree(t.head, kids) if j in pos)
            words = [k for k in span if toks[order[k]].upos != "PUNCT"]
            if pos[i] in words and words.index(pos[i]) > 2:
                bad["wackernagel misplaced"] += 1
    if toks[-1].upos == "PUNCT" and order[-1] != n - 1:
        bad["final punct moved"] += 1
    return bad


def _subtree(i: int, kids: dict[int, list[int]]) -> list[int]:
    out, stack = [], [i]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(kids[v])
    return out


def projective_labels(heads: list[int], labels: list[str]) -> set[str]:
    """The deprels spaCy would actually train on, after pseudo-projectivising."""
    proj_heads = [h if h >= 0 else i for i, h in enumerate(heads)]
    _, deco = nonproj.projectivize(proj_heads, labels)
    return set(deco)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu", type=Path)
    ap.add_argument("--rates", default="0.0,0.05,0.10,0.15,0.20,0.30",
                    help="comma-separated p_hyperbaton settings to sweep")
    ap.add_argument("--p-rise", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0, help="sentences to use (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print N reordered sentences per rate")
    args = ap.parse_args()

    sents = read_conllu(args.conllu)
    if args.limit:
        sents = sents[:args.limit]
    parsed = [to_toks(s) for s in sents]

    base_np = sum(1 for t in parsed if crossing_arcs([x.head for x in t]))
    base_pairs = sum(crossing_arcs([x.head for x in t]) for t in parsed)
    base_labels: set[str] = set()
    for t in parsed:
        base_labels |= projective_labels([x.head for x in t], [x.deprel for x in t])
    n = len(parsed)
    print(f"{args.conllu.name}: {n} sentences, "
          f"{sum(len(t) for t in parsed)} tokens")
    print(f"  corpus as it stands: {base_np / n:.2%} of sentences have a crossing arc "
          f"({base_pairs / n:.2f} crossing pairs per sentence), "
          f"{len(base_labels)} projectivised deprel labels\n")

    hdr = (f"{'p_hyp':>6s} {'crossing':>9s} {'pairs/s':>8s} {'moved':>7s} "
           f"{'labels':>7s} {'NEW':>5s}  violations")
    print(hdr)
    print("-" * len(hdr))
    for rate in [float(r) for r in args.rates.split(",")]:
        rng = random.Random(args.seed)
        pol = OrderPolicy(p_sentence=1.0, p_hyperbaton=rate, p_rise=args.p_rise)
        np_sents = pairs = moved = total = 0
        labels: set[str] = set()
        bad: Counter = Counter()
        shown = 0
        for toks in parsed:
            r = reorder_sentence(toks, rng, pol)
            # The tree never moves, so the head list is simply re-indexed by the permutation.
            where = {old: new for new, old in enumerate(r.order)}
            heads = [where[toks[i].head] if 0 <= toks[i].head < len(toks)
                     and toks[i].head != i else -1 for i in r.order]
            deprels = [toks[i].deprel for i in r.order]
            c = crossing_arcs(heads)
            np_sents += bool(c)
            pairs += c
            moved += sum(1 for k, i in enumerate(r.order) if k != i)
            total += len(toks)
            labels |= projective_labels(heads, deprels)
            bad += violations(toks, r.order)
            if shown < args.show and len(toks) > 8:
                shown += 1
                print(f"   -  {render(toks, list(range(len(toks))), [t.form for t in toks], [t.space_after for t in toks])}")
                print(f"   +  {render(toks, r.order, r.forms, r.spaces)}")
        v = ", ".join(f"{k} {c}" for k, c in sorted(bad.items())) or "none"
        print(f"{rate:6.2f} {np_sents / n:8.2%} {pairs / n:8.2f} {moved / total:6.1%} "
              f"{len(labels):7d} {len(labels - base_labels):5d}  {v}")


if __name__ == "__main__":
    main()
