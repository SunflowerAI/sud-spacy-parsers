#!/usr/bin/env python3
"""Score parser arms by TRAIN-FREQUENCY SLICE, not just at the headline.

WHY. The XPOS-lexicon channel was justified on a mechanism, not on a hunch: a form seen once or
twice has its coarse XPOS field right 88.6 % of the time against 76.0 % for the whole tag, so the
channel should pay on RARE forms, where `NORM`'s embedding is worst estimated. That population is
3.5 % of lzh test tokens at frequency <= 10. A headline LAS averages it away entirely -- the same
reason CLAUDE.md's standing hazard 6 says a decision resting on a rare slice must be re-measured
there and never inferred from the headline.

So this scores each arm on the slice its own rationale names. If the channel is doing what it was
built to do, the gain shows here even when the headline is flat; if it is flat HERE too, the
mechanism is absent and no number of seeds will find it.

Gold tokens throughout (the `--gold-preproc` regime): the doc is built from the gold words and the
pipeline is run component by component, so the tokeniser is bypassed exactly as `spacy evaluate
--gold-preproc` bypasses it.

Usage:

    .venv/bin/python scripts/eval_lex_slices.py \\
        --train assets_lzh/.../lzh_kyoto-sud-train.<suffix>.conllu \\
        --test  assets_lzh/.../lzh_kyoto-sud-test.<suffix>.conllu \\
        --arm baseline=training_lzh_trad/model-best \\
        --arm fields=training_lzh_xposlex/model-best
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_code  # noqa: E402,F401  (registers every custom architecture the arms name)
import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

BUCKETS = [("unseen", 0, 0), ("1-2", 1, 2), ("3-10", 3, 10),
           ("11-50", 11, 50), (">50", 51, 10 ** 9)]

# spaCy's `parser_scorer` scores heads and labels with these gold labels EXCLUDED. Matching it is
# not cosmetic: counting punctuation here would put the "all" column ~10 points below the
# `spacy evaluate` headline the arms are being judged on, and an unreconciled gap in a slice table
# is indistinguishable from a bug in it.
IGNORE = {"p", "punct"}


def norm_dep(d):
    """spaCy uppercases the root label, the treebank does not. Comparing them raw cost every root
    token its LAS -- 341 of the first 471 head-correct "mismatches" -- and showed up only because
    the `all` column refused to reconcile with the `spacy evaluate` headline. Reconcile the
    aggregate against the tool being replaced BEFORE reading any slice off it."""
    return "root" if d.lower() == "root" else d


def read(path):
    sent = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if sent:
                yield sent
            sent = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 8 or "-" in c[0] or "." in c[0]:
            continue
        sent.append((c[1], int(c[6]), c[7]))
    if sent:
        yield sent


def bucket(f):
    for name, lo, hi in BUCKETS:
        if lo <= f <= hi:
            return name
    return ">50"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    a = ap.parse_args()

    freq = collections.Counter()
    for sent in read(a.train):
        for form, _, _ in sent:
            freq[form] += 1
    test = list(read(a.test))
    n_tok = sum(len(s) for s in test)

    counts = collections.Counter(bucket(freq.get(f, 0)) for s in test for f, _, _ in s)
    print(f"test: {len(test)} sentences, {n_tok} tokens")
    print("  slice sizes: " + "  ".join(
        f"{name} {counts[name]} ({counts[name] / n_tok:.2%})" for name, _, _ in BUCKETS))
    print()

    results = {}
    for spec in a.arm:
        name, _, path = spec.partition("=")
        nlp = spacy.load(path)
        uas = collections.Counter()
        las = collections.Counter()
        tot = collections.Counter()
        for sent in test:
            words = [f for f, _, _ in sent]
            doc = Doc(nlp.vocab, words=words, spaces=[False] * len(words))
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            for i, (form, ghead, gdep) in enumerate(sent):
                if gdep.split(":")[0].split("@")[0] in IGNORE:
                    continue
                b = bucket(freq.get(form, 0))
                tot[b] += 1
                tok = doc[i]
                # CoNLL-U head 0 = root; spaCy marks a root as its own head.
                phead = 0 if tok.head.i == tok.i and tok.dep_.lower() == "root" else tok.head.i + 1
                if phead == ghead:
                    uas[b] += 1
                    if norm_dep(tok.dep_) == norm_dep(gdep):
                        las[b] += 1
        results[name] = (uas, las, tot)

    width = max(len(n.partition("=")[0]) for n in a.arm)
    for metric, idx in (("UAS", 0), ("LAS", 1)):
        print(f"  {metric}")
        print(f"    {'arm'.ljust(width)}  " + "  ".join(f"{n:>7s}" for n, _, _ in BUCKETS) + "     all")
        for name, (u, l, t) in results.items():
            got = (u, l)[idx]
            cells = " ".join(
                f"{got[b] / t[b] * 100:7.2f}" if t[b] else "      -" for b, _, _ in BUCKETS)
            allv = sum(got.values()) / max(sum(t.values()), 1) * 100
            print(f"    {name.ljust(width)}  {cells}  {allv:6.2f}")
        print()


if __name__ == "__main__":
    main()
