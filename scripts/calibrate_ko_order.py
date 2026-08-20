#!/usr/bin/env python3
"""What a Korean word-order augmenter is allowed to do, measured on the treebank rather than assumed.

Latin's augmenter had a rate to calibrate (`p_hyperbaton` against the corpus's own 37.75 % crossing
rate). Korean has almost no such freedom in the vertical dimension and a great deal in the
horizontal one, so the quantities are different:

  * WHICH RELATIONS ARE FIXED. Korean is head-final, and the relations that are not
    (`flat`, `conj:coord`, `conj:appos`) are head-initial by SUD's chaining convention, not by
    Korean word order. Neither kind may be re-linearised: the first because it would produce
    ungrammatical strings, the second because it would produce a different annotation.
  * HOW SIBLINGS ORDER AMONG THEMSELVES. This is the one real degree of freedom — scrambling among
    the pre-head dependents of a verb — and it is NOT uniform. `mod` before `subj` is not as likely
    as `subj` before `mod`, and an augmenter that shuffles uniformly teaches orders Korean does not
    use. The bigram table this writes lets the augmenter SAMPLE from the attested distribution
    instead of from a uniform one.

    .venv/bin/python scripts/calibrate_ko_order.py \
        assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu \
        --out scripts/ko_order_bigrams.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

BOS = "<s>"


def sents(path: pathlib.Path):
    s = []
    for line in path.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if s:
                yield s
                s = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        s.append(f)
    if s:
        yield s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    before = collections.Counter()
    total = collections.Counter()
    sib_counts = collections.Counter()
    bigrams = collections.Counter()
    unigrams = collections.Counter()
    pair_order = collections.Counter()
    jx_initial = jx_total = 0
    n_sent = 0

    for s in sents(args.conllu):
        n_sent += 1
        kids = collections.defaultdict(list)
        for f in s:
            h = int(f[6])
            i = int(f[0])
            d = f[7]
            total[d.split("@")[0]] += 1
            if i < h:
                before[d.split("@")[0]] += 1
            if h and d != "punct":
                kids[h].append(i)
        first = int(s[0][0])
        for f in s:
            if f[4].split("+")[-1] == "JX":
                jx_total += 1
                jx_initial += int(f[0]) == first
        for h, ks in kids.items():
            pre = [k for k in ks if k < h]
            sib_counts[len(pre)] += 1
            if len(pre) < 2:
                continue
            deps = [s[k - 1][7].split("@")[0] for k in sorted(pre)]
            prev = BOS
            for d in deps:
                bigrams[(prev, d)] += 1
                unigrams[prev] += 1
                prev = d
            for a in range(len(deps)):
                for b in range(a + 1, len(deps)):
                    x, y = deps[a], deps[b]
                    pair_order[(min(x, y), max(x, y), x < y or x == y)] += 1

    print(f"{args.conllu.name}: {n_sent} sentences\n")
    print("relation          n     % dependent BEFORE head")
    for d, n in total.most_common(12):
        print(f"  {d:<14}{n:>7}   {before[d]/n:6.1%}")
    print("\npre-head dependents per head:",
          {k: v for k, v in sorted(sib_counts.items()) if k < 7})
    npre = sum(v for k, v in sib_counts.items() if k >= 2)
    print(f"heads with 2 or more (the permutable ones): {npre} "
          f"= {npre/sum(sib_counts.values()):.1%} of heads")
    print(f"\ntopic-marked (JX-final) eojeol: {jx_total}, "
          f"{jx_initial/jx_total:.1%} of them sentence-initial")

    print("\nattested order of sibling pairs (both pre-head, either order):")
    seen = set()
    rows = []
    for (x, y, _), _n in pair_order.items():
        if (x, y) in seen or x == y:
            # A pair of like relations has no order to attest — `mod before mod` is 100 % by
            # construction, and printing it would read as a finding.
            continue
        seen.add((x, y))
        fwd = pair_order[(x, y, True)]
        rev = pair_order[(x, y, False)]
        if fwd + rev >= 40:
            rows.append((fwd + rev, x, y, fwd / (fwd + rev)))
    for n, x, y, p in sorted(rows, reverse=True)[:12]:
        print(f"  {x:>10} before {y:<12}{n:>6}   {p:6.1%}")

    if args.out:
        table = {"bigrams": {f"{a}\t{b}": n for (a, b), n in bigrams.items()},
                 "contexts": dict(unigrams)}
        args.out.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out} ({len(bigrams)} bigrams over {len(unigrams)} contexts)")


if __name__ == "__main__":
    main()
