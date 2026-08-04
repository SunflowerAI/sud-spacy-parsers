#!/usr/bin/env python
"""Build a .spacy corpus from the merged treebank, leaving the DEFAULT cross-unit edges unsupervised.

`cross_unit_rules.py` commits 31.2 % of cross-unit boundaries from derived rules and fills the rest
with `parataxis` — a default, not a decision. Training on that default cost ~7 LAS on the WITHIN-unit
edges, which are real gold, so the noise plausibly hurt the rest of the tree. This builds the corpus
that tests it: `CrossUnit=default` dependents get NO head and NO label, so the parser is free to
attach and label them from what it learned on the in-unit clause links.

**It has to be head AND label, and it cannot be done through `spacy convert`.** Two traps:

  * A CoNLL-U `_` in DEPREL is kept by spaCy as a LITERAL label, not as missing (the same trap that
    once taught the sandhi transducer `FORM -> "_"`), so blanking the column does not work.
  * Blanking only the LABEL does not free the parser either: `ArcEager._replace_unseen_labels`
    rewrites any label it has not seen to the backoff label `dep`, so the parser would simply be
    taught to emit `dep` at every such boundary. Only `is_head_unknown` makes an arc genuinely
    cost-free, and that needs the HEAD unset too.

So the Docs are built directly, with `heads=[..., None, ...]`, which yields `has_head() == False`.
A parser cannot invent a label in any case — its output vocabulary is fixed to the labels seen in
training — so what this can show is which EXISTING relation it generalises to these boundaries.

Usage:
    build_partial_corpus.py --out DIR FILE.merged.conllu [FILE ...] [--supervise-all]
"""
import argparse
import os

import spacy
from spacy.tokens import Doc, DocBin

GROUP = 10          # sentences per Doc, mirroring `spacy convert -n 10` for the other corpora


def read(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur); cur = []
        elif not line.startswith("#"):
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]:
                cur.append(f)
    if cur:
        sents.append(cur)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--supervise-all", action="store_true",
                    help="keep every edge supervised (the control: same grouping, same code path)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nlp = spacy.blank("xx")

    for path in args.files:
        sents = read(path)
        db = DocBin()
        unsup = tot = 0
        for g in range(0, len(sents), GROUP):
            chunk = sents[g:g + GROUP]
            words, heads, deps, tags, poss, lemmas, starts = [], [], [], [], [], [], []
            for s in chunk:
                base = len(words)
                for k, t in enumerate(s):
                    free = (not args.supervise_all) and "CrossUnit=default" in t[9]
                    words.append(t[1])
                    lemmas.append(t[2])
                    poss.append(t[3] if t[3] != "_" else "")
                    tags.append(t[4] if t[4] != "_" else "")
                    starts.append(k == 0)
                    tot += 1
                    if free:
                        heads.append(None); deps.append(""); unsup += 1
                    elif t[6] == "0":
                        heads.append(base + k); deps.append("ROOT")   # as `spacy convert` maps it
                    else:
                        heads.append(base + int(t[6]) - 1); deps.append(t[7])
            db.add(Doc(nlp.vocab, words=words, spaces=[False] * len(words), heads=heads,
                       deps=deps, tags=tags, pos=poss, lemmas=lemmas, sent_starts=starts))
        name = os.path.basename(path).replace(".conllu", ".spacy")
        db.to_disk(os.path.join(args.out, name))
        print(f"{name}: {len(sents)} sentences, {tot} tokens, "
              f"{unsup} left unsupervised ({100*unsup/tot:.1f} %)")


if __name__ == "__main__":
    main()
