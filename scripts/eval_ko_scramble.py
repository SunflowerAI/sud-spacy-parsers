#!/usr/bin/env python3
"""How much of the Korean parser's accuracy is carried by the ORDER of pre-head siblings?

The one degree of freedom Korean word order has is which pre-head dependent of a head comes first,
and `scripts/ko_order.py` re-linearises exactly that, leaving the tree untouched. Scoring an arm
against its OWN re-linearised gold therefore isolates order-sensitivity from everything else: the
sentence means the same thing, has the same tree, and only the string differs.

Two styles, and they answer different questions:

  attested   siblings resampled from the corpus's own bigram distribution over relations — orders
             Korean uses, at roughly the rates Korean uses them
  uniform    siblings shuffled uniformly — includes orders Korean barely uses (`comp:obj` before
             `subj` at 50 % against an attested 3.9 %). The worst case, not a recipe.

The comparison this exists for: Latin's augmenter took its spread across word orders from 17.44 to
8.38 while buying +0.13 on natural order (docs/latin.md). The Korean spread is what says whether
there is a comparable amount to win here.

    .venv/bin/python scripts/eval_ko_scramble.py \
        corpus_ko_eojeol/ko_gsd-sud-test.relabeled_ext.spacy \
        --model released=training_ko_eojeol_lemma/model-best \
        --model analyser=training_ko_analyser_s0/model-best

⚠ Gold sentences, gold tokens, as `--gold-preproc` gives: each gold sentence is parsed as its own
Doc. Sentence segmentation is a different defect with a different fix (CLAUDE.md hazard 4).
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import spacy
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.training import Example

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ko_order  # noqa: E402


def rendered(docs, nlp, style: str, seed: int, table: str):
    """The gold docs, re-linearised. `identity` returns them untouched and must therefore score
    exactly what an ordinary evaluation scores — which is what makes the rest of the column
    meaningful."""
    if style == "identity":
        return docs
    bigrams = None if style == "uniform" else ko_order.load_table(table)[0]
    rng = random.Random(seed)
    out = []
    for doc in docs:
        eg = Example(Doc(nlp.vocab, words=[t.text for t in doc],
                         spaces=[bool(t.whitespace_) for t in doc]), doc)
        out.append(ko_order.order_example(nlp, eg, rng, 1.0, bigrams).reference)
    return out


def score(nlp, docs):
    uas = las = n = 0
    for g in docs:
        for sent in g.sents:
            pred = nlp(Doc(nlp.vocab, words=[t.text for t in sent],
                           spaces=[bool(t.whitespace_) for t in sent]))
            for k, gt in enumerate(sent):
                # Punctuation excluded, as `spacy.parser_scorer.v1` excludes it, so these figures
                # sit on the same scale as every `metrics/ko/metrics_ko_*.json` and as `eval_ko_oov.py`.
                if gt.dep_ in ("punct", "p"):
                    continue
                n += 1
                if pred[k].head.i == gt.head.i - sent.start:
                    uas += 1
                    las += pred[k].dep_ == gt.dep_
    return 100 * uas / n, 100 * las / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--code", type=pathlib.Path, default=pathlib.Path("scripts/seg_code.py"))
    ap.add_argument("--table", default="scripts/ko_order_bigrams.json")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    blank = spacy.blank("ko")
    gold = list(DocBin().from_disk(args.corpus).get_docs(blank.vocab))

    renderings = [("identity", 0)]
    renderings += [("attested", s) for s in range(args.seeds)]
    renderings += [("uniform", s) for s in range(args.seeds)]
    sets = {(st, sd): rendered(gold, blank, st, sd, args.table) for st, sd in renderings}
    moved = {}
    for (st, sd), ds in sets.items():
        if st == "identity":
            continue
        n = sum(1 for a, b in zip(gold, ds) for x, y in zip(a, b) if x.text != y.text)
        moved[(st, sd)] = n / sum(len(d) for d in gold)

    print(f"{args.corpus}: {sum(len(d) for d in gold)} tokens")
    for st in ("attested", "uniform"):
        share = [moved[(st, s)] for s in range(args.seeds)]
        print(f"  {st:<9} moves {sum(share)/len(share):.1%} of tokens off their attested position")
    print()
    print(f"{'arm':<24}{'identity':>18}{'attested':>18}{'uniform':>18}")
    print(f"{'':<24}" + "".join(f"{'UAS    LAS':>18}" for _ in range(3)))
    for spec in args.model:
        name, path = spec.split("=", 1)
        nlp = spacy.load(path)
        row = f"{name:<24}"
        for st in ("identity", "attested", "uniform"):
            seeds = [0] if st == "identity" else list(range(args.seeds))
            got = [score(nlp, sets[(st, s)]) for s in seeds]
            row += f"{sum(u for u, _, _ in got)/len(got):10.2f}{sum(l for _, l, _ in got)/len(got):8.2f}"
        print(row)


if __name__ == "__main__":
    main()
