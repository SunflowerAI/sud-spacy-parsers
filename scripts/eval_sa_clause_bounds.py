#!/usr/bin/env python3
"""Score a merged-corpus Sanskrit arm with and without the CROSS-CLAUSAL decoding constraint.

The constraint forbids any arc spanning a clause mark unless its label is in an allowed set — see
`sud_constrained_parse.parse_with_clause_bounds`, which also records why the mark set defaults to
the double daṇḍa alone.

Reports LAS/UAS overall AND on the crossing arcs themselves, since a constraint aimed at those can
easily buy them at the expense of everything else.

    eval_sa_clause_bounds.py MODEL CORPUS.spacy [--allowed strict|merge|LIST] [--marks ‖|both]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401

import spacy  # noqa: E402
from gold_tok_corpus import GoldTokNormCorpus  # noqa: E402
from sud_constrained_parse import (MERGE_RELATIONS, parse_with_clause_bounds,  # noqa: E402
                                   parse_with_unit_roots)

SETS = {"strict": {"conj:coord", "parataxis"}, "merge": MERGE_RELATIONS}


def score(egs, get_pred, marks):
    n = las = uas = 0
    xn = xlas = 0
    fails = 0
    for eg in egs:
        ref = eg.reference
        pred = get_pred(eg)
        if pred is None or len(pred) != len(ref):
            fails += 1
            continue
        ms = {t.i for t in ref if t.text in marks}
        for i, gt in enumerate(ref):
            if gt.head.i == i:
                continue
            n += 1
            ok_h = pred[i].head.i == gt.head.i
            ok = ok_h and pred[i].dep_ == gt.dep_
            uas += ok_h
            las += ok
            lo, hi = sorted((i, gt.head.i))
            if any(lo < m < hi for m in ms):
                xn += 1
                xlas += ok
    return n, las, uas, xn, xlas, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("corpus")
    ap.add_argument("--allowed", default="strict")
    ap.add_argument("--marks", default="double", choices=["double", "both"])
    ap.add_argument("--mode", default="label", choices=["label", "unit_roots"],
                    help="label: allowed-relation mask. unit_roots: two-pass, a "
                         "crossing arc may only have a unit ROOT as its dependent.")
    a = ap.parse_args()

    marks = ("‖",) if a.marks == "double" else ("‖", "|")
    allowed = SETS.get(a.allowed) or set(a.allowed.split(","))
    nlp = spacy.load(a.model)
    parser = nlp.get_pipe("parser")
    egs = list(GoldTokNormCorpus(a.corpus)(nlp))

    desc = f"allowed={sorted(allowed)}" if a.mode == "label" else "mode=unit_roots"
    print(f"{a.model}\n  {desc}  marks={list(marks)}  {len(egs)} examples")
    for label, fn in (("unconstrained", lambda eg: nlp(eg.predicted.copy())),
                      ("constrained", (lambda eg: parse_with_clause_bounds(
                          parser, nlp(eg.predicted.copy()), allowed, marks))
                       if a.mode == "label" else
                       (lambda eg: parse_with_unit_roots(
                           parser, nlp(eg.predicted.copy()), marks)))):
        n, las, uas, xn, xlas, fails = score(egs, fn, marks)
        print(f"  {label:14s} LAS {100*las/n:.2f}  UAS {100*uas/n:.2f}   "
              f"crossing-arc LAS {100*xlas/xn:.2f} on {xn} arcs"
              + (f"   ⚠ {fails} decodes failed" if fails else ""))


if __name__ == "__main__":
    main()
