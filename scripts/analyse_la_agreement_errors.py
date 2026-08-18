#!/usr/bin/env python3
"""How many Latin attachment errors could agreement alone have ruled out?

The question this answers is not "how many errors involve an agreeing word" -- that is just a
frequency count of adjectives. It is the useful version: of the attachments the parser got wrong,
how many went to a head the dependent DOES NOT AGREE WITH, when the correct head was one it does?
Those are the errors a model that consulted Case/Number/Gender could in principle have avoided, and
their count is an upper bound on what the per-feature morphology channel can be worth.

Each erroneous attachment on an agreement-bearing dependent falls into one of three buckets:

  detectable   gold head agrees, predicted head does not  -> agreement alone rules the error out
  ambiguous    both heads agree                           -> agreement cannot separate them
  gold-clash   gold head does NOT agree                   -> either an annotation the rule does not
                                                             model (predicative, ExtPos, ellipsis)
                                                             or a treebank error; NOT evidence
                                                             about the parser

Two agreement relations are checked, both restricted to dependents where every relevant feature is
actually present on both tokens, so a missing FEATS value never counts as a disagreement:

  nominal   ADJ / DET / NUM / participle under mod|det|... agrees with its noun in Case+Number+Gender
  subject   a `subj` dependent agrees with its verb in Number (+Person where the pronoun has one)

GOLD morphology is used throughout, deliberately. The point is to bound what the INFORMATION is
worth, not what this particular morphologiser recovers of it -- predicted FEATS would conflate the
two and understate the bound by the morphologiser's own error rate.

    .venv/bin/python scripts/analyse_la_agreement_errors.py \
        corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy \
        --model aug=training_la_aug/model-best --model lemvec=training_la_lemvec/model-best
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy import util
from spacy.tokens import Doc, DocBin

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Nominal agreement is over these three; a value missing on either side means "not comparable".
NOMINAL = ("Case", "Number", "Gender")
#: Dependents that agree with their nominal head. `mod@poss` etc. share the base label.
NOMINAL_DEPS = {"mod", "det", "conj"}
NOMINAL_POS = {"ADJ", "DET", "NUM", "VERB", "AUX", "PRON"}


def base(dep: str) -> str:
    return dep.split("@", 1)[0].split(":", 1)[0]


def feats(tok, keys) -> dict | None:
    """The token's values for ``keys``, or None if any is absent."""
    out = {}
    for k in keys:
        v = tok.morph.get(k)
        if not v:
            return None
        out[k] = tuple(v)
    return out


def agrees(dep_tok, head_tok, keys) -> bool | None:
    """True/False if comparable on every key, None if not comparable at all.

    Multi-valued features (``Case=Nom,Acc``) count as agreeing on any shared value, which is what
    the annotation means -- an underspecified form is compatible with either reading.
    """
    a, b = feats(dep_tok, keys), feats(head_tok, keys)
    if a is None or b is None:
        return None
    return all(set(a[k]) & set(b[k]) for k in keys)


def classify(gold_tok, pred_head_i: int, gold_doc) -> tuple[str, str] | None:
    """(relation, bucket) for one erroneous attachment, or None if agreement does not apply."""
    dep = base(gold_tok.dep_)
    gold_head = gold_tok.head
    pred_head = gold_doc[pred_head_i]
    if gold_head.i == gold_tok.i or pred_head.i == gold_tok.i:
        return None                                    # root: nothing to agree with

    if dep == "subj":
        keys = ("Number",)
        rel = "subject"
    elif dep in NOMINAL_DEPS and gold_tok.pos_ in NOMINAL_POS:
        keys = NOMINAL
        rel = "nominal"
    else:
        return None

    g = agrees(gold_tok, gold_head, keys)
    p = agrees(gold_tok, pred_head, keys)
    if g is None or p is None:
        return None                                    # not comparable; says nothing either way
    if not g:
        return rel, "gold-clash"
    return rel, ("ambiguous" if p else "detectable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--code", type=Path, default=Path("scripts/seg_code.py"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    blank = spacy.blank("la")
    gold_docs = list(DocBin().from_disk(args.corpus).get_docs(blank.vocab))
    if args.limit:
        gold_docs = gold_docs[:args.limit]

    for spec in args.model:
        name, path = spec.split("=", 1)
        nlp = spacy.load(path)
        n_tok = n_err = 0
        buckets: Counter = Counter()
        applicable: Counter = Counter()

        for gold in gold_docs:
            # Gold tokenisation, as --gold-preproc does: the tokeniser is bypassed entirely, so
            # every index lines up and TOK contributes nothing to the error count.
            pred = nlp(Doc(nlp.vocab, words=[t.text for t in gold],
                           spaces=[bool(t.whitespace_) for t in gold]))
            for g, p in zip(gold, pred):
                n_tok += 1
                if p.head.i == g.head.i:
                    continue
                n_err += 1
                got = classify(g, p.head.i, gold)
                if got is None:
                    continue
                rel, bucket = got
                applicable[rel] += 1
                buckets[(rel, bucket)] += 1

        print(f"\n== {name}  ({path})")
        print(f"   {n_tok} tokens, {n_err} wrong attachments (UAS {100 * (1 - n_err / n_tok):.2f})")
        tot_det = 0
        for rel in ("nominal", "subject"):
            if not applicable[rel]:
                continue
            print(f"   {rel}: {applicable[rel]} erroneous attachments where agreement applies")
            for b in ("detectable", "ambiguous", "gold-clash"):
                c = buckets[(rel, b)]
                print(f"     {b:<11} {c:6d}  {c / applicable[rel]:6.1%} of applicable"
                      f"  {c / n_err:6.2%} of all errors")
            tot_det += buckets[(rel, "detectable")]
        print(f"   AGREEMENT-DETECTABLE OVERALL: {tot_det} = {tot_det / n_err:.2%} of all "
              f"attachment errors ({100 * tot_det / n_tok:.2f} UAS points if all were fixed)")


if __name__ == "__main__":
    main()
