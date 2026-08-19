#!/usr/bin/env python3
"""How many Latin attachment errors sit in non-projective territory?

spaCy's parser is a projective transition system. Non-projective gold trees survive training only
because they are PSEUDO-PROJECTIVISED -- the offending arc is lifted onto an ancestor and the deprel
gains a `||` suffix naming where it came from -- and recovered at decode time by `deprojectivize`,
which reads that suffix back. So the arcs Latin's discontinuity actually consists of are learned
through a lossy detour, and the question is what the detour costs.

Three things are counted, and they are NOT the same number:

  the non-projective ARC     the gold arc that crosses. Can the parser recover it at all?
  its SENTENCE              every other arc in a sentence that contains one. Discontinuity is
                            contagious: one lifted arc perturbs the transition sequence around it.
  the parser's OWN output   how many non-projective arcs it produces, and how many are right.
                            A parser that has quietly learned to never emit one is a different
                            failure from one that emits them in the wrong places.

An arc h->d is non-projective iff some token strictly between h and d is not a descendant of h --
the standard definition, computed here rather than taken from spacy.pipeline._parser_internals so
that the gold side and the predicted side are measured by identical code.

    .venv/bin/python scripts/analyse_la_nonproj_errors.py \
        corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy \
        --model aug=training_la_aug/model-best
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


def nonproj_arcs(heads: list[int]) -> set[int]:
    """Indices whose incoming arc is non-projective, over one sentence's local heads."""
    n = len(heads)
    anc: list[set[int]] = []
    for i in range(n):
        seen, k = set(), i
        for _ in range(n):                       # bounded: a cycle cannot outrun the token count
            k = heads[k]
            if k in seen or k == i:
                break
            seen.add(k)
        anc.append(seen)
    bad = set()
    for d, h in enumerate(heads):
        if h == d:
            continue
        for k in range(min(h, d) + 1, max(h, d)):
            if k != h and h not in anc[k]:
                bad.add(d)
                break
    return bad


def sent_local(span_start: int, toks) -> list[int]:
    return [t.head.i - span_start for t in toks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--code", type=Path, default=Path("scripts/seg_code.py"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="la")
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    blank = spacy.blank(args.lang)
    gold_docs = list(DocBin().from_disk(args.corpus).get_docs(blank.vocab))
    if args.limit:
        gold_docs = gold_docs[:args.limit]

    # --- the gold side, once: it does not depend on the model ---
    gold_np: list[set[int]] = []          # doc-global indices of non-projective gold arcs
    gold_np_sent: list[set[int]] = []     # doc-global indices of tokens IN a non-projective sentence
    n_sent = n_sent_np = 0
    for doc in gold_docs:
        a, s = set(), set()
        for sent in doc.sents:
            n_sent += 1
            bad = nonproj_arcs(sent_local(sent.start, sent))
            if bad:
                n_sent_np += 1
                s.update(range(sent.start, sent.end))
                a.update(sent.start + i for i in bad)
        gold_np.append(a)
        gold_np_sent.append(s)

    n_tok = sum(len(d) for d in gold_docs)
    n_np_arc = sum(len(a) for a in gold_np)
    n_np_tok = sum(len(s) for s in gold_np_sent)
    print(f"GOLD: {len(gold_docs)} docs, {n_sent} sentences, {n_tok} tokens")
    print(f"  non-projective sentences {n_sent_np:6d}  {n_sent_np / n_sent:6.2%}")
    print(f"  tokens in one            {n_np_tok:6d}  {n_np_tok / n_tok:6.2%}")
    print(f"  non-projective ARCS      {n_np_arc:6d}  {n_np_arc / n_tok:6.2%} of tokens")

    for spec in args.model or []:
        name, path = spec.split("=", 1)
        nlp = spacy.load(path)
        c: Counter = Counter()
        for doc, np_arc, np_sent in zip(gold_docs, gold_np, gold_np_sent):
            pred = nlp(Doc(nlp.vocab, words=[t.text for t in doc],
                           spaces=[bool(t.whitespace_) for t in doc]))
            # what the parser itself emitted, measured over ITS OWN sentence boundaries
            pred_np = set()
            for sent in pred.sents:
                pred_np.update(sent.start + i for i in nonproj_arcs(sent_local(sent.start, sent)))
            for g, p in zip(doc, pred):
                i = g.i
                bucket = ("np-arc" if i in np_arc else
                          "np-sent" if i in np_sent else "proj-sent")
                c[f"tok/{bucket}"] += 1
                ok = p.head.i == g.head.i
                c[f"err/{bucket}"] += not ok
                c[f"lerr/{bucket}"] += not (ok and p.dep_ == g.dep_)
                if i in pred_np:
                    c["pred-np"] += 1
                    c["pred-np-correct"] += ok
                    c["pred-np-was-gold-np"] += i in np_arc
                if i in np_arc:
                    c["gold-np-recovered-as-np"] += i in pred_np

        errs = sum(c[f"err/{b}"] for b in ("np-arc", "np-sent", "proj-sent"))
        print(f"\n== {name}  ({path})")
        print(f"   {errs} wrong attachments (UAS {100 * (1 - errs / n_tok):.2f})")
        print(f"   {'bucket':<11} {'tokens':>7} {'errors':>7} {'UAS':>7} {'LAS':>7} {'of all errors':>14}")
        for b, lbl in (("np-arc", "the arc"), ("np-sent", "its sentence"), ("proj-sent", "projective")):
            t, e, le = c[f"tok/{b}"], c[f"err/{b}"], c[f"lerr/{b}"]
            print(f"   {lbl:<11} {t:7d} {e:7d} {100 * (1 - e / t):7.2f} {100 * (1 - le / t):7.2f}"
                  f" {e / errs:13.2%}")
        np_all = c["err/np-arc"] + c["err/np-sent"]
        print(f"   in a non-projective sentence at all: {np_all} = {np_all / errs:.2%} of all errors")
        print(f"   parser emitted {c['pred-np']} non-projective arcs "
              f"({c['pred-np'] / max(n_np_arc, 1):.2f}x the gold count); "
              f"{c['pred-np-correct'] / max(c['pred-np'], 1):.1%} attached correctly, "
              f"{c['pred-np-was-gold-np'] / max(c['pred-np'], 1):.1%} were gold non-projective")
        print(f"   of the {n_np_arc} gold non-projective arcs, "
              f"{c['gold-np-recovered-as-np']} ({c['gold-np-recovered-as-np'] / max(n_np_arc, 1):.1%}) "
              f"came back non-projective at all")


if __name__ == "__main__":
    main()
