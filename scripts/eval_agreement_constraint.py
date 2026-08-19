#!/usr/bin/env python3
"""Does forbidding agreement-violating `mod` arcs actually improve the parse?

The detector is excellent — an `ADJ/participle --mod--> NOUN` arc breaking case/number/gender
agreement is a wrong attachment 95.1 % of the time. That is NOT the same as the constraint helping:
forbidding an arc does not say where the token should go instead, and the re-decode may attach it
somewhere equally wrong. This script measures the thing that matters rather than the thing that is
encouraging.

Reported per condition: UAS/LAS over all tokens, plus the constraint's own hit rate — how often it
fired, and how often the token it fired on ended up with the RIGHT head afterwards.

    eval_agreement_constraint.py MODEL TEST.spacy [--limit N]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from sud_constrained_parse import agreement_violation, parse_with_agreement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    parser = nlp.get_pipe("parser")
    blank = spacy.blank("sa")
    refs = list(DocBin().from_disk(a.test).get_docs(blank.vocab))
    sents = [s for d in refs for s in d.sents]
    if a.limit:
        sents = sents[: a.limit]

    def prep(s):
        d = Doc(nlp.vocab, words=[t.text for t in s], spaces=[bool(t.whitespace_) for t in s])
        for pt, rt in zip(d, s):
            pt.norm_ = rt.norm_
            if rt.morph.get("Compound"):
                pt.set_morph("Compound=Yes")
        for name, pipe in nlp.pipeline:
            if name == "parser":
                break
            d = pipe(d)
        return d

    tot = 0
    base = [0, 0]
    con = [0, 0]
    fired = fixed = broke = same = 0
    for s in sents:
        gold = [(t.head.i - s.start, t.dep_) for t in s]
        d = prep(s)
        b = parser(prep(s))
        # the morphologiser has not run on `d`; the constraint needs FEATS, so annotate a copy
        md = nlp.get_pipe("morphologizer")(prep(s))
        # POS as well as MORPH: `agreement_violation` tests `head.pos_ == "NOUN"` and the child's
        # ADJ/VerbForm, so copying FEATS alone leaves every pos_ empty and the predicate never
        # fires — it reported 0 hits on 3 640 tokens, which is how this was caught.
        for tok, m in zip(b, md):
            tok.pos_ = m.pos_
            comp = tok.morph.get("Compound")
            tok.set_morph(str(m.morph))
            if comp and not tok.morph.get("Compound"):
                tok.set_morph((str(tok.morph) + "|Compound=Yes").lstrip("|"))
        hits = [i for i, t in enumerate(b)
                if t.dep_ == "mod" and t.head.i != i and agreement_violation(t, b[t.head.i])]
        c = b
        if hits:
            fired += len(hits)
            o = parse_with_agreement(parser, md)
            if o is not None:
                c = o
        for i, (gh, gl) in enumerate(gold):
            tot += 1
            base[0] += b[i].head.i == gh
            base[1] += b[i].head.i == gh and b[i].dep_ == gl
            con[0] += c[i].head.i == gh
            con[1] += c[i].head.i == gh and c[i].dep_ == gl
        for i in hits:
            was, now = b[i].head.i == gold[i][0], c[i].head.i == gold[i][0]
            fixed += (not was) and now
            broke += was and (not now)
            same += was == now
    print(f"{len(sents)} sentences / {tot} tokens; constraint fired on {fired} arcs")
    print(f"  baseline    UAS {base[0]/tot:.4f}  LAS {base[1]/tot:.4f}")
    print(f"  constrained UAS {con[0]/tot:.4f}  LAS {con[1]/tot:.4f}"
          f"   ({100*(con[1]-base[1])/tot:+.2f} LAS)")
    print(f"  on the arcs it fired on: fixed {fixed}, broke {broke}, unchanged {same}")


if __name__ == "__main__":
    main()
