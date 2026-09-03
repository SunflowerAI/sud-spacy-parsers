#!/usr/bin/env python3
"""Split a parser's errors into ATTACHMENT and LABELLING, before proposing any architecture.

WHY THIS IS THE GATE. Nine parser architectures were measured on zh/id in 2026-08 and **none beat
the shipped transition parser** — biaffine + CLE came in 8.71 LAS behind, and 3.52 behind even with
a BiLSTM equalising the encoder; coarsening the label set moved UAS by 0.01; a BiLSTM encoder is
worth **-0.29 to the transition parser** (it is +5.19 to biaffine, a clean 2x2). The one live lead
out of all of it was an ADDITIVE labeller over the existing parser, for a language whose errors are
labelling-bound: id had 8.14 % right-head-wrong-label, a third of its errors, while zh was
attachment-bound. The standing instruction is to run this decomposition per language — about a
minute — BEFORE building anything.

    right head, right label   correct
    right head, WRONG label   a labeller could fix this without touching attachment
    WRONG head                only a better parser or encoder can fix this

⚠ Scored under `--gold-preproc` semantics (gold tokens, one doc per gold sentence), which is the
regime every released figure uses, and `punct` is excluded exactly as `spacy evaluate` excludes it.
"""
import argparse
import collections
import importlib.util
import pathlib


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ignore", default="punct")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc, DocBin

    nlp = spacy.load(a.model)
    docs = list(DocBin().from_disk(a.corpus).get_docs(nlp.vocab))
    gold = [s.as_doc() for d in docs for s in d.sents]        # gold_preproc: one sentence per doc
    preds = [Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
             for g in gold]
    preds = list(nlp.pipe(preds, batch_size=64))

    n = ok = head_ok_lab_bad = head_bad = 0
    lab_conf = collections.Counter()
    head_bad_by_lab = collections.Counter()
    for p, g in zip(preds, gold):
        for tp, tg in zip(p, g):
            if tg.dep_ == a.ignore:
                continue
            n += 1
            if tp.head.i == tg.head.i:
                if tp.dep_ == tg.dep_:
                    ok += 1
                else:
                    head_ok_lab_bad += 1
                    lab_conf[(tg.dep_, tp.dep_)] += 1
            else:
                head_bad += 1
                head_bad_by_lab[tg.dep_] += 1
    err = n - ok
    print(f"{n} scored tokens ({a.ignore} excluded), {err} errors")
    print(f"   right head, right label   {ok:6d}  {ok/n:6.2%}   (= LAS)")
    print(f"   right head, WRONG label   {head_ok_lab_bad:6d}  {head_ok_lab_bad/n:6.2%}"
          f"   = {head_ok_lab_bad/max(err,1):6.1%} of errors  <- what a labeller could fix")
    print(f"   WRONG head                {head_bad:6d}  {head_bad/n:6.2%}"
          f"   = {head_bad/max(err,1):6.1%} of errors  <- needs a better parser/encoder")
    print("\n   top label confusions on a correctly attached token (gold -> predicted):")
    for (gd, pd), c in lab_conf.most_common(8):
        print(f"      {gd:<16} -> {pd:<16}{c:5d}")
    print("\n   gold labels most often MIS-ATTACHED:")
    for lab, c in head_bad_by_lab.most_common(6):
        print(f"      {lab:<16}{c:5d}")


if __name__ == "__main__":
    main()
