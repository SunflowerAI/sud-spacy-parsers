#!/usr/bin/env python3
"""Dump gold+predicted token records for the lzh depmorph arm, one JSON per split.

Predicted docs are built from GOLD WORDS (never re-tokenised), matching the harness the
error profile was taken with. Everything a runtime rule could read is dumped: form,
neighbour forms come from the sentence itself, predicted deprel/head, predicted xpos,
predicted morph, plus the gold UPOS/head/deprel to score against.
"""
import argparse
import importlib.util
import json
import pathlib
import sys


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="training_lzh_depmorph/model-best")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus",
                    default="corpus_lzh_zhipart/lzh_kyoto-sud-%s.relabeled_ext."
                            "udep_ruled.punct.rulemerged.zhipart.spacy")
    a = ap.parse_args()

    sys.path.insert(0, "scripts")
    import seg_code  # noqa: F401
    import spacy
    from spacy.tokens import Doc, DocBin

    corpus = a.corpus % a.split
    nlp = spacy.load(a.model)
    golds = list(DocBin().from_disk(corpus).get_docs(nlp.vocab))

    out = []
    for gi, gold in enumerate(golds):
        pred = Doc(nlp.vocab,
                   words=[t.text for t in gold],
                   spaces=[bool(t.whitespace_) for t in gold])
        pred = nlp(pred)
        assert len(pred) == len(gold)
        # sentence boundaries from GOLD, so rules can talk about "clause root"
        gsent = {}
        for si, s in enumerate(gold.sents):
            for t in s:
                gsent[t.i] = si
        psent = {}
        for si, s in enumerate(pred.sents):
            for t in s:
                psent[t.i] = si
        rec = []
        for i in range(len(gold)):
            g, p = gold[i], pred[i]
            rec.append({
                "form": g.text,
                "gu": g.pos_, "pu": p.pos_,
                "gh": g.head.i, "ph": p.head.i,
                "gd": g.dep_, "pd": p.dep_,
                "gx": g.tag_, "px": p.tag_,
                "pm": str(p.morph),
                "gs": gsent[i], "ps": psent.get(i, -1),
            })
        out.append(rec)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    n = sum(len(d) for d in out)
    print("%s: %d docs, %d tokens -> %s" % (a.split, len(out), n, a.out))


if __name__ == "__main__":
    main()
