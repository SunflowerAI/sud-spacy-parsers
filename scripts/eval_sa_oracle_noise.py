#!/usr/bin/env python3
"""How much of the ORACLE arm's gain survives being fed by real components?

`training_sa_mwt_oracle` reads a LEMMA column and eleven per-FEATURE columns that
`sud.OracleCorpus.v1` fills from the treebank. The ARCHITECTURE is shippable — it is an ordinary
embed layer. The INPUT is not: at inference those columns can only be filled by the morphologiser
(morph_acc 0.78) and the lemmatiser (lemma_acc 0.88), and in the released arm both run AFTER the
parser. This script substitutes their predictions for the gold, one channel at a time, and scores
the same trained parser.

⚠ WHAT THIS IS AND IS NOT. The arm was TRAINED on gold, so it has learned to trust a channel that is
about to get noisier — the classic train/inference skew, and the reason `config_sa_morphfirst.cfg`
trains through `annotating_components` instead. So these numbers are a LOWER bound on what a
properly trained predicted-input arm reaches, not an estimate of it. They are worth taking first
because they cost one evaluation rather than one training run, and because a channel that collapses
here is one whose noise-sensitivity dominates everything else about it.

`Compound` is re-imposed from the reference after the morphologiser runs, because the morphologiser
OVERWRITES token.morph and would otherwise destroy a feature the tokeniser supplies at P/R 0.9998 —
the same fix `clause_parser` applies in the released pipeline.

⚠ THE EXAMPLES COME FROM `sud.OracleCorpus.v1` ITSELF, not from a hand-rolled loop over the DocBin.
Under `gold_preproc` spaCy yields ONE SENTENCE per example, which is the regime this arm was trained
and scored in; feeding it whole ten-sentence docs instead makes it find its own boundaries and cost
this measurement 16 LAS the first time it was written. Reuse the reader, do not reimplement it.

    eval_sa_oracle_noise.py <parser-arm> <test.spacy> [--annotator DIR]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401

import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from gold_tok_corpus import OracleCorpus  # noqa: E402


def predictions(ann, egs):
    """Predicted (lemma, morph-dict) per token, in the annotator's own regime: gold words,
    one sentence at a time, tokeniser-supplied Compound."""
    units = []
    for eg in egs:
        d = Doc(ann.vocab, words=[t.text for t in eg.reference],
                spaces=[bool(t.whitespace_) for t in eg.reference])
        for pt, rt in zip(d, eg.reference):
            if rt.morph.get("Compound"):
                pt.set_morph("Compound=Yes")
        units.append(d)
    return [[(t.lemma_, t.morph.to_dict()) for t in d]
            for d in ann.pipe(units, batch_size=64)]


def restamp(egs, pred, lemma_src, morph_src):
    """Overwrite each predicted doc's LEMMA/FEATS in place. The reader already put GOLD in both, so
    a 'gold' source is a no-op and the two channels vary independently."""
    for eg, pr in zip(egs, pred):
        for tok, rt, (plem, pmorph) in zip(eg.predicted, eg.reference, pr):
            if lemma_src == "pred":
                tok.lemma_ = plem or rt.text
            if morph_src == "pred":
                m = dict(pmorph)
                # the tokeniser's verdict outranks the morphologiser's, as in the released pipeline
                m.pop("Compound", None)
                if rt.morph.get("Compound"):
                    m["Compound"] = "Yes"
                # set_morph("") stores the EMPTY morph (key 456) where untouched is UNSET (key 0);
                # the reader guarantees every token already carries gold, so overwrite or leave be
                if m:
                    tok.set_morph("|".join(f"{k}={v}" for k, v in sorted(m.items())))
    return egs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("test")
    ap.add_argument("--annotator", default="training_sa_mwt_lemma_sfx5/model-best")
    a = ap.parse_args()

    nlp = spacy.load(a.arm)
    ann = spacy.load(a.annotator)
    base = list(OracleCorpus(a.test, gold_preproc=True)(nlp))
    pred = predictions(ann, base)

    n = okl = okm = 0
    for eg, pr in zip(base, pred):
        for rt, (plem, pmorph) in zip(eg.reference, pr):
            n += 1
            okl += (plem == rt.lemma_)
            okm += (pmorph == rt.morph.to_dict())
    print(f"annotator {a.annotator}: lemma {okl/n:.4f}  FEATS(exact bundle) {okm/n:.4f}  (n={n})\n")

    print(f"{'LEMMA':<8}{'FEATS':<8}{'UAS':>8}{'LAS':>8}")
    for ls in ("gold", "pred"):
        for ms in ("gold", "pred"):
            egs = restamp(list(OracleCorpus(a.test, gold_preproc=True)(nlp)), pred, ls, ms)
            sc = nlp.evaluate(egs)
            print(f"{ls:<8}{ms:<8}{sc['dep_uas']:>8.4f}{sc['dep_las']:>8.4f}")


if __name__ == "__main__":
    main()
