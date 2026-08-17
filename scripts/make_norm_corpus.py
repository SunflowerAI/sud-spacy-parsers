#!/usr/bin/env python3
"""Write the sandhi-reversed (padapāṭha) form into NORM, so the parser's lexical channel stops
paying for sandhi.

WHY. Under the DCS representation a STANDALONE token keeps its sandhied surface, which is what took
`corpus_sa_csl_mwt` to 38 883 form types / 61.6 % hapax (against 12 024 lemma types / 44.2 %). The
parser's `NORM` column therefore spends most of its capacity on alternations that carry no syntax.
`sud_unsandhi` already undoes exactly those alternations, at 0.964 on the Vedic test — and, unlike a
lemma, it is available BEFORE any component runs, because the released frontend carries the
transducer inside the TOKENISER (`sa_tokenizer.py` stage 2). So this needs no pipeline reordering:
only the string in `NORM`.

PREFIX / SUFFIX / SHAPE are lexeme attributes computed from the ORTH, so the sandhied surface stays
visible to the model through those three columns. This swaps the identity channel, it does not throw
the form away.

PREDICTED, NOT GOLD. The treebank has `Unsandhied=` on 98.9 % of tokens and using it would be an
oracle — the deployed path will have the transducer's output, so that is what the corpus gets. The
residual dishonesty is measured and printed: the transducer was trained on this very training split,
so its accuracy there is higher than on held-out text, and the parser therefore trains on a slightly
cleaner NORM than it will meet at inference. `--report` prints both numbers so the gap is on the
record rather than assumed away.

⚠ ASK THE MODEL ITS INPUT REGIME. The transducer was trained through `sud.CompoundCorpus.v1` with
`gold_preproc = true`, i.e. on SINGLE SENTENCES with `Compound=Yes` supplied. Both are reproduced
here, and `--report` scores the whole-doc regime too so the choice is measured rather than assumed
(this is the `_cslise` bug's shape: an input regime that looked obviously equivalent and cost 4.83 F).

    make_norm_corpus.py --transducer training_sa_mwt_unsandhi/model-best \
        --in corpus_sa_csl_mwt --out corpus_sa_mwt_norm [--report]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401  (registers the sa tokenizer, readers and custom layers)

import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402


def _predict(nlp, refs, by_sentence=True):
    """Run the transducer over gold-word copies of `refs`; return one list of strings per doc."""
    units, owner = [], []
    for di, r in enumerate(refs):
        spans = list(r.sents) if by_sentence and r.has_annotation("SENT_START") else [r[:]]
        for sp in spans:
            d = Doc(nlp.vocab, words=[t.text for t in sp],
                    spaces=[bool(t.whitespace_) for t in sp])
            # the tokeniser supplies this at inference (sa_tokenizer.py: "the transducer's encoder
            # reads MORPH"), so training-time input must carry it too
            for pt, rt in zip(d, sp):
                if rt.morph.get("Compound"):
                    pt.set_morph("Compound=Yes")
            units.append(d)
            owner.append(di)
    out = [[] for _ in refs]
    for di, d in zip(owner, nlp.pipe(units, batch_size=64)):
        out[di].extend(t.lemma_ for t in d)
    for r, got in zip(refs, out):
        assert len(got) == len(r), f"length skew: {len(got)} vs {len(r)}"
    return out


def _score(refs, preds):
    """Exact-match against gold `Unsandhied`, which `make_unsandhi_corpus.py` parked in LEMMA."""
    n = ok = 0
    for r, got in zip(refs, preds):
        for t, g in zip(r, got):
            gold = t.lemma_
            if not gold or gold == "_":
                continue
            n += 1
            ok += (g == gold)
    return ok / n if n else float("nan"), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transducer", default="training_sa_mwt_unsandhi/model-best")
    ap.add_argument("--in", dest="src", default="corpus_sa_csl_mwt")
    ap.add_argument("--out", default="corpus_sa_mwt_norm")
    ap.add_argument("--report", action="store_true",
                    help="also score the transducer against gold Unsandhied, per split and regime")
    a = ap.parse_args()

    nlp = spacy.load(a.transducer)
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(exist_ok=True)

    # gold Unsandhied lives in a parallel corpus (LEMMA column), keyed by the same split
    gold_dir = pathlib.Path("corpus_sa_unsandhi")
    gold_for = {"train.csl_mwt.spacy": "train.unsandhi.spacy",
                "sa_vedic-sud-dev.csl_mwt.spacy": "sa_vedic-sud-dev.unsandhi.spacy",
                "sa_vedic-sud-test.csl_mwt.spacy": "sa_vedic-sud-test.unsandhi.spacy"}

    for f in sorted(src.glob("*.spacy")):
        refs = list(DocBin().from_disk(f).get_docs(nlp.vocab))
        preds = _predict(nlp, refs, by_sentence=True)
        changed = same = 0
        for r, got in zip(refs, preds):
            for t, g in zip(r, got):
                if g and g != t.text:
                    t.norm_ = g
                    changed += 1
                else:
                    same += 1
        db = DocBin(docs=refs, store_user_data=True)
        db.to_disk(out / f.name)
        print(f"{f.name}: {len(refs)} docs, NORM rewritten on {changed}/{changed+same} "
              f"({changed/(changed+same):.1%}) tokens -> {out / f.name}")

        if a.report and f.name in gold_for:
            gpath = gold_dir / gold_for[f.name]
            if not gpath.exists():
                continue
            grefs = list(DocBin().from_disk(gpath).get_docs(nlp.vocab))
            if len(grefs) != len(refs):
                print(f"    (skipped scoring: {gpath.name} has {len(grefs)} docs vs {len(refs)})")
                continue
            acc_s, n = _score(grefs, preds)
            acc_d, _ = _score(grefs, _predict(nlp, refs, by_sentence=False))
            print(f"    transducer exact-match: by sentence {acc_s:.4f} | whole doc {acc_d:.4f}"
                  f"  (n={n})")


if __name__ == "__main__":
    main()
