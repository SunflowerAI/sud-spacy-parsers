#!/usr/bin/env python3
"""Build the corpus for joint multi-task training: ONE shared encoder, four objectives.

The point of the joint arm is a single `tok2vec` serving tagger + parser + morphologizer +
lemmatizer, so the wheel ships one encoder instead of three. That needs a corpus carrying every
annotation each component wants — but the two halves have different coverage:

    Vedic + UFAL   21 707 sentences   syntax + tag + morph + lemma
    DCS           244 481 sentences   tag + morph + lemma, NO syntax

**The trap.** DCS's CoNLL-U does not leave HEAD/DEPREL empty — it carries a FABRICATED flat tree
(`HEAD=1 / DEPREL=dep` on 1 488 394 tokens, `HEAD=0 / root` on the 244 481 sentence-initial ones).
Trained on directly, a parser learns to emit that star shape. Blanking the columns to `_` does NOT
fix it either: `spacy convert` turns `_` into the LITERAL dep label `_`, with
`has_annotation("DEP") == True` — the same class of bug as the `_` lemma that once taught
`sud_unsandhi` to predict a literal underscore.

The only representation spaCy reads as genuinely missing is a Doc built with **no heads/deps at
all**, which is what this script produces: DCS docs keep tag/pos/morph/lemma and drop syntax
entirely, so the parser takes no gradient from them while the other three components take all of
theirs. Verified by asserting `has_annotation("DEP")` is False on every DCS doc and True on every
Vedic/UFAL one.

    make_sa_multitask_corpus.py --out corpus_sa_multitask
"""
import argparse
import pathlib

import spacy
from spacy.tokens import Doc, DocBin


def strip_syntax(doc, vocab):
    """Rebuild a doc keeping every annotation EXCEPT head/dep."""
    out = Doc(
        vocab,
        words=[t.text for t in doc],
        spaces=[bool(t.whitespace_) for t in doc],
        tags=[t.tag_ for t in doc],
        pos=[t.pos_ for t in doc],
        morphs=[str(t.morph) for t in doc],
        lemmas=[t.lemma_ for t in doc],
        # NORM carries the padapāṭha that sud.AnalyserFeatsEmbed.v1 keys its lookup on. Omitting it
        # here does not fail: the DCS half is 92 % of the joint arm's tokens, so the analyser channel
        # would read "silent" on almost every training token while being populated at inference —
        # the model would learn to ignore a feature that then appears from nowhere. Standing rule:
        # anything that rebuilds a Doc owns carrying EVERY annotation.
    )
    # NORM is NOT a Doc() keyword (TypeError: unexpected keyword argument 'norms') — it is a
    # per-token attribute set afterwards. A rebuild that quietly leaves it at the lexeme default
    # looks identical in every string comparison.
    for src, dst in zip(doc, out):
        dst.norm_ = src.norm_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syntax", default="corpus_sa_split/train.spacy",
                    help="Vedic + UFAL-train: full annotation, keep as is")
    ap.add_argument("--full", default="corpus_sa_dcs_split/train.spacy",
                    help="the same PLUS DCS; its tail is the syntax-free half")
    ap.add_argument("--dev", default="corpus_sa_split/dev.spacy")
    ap.add_argument("--out", default="corpus_sa_multitask")
    ap.add_argument("--dcs-from", type=int, default=None,
                    help="index in --full where the syntax-free DCS tail begins. Defaults to "
                         "len(syntax), which is only right when --full literally starts with the "
                         "SAME syntax corpus. A relabelled syntax half has a different doc count "
                         "(2171 vs 2165), so the default would silently slice 6 docs off the DCS "
                         "tail and treat 6 syntax docs as DCS — pass it explicitly.")
    a = ap.parse_args()

    nlp = spacy.blank("xx")
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)

    syn = list(DocBin().from_disk(a.syntax).get_docs(nlp.vocab))
    full = list(DocBin().from_disk(a.full).get_docs(nlp.vocab))
    off = a.dcs_from if a.dcs_from is not None else len(syn)
    dcs = full[off:]
    assert len(dcs) == len(full) - off
    print(f"syntax {len(syn)} docs | DCS tail starts at {off} -> {len(dcs)} docs")

    db = DocBin(store_user_data=True)
    for d in syn:
        assert d.has_annotation("DEP"), "a syntax doc lost its parse"
        db.add(d)
    n_bad = 0
    for d in dcs:
        s = strip_syntax(d, nlp.vocab)
        if s.has_annotation("DEP"):
            n_bad += 1
        db.add(s)
    assert n_bad == 0, f"{n_bad} DCS docs still carry DEP annotation"
    db.to_disk(out / "train.spacy")

    DocBin(store_user_data=True, docs=list(
        DocBin().from_disk(a.dev).get_docs(nlp.vocab))).to_disk(out / "dev.spacy")

    tok_syn = sum(len(d) for d in syn)
    tok_dcs = sum(len(d) for d in dcs)
    print(f"  {out}/train.spacy")
    print(f"    with syntax : {len(syn):6d} docs  {tok_syn:8d} tokens  (parser trains on these)")
    print(f"    no syntax   : {len(dcs):6d} docs  {tok_dcs:8d} tokens  (tag/morph/lemma only)")
    print(f"    parser sees {tok_syn / (tok_syn + tok_dcs):.1%} of tokens")


if __name__ == "__main__":
    main()
