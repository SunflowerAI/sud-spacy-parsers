#!/usr/bin/env python
"""`han_lemma_lut`: lemmas by lookup, for the Han-script arms where lemma ≈ form.

In zh / yue / lzh the lemma equals the form for 97.9–99.3 % of tokens, and the exceptions are not
morphology at all — they are **variant-character normalisation** (異體字): 為→爲, 抜→拔, 呪→咒,
処→處, 教→敎. There is no edit rule to learn there, only a list, and the list is small: 163 entries
for lzh, 141 for zh, 17 for yue.

Measured on test, against the trained `trainable_lemmatizer` those arms currently ship:

    arm    n        identity   table    trained
    lzh    68 466   99.016     99.733   99.649    <- table wins by ~57 tokens
    zh     24 020   99.334     99.900   99.904    <- 1 token apart
    yue     1 261   97.938     99.762   99.841    <- 1 token apart

So for **lzh the table is genuinely better**, and for zh/yue it is a tie — the argument there is the
~1 MB per wheel and one fewer training step per base change, not accuracy. The trained layer's
errors are exactly the variants it leaves untouched (predicting 抜 for 拔), i.e. it defaults to
identity on the cases a table has memorised, so its notional advantage — generalising to unseen
variants — is not one it actually delivers. It also mis-fires across scripts on the both-script
arms, emitting the simplified 举 where gold wants 舉; a form-keyed table cannot do that.

⚠ **Do not read this as "trained lemmatisers are pointless".** It holds where lemma ≈ form. The
same component would be useless for ar/fa/la/sa, whose lemma_acc rests on real morphological edits.

Usage:
    han_lemma_lut.py --build IN_MODEL OUT_MODEL --conllu TRAIN.conllu [--keep-lemmatizer]
"""
import argparse
import json
import pathlib

from spacy.language import Language
from spacy.tokens import Doc  # noqa: F401  (imported for parity with the other bundled modules)


def make_han_lemma_lut(nlp, name):
    return HanLemmaLut()


# Guard registration: several wheels bundle this module, so it can be imported more than once in
# one process (the same reason clause_parser guards its factory).
if not Language.has_factory("han_lemma_lut"):
    Language.factory("han_lemma_lut")(make_han_lemma_lut)


class HanLemmaLut:
    """Sets `token.lemma_` from a form→lemma table, falling back to the form itself.

    The table travels INSIDE the component's serialised directory, so a wheel is self-contained and
    the component degrades to pure identity if the file is ever missing rather than raising — the
    same reasoning as `la_macronise`'s `require_data=False`: this sits in the default pipeline, so
    raising would break every ordinary `nlp(text)`.
    """

    def __init__(self, table=None):
        self.table = dict(table or {})

    def __call__(self, doc):
        for tok in doc:
            tok.lemma_ = self.table.get(tok.text, tok.text)
        return doc

    def to_disk(self, path, exclude=tuple()):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "table.json").write_text(
            json.dumps(self.table, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def from_disk(self, path, exclude=tuple()):
        f = pathlib.Path(path) / "table.json"
        if f.exists():
            self.table = json.loads(f.read_text(encoding="utf-8"))
        return self


def harvest(conllu):
    """form → lemma, keeping only entries where they DIFFER (identity is the fallback).

    Majority wins on the rare ambiguous form (92 of 12 137 lzh types take more than one lemma), and
    a form whose majority lemma IS itself is left out of the table entirely — that keeps the table
    to the exceptions and makes it readable.
    """
    import collections
    obs = collections.defaultdict(collections.Counter)
    for line in open(conllu, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 3 or "-" in f[0] or "." in f[0]:
            continue
        obs[f[1]][f[2]] += 1
    return {form: c.most_common(1)[0][0] for form, c in obs.items()
            if c.most_common(1)[0][0] != form}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs=2, metavar=("IN_MODEL", "OUT_MODEL"), required=True)
    ap.add_argument("--conllu", required=True, help="treebank to harvest the table from (train)")
    ap.add_argument("--keep-lemmatizer", action="store_true",
                    help="leave the trained lemmatizer in place (for A/B rather than replacement)")
    args = ap.parse_args()

    import importlib.util
    import spacy
    # register the custom languages/tokenisers (lzh has no `spacy.lang.lzh`), as add_clause_parser
    # does — without this `spacy.load` on an lzh or yue arm raises E048.
    _p = pathlib.Path(__file__).with_name("seg_code.py")
    if _p.exists():
        _s = importlib.util.spec_from_file_location("seg_code", _p)
        _s.loader.exec_module(importlib.util.module_from_spec(_s))
    src, dst = args.build
    table = harvest(args.conllu)
    nlp = spacy.load(src)
    if "lemmatizer" in nlp.pipe_names and not args.keep_lemmatizer:
        nlp.remove_pipe("lemmatizer")
    if "han_lemma_lut" in nlp.pipe_names:
        nlp.remove_pipe("han_lemma_lut")
    # after the morphologizer, before any clause_parser / sud_* pipe that rebuilds the Doc
    pipe = nlp.add_pipe("han_lemma_lut")
    pipe.table = table
    nlp.to_disk(dst)
    print(f"{dst}: {len(table)} table entries, pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
