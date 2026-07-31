#!/usr/bin/env python3
"""Parse Latin and emit the dependency analysis WITH macronised word forms.

`la_macronise` attaches macrons as extensions (`token._.macron`), leaving `token.text` alone --
spaCy tokens are immutable, so the component cannot rewrite the surface in place. This script is
the last mile: it joins the parse and the macrons into one output.

Two modes, and the difference matters:

  --mode attach   (default)  parse the plain text once, then print the parse against the
                             MACRONISED forms. The parse is the model's analysis of the plain
                             text; macrons are a display layer over it. One pass, and the parse
                             is exactly what you would get without macronisation.

  --mode reparse             macronise, then run the pipeline AGAIN over the macronised text, so
                             `token.text` itself carries macrons and the parse is the model's
                             analysis OF the macronised string. Legitimate because the released
                             Latin model is trained on the union of plain and macronised data.

MEASURED (scripts/eval_la_reparse.py, 4300 test sentences, gold tokens): re-parsing costs nothing
in aggregate -- LAS 70.30 on the plain forms, 70.33 on our own macronised forms, 70.31 on the
Alatius ones. A spread of 0.04 is noise. Individual sentences DO move (a wrong macron is a false
Case cue: `Galliā` for nominative `Gallia` flips it PROPN/comp:pred -> ADJ/subj), but the shifts
cancel rather than accumulate. So prefer `attach` for being one pass and for guaranteeing the parse
is byte-identical to the un-macronised one; reach for `reparse` only when you need `token.text`
itself to carry the macrons.

Output formats: `conllu` (FORM column macronised) or `table` (human-readable).

    la_parse_macronised.py build_la_macron/model --text "Gallia est omnis divisa in partes tres."
    la_parse_macronised.py build_la_macron/model --file in.txt --format conllu > out.conllu
"""
import argparse
import importlib.util
import sys

import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(path.split("/")[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conllu(doc, forms):
    """CoNLL-U with the macronised surface in FORM. HEAD is 1-based within the sentence."""
    out = []
    for sent in doc.sents:
        start = sent.start
        out.append(f"# text = {''.join(forms[t.i] + t.whitespace_ for t in sent).strip()}")
        for tok in sent:
            head = 0 if tok.head.i == tok.i else tok.head.i - start + 1
            dep = "root" if tok.head.i == tok.i else tok.dep_
            out.append("\t".join([
                str(tok.i - start + 1), forms[tok.i], tok.lemma_ or "_",
                tok.pos_ or "_", tok.tag_ or "_", str(tok.morph) or "_",
                str(head), dep, "_",
                "_" if tok.whitespace_ else "SpaceAfter=No",
            ]))
        out.append("")
    return "\n".join(out)


def table(doc, forms):
    rows = [f"{'ID':>3}  {'FORM (macronised)':22} {'LEMMA':14} {'UPOS':6} {'HEAD':>4}  DEPREL"]
    for sent in doc.sents:
        start = sent.start
        for tok in sent:
            head = 0 if tok.head.i == tok.i else tok.head.i - start + 1
            dep = "root" if tok.head.i == tok.i else tok.dep_
            rows.append(f"{tok.i - start + 1:>3}  {forms[tok.i]:22} {tok.lemma_:14} "
                        f"{tok.pos_:6} {head:>4}  {dep}")
        rows.append("")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--mode", choices=("attach", "reparse"), default="attach")
    ap.add_argument("--format", choices=("table", "conllu"), default="table")
    ap.add_argument("--code", default="scripts/la_macronise.py")
    args = ap.parse_args()

    load_code(args.code)
    nlp = spacy.load(args.model)
    if "la_macronise" not in nlp.pipe_names:
        sys.exit("model has no la_macronise pipe -- run scripts/build_la_macron.sh first")

    text = args.text if args.text else open(args.file, encoding="utf-8").read()
    doc = nlp(text)
    if args.mode == "reparse":
        # rebuild the input from the macronised forms and parse THAT, so token.text carries macrons
        doc = nlp("".join(t._.macron + t.whitespace_ for t in doc))
        forms = [t.text for t in doc]
    else:
        forms = [t._.macron for t in doc]

    print((conllu if args.format == "conllu" else table)(doc, forms))


if __name__ == "__main__":
    main()
