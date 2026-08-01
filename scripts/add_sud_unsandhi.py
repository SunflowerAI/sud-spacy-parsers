#!/usr/bin/env python3
"""Append the trained sandhi-reversal component to a Sanskrit pipeline, as `sud_unsandhi`.

The component is TRAINED as a stock `trainable_lemmatizer` (on a corpus whose LEMMA column holds
the `Unsandhied` value — see `scripts/make_unsandhi_corpus.py`, and why in `scripts/sud_unsandhi.py`)
and is re-homed here into `SudUnsandhi`, which writes `Token._.unsandhied` instead of
`token.lemma_`. The two edit-tree components then coexist without colliding: `lemmatizer` produces
lemmas, `sud_unsandhi` produces padapāṭha forms.

The model architecture is read out of the trained arm's own config, so an arm trained with the
affix embed loads just as well as one trained with the stock embed.

    add_sud_unsandhi.py IN_MODEL OUT_MODEL --unsandhi training_sa_mwt_unsandhi/model-best
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                     # noqa: E402
import sa_tokenizer                              # noqa: E402,F401
import clause_parser                             # noqa: E402,F401
import sud_affix_embed                           # noqa: E402,F401
import sud_unsandhi                              # noqa: E402,F401
from spacy.tokens import DocBin                  # noqa: E402
from spacy.training import Example               # noqa: E402
from thinc.api import Config                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--unsandhi", required=True, help="trained arm holding the transducer")
    ap.add_argument("--sample", default="corpus_sa_unsandhi/sa_vedic-sud-dev.unsandhi.spacy",
                    help="a few docs, only to shape the model at initialize()")
    ap.add_argument("--last", action="store_true",
                    help="append at the very end (default: before clause_parser, which rebuilds "
                         "the doc and would drop the extension)")
    a = ap.parse_args()

    nlp = spacy.load(a.inp)
    src = spacy.load(a.unsandhi).get_pipe("lemmatizer")
    model_cfg = Config().from_disk(pathlib.Path(a.unsandhi) / "config.cfg",
                                   interpolate=True)["components"]["lemmatizer"]["model"]

    docs = list(DocBin().from_disk(a.sample).get_docs(nlp.vocab))[:5]
    sample = [Example(d.copy(), d) for d in docs]

    kwargs = {}
    if not a.last and "clause_parser" in nlp.pipe_names:
        kwargs["before"] = "clause_parser"
    comp = nlp.add_pipe("sud_unsandhi", config={"model": model_cfg}, **kwargs)
    comp.initialize(lambda: sample, labels=src.label_data)
    comp.from_bytes(src.to_bytes())

    nlp.to_disk(a.out)
    print(f"{a.inp} + sud_unsandhi -> {a.out}")
    print(f"  pipeline: {nlp.pipe_names}")


if __name__ == "__main__":
    main()
