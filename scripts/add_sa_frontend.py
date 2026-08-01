#!/usr/bin/env python
"""Assemble the shippable Sanskrit model: raw IAST/Devanagari in, annotated tokens out.

The trained arm knows nothing about input handling — it is `[tok2vec, tagger, parser,
morphologizer, lemmatizer]` over CSL-derived tokens. This bolts on everything needed to take real
text, in the order that matters:

    tokenizer  sa.SanskritInputTokenizer.v3 with two TRAINED models attached
                 stage 0  CSLise      raw text -> CSL          (sa_presegment)
                 stage 1  de-CSLise   CSL -> tokens + Compound (mechanical)
                 stage 2  de-sandhi   MWT members -> unsandhied (sud_unsandhi transducer)
    sa_compound  FIRST — re-derives Compound when a caller passes TOKENS, so the encoder's MORPH
                 input is present however the Doc was built
    ... the trained components ...
    clause_parser  per-sentence re-parse + punctuation morphology on raw multi-clause input
    sa_deva        LAST — Devanagari FORM/LEMMA + Translit/LTranslit, only if the input was
                   Devanagari. Must follow clause_parser, which also rebuilds the doc.

CSL is now purely an INTERNAL representation: no caller ever has to produce it.

    add_sa_frontend.py IN_MODEL OUT_MODEL --csliser models/sa_presegment_ortho \\
        --unsandhi training_sa_mwt_unsandhi/model-best
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
import sa_devanagari                             # noqa: E402,F401
from thinc.api import Config                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--csliser", required=True)
    ap.add_argument("--unsandhi", required=True,
                    help="trained arm holding the sandhi-reversal transducer")
    a = ap.parse_args()

    nlp = spacy.load(a.in_model)

    # --- the two trained front-end models, into the tokenizer -------------------------------
    uns = pathlib.Path(a.unsandhi)
    comp_dir = uns / "desandhi_component"
    if not comp_dir.exists():                    # export the transducer out of its training arm
        spacy.load(uns).get_pipe("lemmatizer").to_disk(comp_dir)
    model_cfg = Config().from_disk(uns / "config.cfg",
                                   interpolate=True)["components"]["lemmatizer"]["model"]
    tok = sa_tokenizer.SanskritInputTokenizer(nlp.vocab, split_only=True, cslise=True)
    tok.load_desandhi(comp_dir, model_cfg)
    tok.load_csliser(a.csliser)
    nlp.tokenizer = tok
    # the config must name v3, or a reloaded model builds a v1 tokenizer and expects CSL input
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sa.SanskritInputTokenizer.v3"}

    # --- the non-trainable pipes, in their required positions -------------------------------
    for name, kwargs, cfg in (
        ("sa_compound", {"first": True}, {}),
        ("clause_parser", {"last": True}, {"punct_tag": "PUNCT", "sent_scheme": "danda"}),
        ("sa_deva", {"last": True}, {}),
    ):
        if name in nlp.pipe_names:
            nlp.remove_pipe(name)                # rebuild picks up new code
        nlp.add_pipe(name, config=cfg, **kwargs)

    nlp.to_disk(a.out_model)
    print(f"{a.in_model} -> {a.out_model}")
    print(f"  pipeline : {nlp.pipe_names}")
    print(f"  tokenizer: {nlp.config['nlp']['tokenizer']['@tokenizers']}"
          f"  (CSLiser {tok.csliser is not None}, de-sandhifier {tok.desandhi_model is not None})")


if __name__ == "__main__":
    main()
