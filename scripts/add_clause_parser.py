#!/usr/bin/env python
"""Append the (non-trainable) clause_parser pipe to a trained model and save it.

clause_parser splits punctuated input at clause boundaries, parses each 句讀 unit
in isolation and reattaches the marks as punct — needed for raw multi-clause
inference on Kyoto/Vedic editions (which carry no in-text sentence boundaries).
It is added after training because it has no learned weights.

Usage:
    add_clause_parser.py IN_MODEL OUT_MODEL [--punct-tag PUNCT]

--punct-tag "" (default) uses the Kyoto 記号 subtype map (lzh); pass "PUNCT" for
Sanskrit (daṇḍa not stamped with the Japanese tagset).

--sent-punct defaults to the genuinely sentence-final marks (。．.！？!?…।॥, optionally followed by
a run of closing brackets / quotation marks), so a sentence-medial pause — a comma, a semicolon, a
bracket — is still pulled out for parsing but does NOT break the sentence. Pass "" for the old
behaviour, where EVERY mark is a boundary and each 句讀 unit is its own sentence. Sanskrit ignores
this and passes --sent-scheme danda instead.
"""
import argparse
import importlib.util

import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(path.split("/")[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--punct-tag", default="")
    # None (not passed) leaves the factory default — the sentence-final marks. Passing "" is
    # meaningful in its own right (every mark becomes a boundary), so it can't be the default here.
    ap.add_argument("--sent-punct", default=None)
    # "" (default) uses the fixed --sent-punct set (lzh). "danda" uses the document-dependent
    # Sanskrit scheme: ?/! always end a sentence, then periods / double daṇḍas / single daṇḍas by
    # what the text contains, with a trailing quotation mark kept on the closing sentence.
    ap.add_argument("--sent-scheme", default="")
    # Hand the sentence-medial marks to the parser instead of stripping and reattaching them by
    # rule. COUPLED TO THE ARM: worth +2.34 LAS on one trained on a punctuated treebank, and -3.80
    # on one that has never seen a mark. Never set it on an arm that was not trained with punctuation.
    ap.add_argument("--keep-marks", action="store_true")
    args = ap.parse_args()

    # register every custom tokenizer (lzh char, sa CSL, …) + the clause_parser factory before
    # loading, so this works for both the Classical Chinese and Sanskrit models.
    load_code("scripts/seg_code.py")

    nlp = spacy.load(args.in_model)
    # replace any existing clause_parser so the new code + config (sent_punct) take effect — the
    # model may already carry one from an earlier release built with the old factory.
    if "clause_parser" in nlp.pipe_names:
        nlp.remove_pipe("clause_parser")
    config = {"punct_tag": args.punct_tag, "sent_scheme": args.sent_scheme,
              "keep_marks": args.keep_marks}
    if args.sent_punct is not None:
        config["sent_punct"] = args.sent_punct
    # clause_parser MUST come before every `sud_*` pipe. It reassigns every head and deprel, so a
    # pipe that reads the tree -- `sud_shared`'s coordination mask, `sud_idiom`'s `unk` chains,
    # the subject/reported rules -- has to see the tree it leaves behind rather than the whole-doc
    # parse it discards; and it rebuilds the Doc, which can drop annotation written before it.
    #
    # Appending used to be enough because the arm being wrapped had no sud pipes yet, and the
    # invariant held by accident of ordering. It stopped holding the moment lzh started taking a
    # TRAINED arm as its base (for `sud_shared`) -- the pipeline came out with sud_shared first,
    # which builds and loads and says nothing. Position it explicitly instead.
    sud_pipes = [n for n in nlp.pipe_names if n.startswith("sud_")]
    if sud_pipes:
        nlp.add_pipe("clause_parser", config=config, before=sud_pipes[0])
    else:
        nlp.add_pipe("clause_parser", config=config)
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
