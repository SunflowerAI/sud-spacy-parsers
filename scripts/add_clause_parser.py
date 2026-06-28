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

--sent-punct "" (default) makes every punctuation mark a sentence boundary (lzh: each 句讀
unit is its own sentence). Sanskrit passes the daṇḍa-class marks ("।॥|/.?!…") so that a
sentence-medial comma is pulled out for isolated parsing but does NOT break the sentence.
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
    ap.add_argument("--sent-punct", default="")
    args = ap.parse_args()

    # register every custom tokenizer (lzh char, sa CSL, …) + the clause_parser factory before
    # loading, so this works for both the Classical Chinese and Sanskrit models.
    load_code("scripts/seg_code.py")

    nlp = spacy.load(args.in_model)
    # replace any existing clause_parser so the new code + config (sent_punct) take effect — the
    # model may already carry one from an earlier release built with the old factory.
    if "clause_parser" in nlp.pipe_names:
        nlp.remove_pipe("clause_parser")
    nlp.add_pipe("clause_parser",
                 config={"punct_tag": args.punct_tag, "sent_punct": args.sent_punct})
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
