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
    args = ap.parse_args()

    # register the lzh tokenizer + clause_parser factory before loading
    load_code("scripts/lzh_tokenizer.py")
    load_code("scripts/clause_parser.py")

    nlp = spacy.load(args.in_model)
    if "clause_parser" not in nlp.pipe_names:
        nlp.add_pipe("clause_parser", config={"punct_tag": args.punct_tag})
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
