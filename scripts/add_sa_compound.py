#!/usr/bin/env python
"""Prepend the (non-trainable) sa_compound pipe to a trained Sanskrit model and save it.

The sa models read `Compound=Yes` as an INPUT feature (MORPH is among the embed attrs), normally
supplied by the tokeniser from the CSL join marker. `sa_compound` re-derives it from token adjacency
when the tokeniser did not run — i.e. when the caller passes TOKENS rather than raw text — so the
feature is present however the Doc was built. See the class docstring in scripts/sa_tokenizer.py.

It must run FIRST, before tok2vec: the shared encoder reads MORPH at its own position in the
pipeline, so anything set after tok2vec is invisible to every listener. Added after training because
it has no learned weights (mirrors add_clause_parser.py, which appends at the other end).

Usage:
    add_sa_compound.py IN_MODEL OUT_MODEL
"""
import argparse
import importlib.util

import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(path.split("/")[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    args = ap.parse_args()

    # registers the sa tokenizer + the sa_compound factory (and clause_parser, harmlessly)
    load_code("scripts/seg_code.py")

    nlp = spacy.load(args.in_model)
    # replace any existing one so new code takes effect on a rebuild
    if "sa_compound" in nlp.pipe_names:
        nlp.remove_pipe("sa_compound")
    nlp.add_pipe("sa_compound", first=True)
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
