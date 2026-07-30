#!/usr/bin/env python
"""Append the (non-trainable) id_lemma_case_fix pipe to a trained Indonesian model and save it.

id_lemma_case_fix is a safety-net override for the trainable_lemmatizer's sentence-initial
capitalisation gap (see scripts/id_lemma_case_fix.py). Added after training because it has no
learned weights -- it reads a static lookup table harvested from the training treebank.

Usage:
    add_id_lemma_case_fix.py IN_MODEL OUT_MODEL
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
    args = ap.parse_args()

    load_code("scripts/id_lemma_case_fix.py")

    nlp = spacy.load(args.in_model)
    if "id_lemma_case_fix" in nlp.pipe_names:
        nlp.remove_pipe("id_lemma_case_fix")
    nlp.add_pipe("id_lemma_case_fix", last=True)
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
