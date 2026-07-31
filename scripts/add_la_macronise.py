#!/usr/bin/env python
"""Append the (non-trainable) la_macronise pipe to a trained Latin model and save it.

Mirrors add_clause_parser.py / add_sa_compound.py: the component has no learned weights, so it is
attached after training rather than being part of a training config. It runs LAST -- it reads the
morphologiser's FEATS and the tagger's UPOS, so everything it depends on must already have run.

The lookup table is copied into the model directory by the component's own ``to_disk`` (~0.55 MB),
so the packaged wheel is self-contained and does not need ``scripts/la_macron_lut.json.gz`` at
runtime -- only ``--code scripts/la_macronise.py`` at package time, for the factory.

Usage:
    add_la_macronise.py IN_MODEL OUT_MODEL [--lut scripts/la_macron_lut.json.gz]
"""
import argparse
import gzip
import json
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
    ap.add_argument("--lut", default="scripts/la_macron_lut.json.gz")
    args = ap.parse_args()

    load_code("scripts/la_macronise.py")

    nlp = spacy.load(args.in_model)
    if "la_macronise" in nlp.pipe_names:      # replace, so a rebuild picks up new code/table
        nlp.remove_pipe("la_macronise")
    # Add the pipe with lut=None so the SAVED config carries no build-time path, then populate the
    # table directly. Writing `nlp.config[...] = None` after the fact does NOT work -- spaCy
    # regenerates the component block from the factory's own config -- and a shipped config naming
    # "scripts/la_macron_lut.json.gz" would send the installed wheel looking for a file that is not
    # there. The table itself is serialised into the model directory by the component's to_disk.
    nlp.add_pipe("la_macronise", config={"lut": None}, last=True)
    comp = nlp.get_pipe("la_macronise")
    comp._load_blob(json.loads(gzip.open(args.lut, "rb").read().decode("utf-8")))
    print(f"  loaded table: L1 {len(comp.l1)} L2 {len(comp.l2)} L3 {len(comp.l3)} "
          f"S4 {len(comp.s4)} S3 {len(comp.s3)}")
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
