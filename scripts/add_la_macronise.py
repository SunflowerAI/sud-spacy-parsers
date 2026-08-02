#!/usr/bin/env python
"""Append the (non-trainable) la_macronise pipe to a trained Latin model and save it.

Mirrors add_clause_parser.py / add_sa_compound.py: the component has no learned weights, so it is
attached after training rather than being part of a training config. It runs LAST -- it reads the
morphologiser's FEATS and the tagger's UPOS, so everything it depends on must already have run.

The lookup table is copied into the model directory by the component's own ``to_disk`` (~0.55 MB),
so the packaged wheel is self-contained and does not need ``scripts/la_macron_lut.json.gz`` at
runtime -- only ``--code scripts/la_macronise.py`` at package time, for the factory.

``--no-lut`` attaches the pipe with NO table, which is how the RELEASED wheel is built: the pipe is
in the shipped pipeline and macronises as soon as the user fetches Morpheus, and until then it
passes every token through unchanged (see ``require_data`` in la_macronise.py). Shipping the pipe
without the data is the only arrangement the licences allow -- the table is Morpheus-derived
(CC BY-SA 3.0 US) and the model is CC BY-NC-SA -- and it is strictly better than shipping neither,
which left ``nlp.add_pipe`` to every caller and made ``doc._.macron`` absent by default.

Usage:
    add_la_macronise.py IN_MODEL OUT_MODEL [--lut scripts/la_macron_lut.json.gz | --no-lut]
"""
import argparse
import gzip
import json
import importlib.util
import pathlib
import sys

import spacy

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))   # so a loaded module can `import` its own siblings


def load_code(path, required=True):
    """Execute one component module so its @Language.factory registers.

    EVERY factory the input model's config names has to be registered before spacy.load, or it
    fails with E002 -- so this runs over the model's OTHER components too, not just ours. That is
    what `--code` is for: in the release pipeline this script is handed a model that already
    carries sud_misc/sud_idiom/sud_tagger (see package_sud.sh), and loading only la_macronise.py
    would have made attaching the macroniser the step that could not open the model."""
    path = str(path)
    if "/" not in path:
        path = str(_HERE / path)
    name = path.split("/")[-1][:-3]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # registered before exec, so a sibling import finds it
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        if required:
            raise
        print(f"add_la_macronise: skipped {path}: {type(exc).__name__}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--lut", default="scripts/la_macron_lut.json.gz")
    ap.add_argument("--no-lut", action="store_true",
                    help="attach the pipe with no lookup table (how the released wheel is built)")
    ap.add_argument("--code", default="",
                    help="comma-separated sibling modules whose factories the INPUT model needs "
                         "(e.g. sud_misc.py,sud_idiom.py); missing ones are skipped, not fatal")
    args = ap.parse_args()

    load_code("scripts/la_macronise.py")
    for extra in filter(None, (c.strip() for c in args.code.split(","))):
        load_code(extra, required=False)

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
    if args.no_lut:
        # Nothing to load. `to_disk` still writes its (empty) lut.json.gz, which is what makes the
        # shipped directory shaped exactly like a populated one -- a user who later drops a real
        # table in has no second layout to get right.
        print("  no table (--no-lut): the pipe ships bare and waits for fetch_morpheus()")
    else:
        comp._load_blob(json.loads(gzip.open(args.lut, "rb").read().decode("utf-8")))
        print(f"  loaded table: L1 {len(comp.l1)} L2 {len(comp.l2)} L3 {len(comp.l3)} "
              f"S4 {len(comp.s4)} S3 {len(comp.s3)}")
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
