#!/usr/bin/env python
"""Append the (non-trainable) `lzh_upos_rules` pipe to a built lzh model and save it.

WHAT IT ADDS. Post-morphologiser UPOS repair keyed on the parser's analysis of a token's CHILDREN
and its HEAD -- the one runtime-visible family the morphologiser cannot see, since its own channel
embeds only the token's own deprel. Measured on the ARM THAT SHIPS
(`training_lzh_seg_sud_xw`, scored against the remapped gold that makes the split scoreable):

    UPOS 91.66 -> 92.69  (+1.02)     之 58.42 -> 89.36  (+30.94)     NOUN/VERB 86.78 -> 86.77

⚠ IT INTRODUCES A UPOS VALUE THE RELEASED MODEL NEVER EMITTED. Genitive 之 becomes PART, where
0.3.0 always said SCONJ. That is the point -- the traditional analysis distinguishes the nominal
genitive from the clausal nominaliser -- but it is a BEHAVIOURAL CHANGE for anything downstream
that keyed on the old output, so it warrants a version bump rather than a clobber.

⚠ POSITION: LAST, and after the morphologiser by necessity. It rewrites `token.pos_`, so any pipe
that READS UPOS must run before it or it becomes coupled to this layer (CLAUDE.md standing
hazard 5). In the released lzh pipeline the tagger reads UPOS+FEATS, so this must follow the
tagger -- which `last=True` gives, since `sud_subject`/`sud_shared` read the tree, not UPOS.

⚠ NO WEIGHTS. This pipe carries none, so every model file must come out BYTE-IDENTICAL to the
input arm; the caller should assert that rather than trust it.

Usage:
    add_lzh_upos_rules.py IN_MODEL OUT_MODEL
"""
import argparse
import importlib.util
import pathlib
import sys


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    a = ap.parse_args()
    load_code("scripts/seg_code.py")
    import spacy
    nlp = spacy.load(a.in_model)
    if "morphologizer" not in nlp.pipe_names:
        sys.exit(f"{a.in_model}: no morphologizer; lzh_upos_rules has nothing to repair")
    if "parser" not in nlp.pipe_names:
        sys.exit(f"{a.in_model}: no parser; the rules read a token's children and head")
    if "lzh_upos_rules" in nlp.pipe_names:
        sys.exit(f"{a.in_model}: already carries lzh_upos_rules")
    nlp.add_pipe("lzh_upos_rules", last=True)
    nlp.to_disk(a.out_model)
    print(f"{a.in_model} -> {a.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
