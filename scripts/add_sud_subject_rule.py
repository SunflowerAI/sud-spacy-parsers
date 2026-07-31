#!/usr/bin/env python
"""Append the (non-trainable) sud_subject_rule pipe to a trained model and save it.

Used for the languages where the frame table beats the trained `sud_tagger` end to end --
lzh (F 80.7 vs 59.0) and zh (31.6 vs 27.7). See scripts/package_sud.sh for the full comparison.

Goes LAST, after clause_parser where present: the rule reads the parser's deprel and the head's
UPOS, so it must see the tree clause_parser leaves behind, not the whole-doc parse it discards.

Usage:
    add_sud_subject_rule.py IN_MODEL OUT_MODEL --lang lzh
"""
import argparse
import importlib.util
import pathlib
import sys

import spacy

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def load_code(name, required=True):
    path = _HERE / name
    try:
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name[:-3]] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        if required:
            raise
        print(f"add_sud_subject_rule: skipped {name}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--lang", required=True, help="frame table to use (key into sud_subject_frames)")
    args = ap.parse_args()

    load_code("sud_misc.py")
    load_code("sud_subject_frames.py")
    load_code("sud_subject_rule.py")
    # The incoming model may already carry sud_idiom; spacy.load needs its factory registered too.
    load_code("sud_idiom.py", required=False)
    try:
        import seg_code  # noqa: F401  (custom tokenisers, so the arm loads at all)
    except Exception as e:
        print(f"add_sud_subject_rule: seg_code not loaded ({type(e).__name__}: {e})")

    nlp = spacy.load(args.in_model)
    if "sud_subject_rule" in nlp.pipe_names:
        nlp.remove_pipe("sud_subject_rule")
    nlp.add_pipe("sud_subject_rule", last=True, config={"lang": args.lang})
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
