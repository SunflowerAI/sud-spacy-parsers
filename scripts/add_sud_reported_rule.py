#!/usr/bin/env python
"""Append the (non-trainable) sud_reported_rule pipe to a trained model and save it.

Used for en/ar/sa, where the rule beats the trained `sud_tagger` end to end by a wide margin
(F 66.7 vs 27.6, 73.5 vs 37.4, 68.8 vs 39.6). fa and la ship no Reported layer at all -- see
scripts/package_sud.sh.

Goes LAST, after clause_parser where present: the rule reads the parser's deprels and the
morphologiser's VerbForm/Mood, so it must see the tree clause_parser leaves behind.

Usage:
    add_sud_reported_rule.py IN_MODEL OUT_MODEL --lang ar
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
        print(f"add_sud_reported_rule: skipped {name}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--lang", required=True, help="lexicon to use (key into sud_reported_data)")
    args = ap.parse_args()

    load_code("sud_misc.py")
    load_code("sud_reported_data.py")
    load_code("sud_reported_rule.py")
    # The incoming model may already carry the other SUD pipes; spacy.load needs every factory.
    load_code("sud_idiom.py", required=False)
    load_code("sud_subject_frames.py", required=False)
    load_code("sud_subject_rule.py", required=False)
    try:
        import seg_code  # noqa: F401
    except Exception as e:
        print(f"add_sud_reported_rule: seg_code not loaded ({type(e).__name__}: {e})")

    nlp = spacy.load(args.in_model)
    if "sud_reported_rule" in nlp.pipe_names:
        nlp.remove_pipe("sud_reported_rule")
    # Drop the trained pipe for the same feature: the rule clears and rewrites `Reported`, so
    # keeping both would ship weights that can never affect the output (and the rule wins by a
    # wide margin anyway -- en F 66.7 vs 27.6, ar 73.5 vs 37.4, sa 68.8 vs 39.6).
    if "sud_reported" in nlp.pipe_names:
        nlp.remove_pipe("sud_reported")
        print("  removed the trained sud_reported pipe (superseded by the rule)")
    nlp.add_pipe("sud_reported_rule", last=True, config={"lang": args.lang})
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
