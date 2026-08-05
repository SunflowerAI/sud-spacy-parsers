#!/usr/bin/env python
"""Append the (non-trainable) sud_shared_rule pipe to a trained model and save it.

Used for the languages where the harvested table beats the trained `sud_tagger` end to end; see
scripts/package_sud.sh for the comparison and scripts/eval_sud_shared.py for how it is measured.

Goes LAST, after clause_parser where present: the rule reads the parser's heads and relations, so
it must see the tree clause_parser leaves behind rather than the whole-doc parse it discards.

`--drop-trained` removes the trained `sud_shared` pipe when the rule wins, so no dead weights ship
-- the same thing add_sud_reported_rule.py does for `Reported`.

Usage:
    add_sud_shared_rule.py IN_MODEL OUT_MODEL --lang lzh [--drop-trained]
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
        print(f"add_sud_shared_rule: skipped {name}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--lang", required=True, help="table to use (key into sud_shared_frames)")
    ap.add_argument("--drop-trained", action="store_true",
                    help="remove the trained sud_shared pipe, so no dead weights ship")
    args = ap.parse_args()

    load_code("sud_misc.py")
    load_code("sud_shared_data.py")
    load_code("sud_shared_frames.py")
    load_code("sud_shared_rule.py")
    # The incoming model may already carry these; spacy.load needs their factories registered.
    load_code("sud_idiom.py", required=False)
    load_code("sud_tagger.py", required=False)
    try:
        import seg_code  # noqa: F401  (custom tokenisers, so the arm loads at all)
    except Exception as e:
        print(f"add_sud_shared_rule: seg_code not loaded ({type(e).__name__}: {e})")

    nlp = spacy.load(args.in_model)
    if args.drop_trained and "sud_shared" in nlp.pipe_names:
        nlp.remove_pipe("sud_shared")
    if "sud_shared_rule" in nlp.pipe_names:
        nlp.remove_pipe("sud_shared_rule")
    nlp.add_pipe("sud_shared_rule", last=True, config={"lang": args.lang})
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
