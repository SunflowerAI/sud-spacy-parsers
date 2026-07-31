#!/usr/bin/env python
"""Append the (non-trainable) sud_idiom pipe to a trained model and save it.

sud_idiom derives SUD's `Idiom=Yes` / `InIdiom=Yes` MISC layer from `ExtPos` and the `unk`
relation, both of which the released pipeline already predicts, so it has no learned weights and
is added after training -- like id_lemma_case_fix and clause_parser.

It goes LAST, after clause_parser on the lzh/sa arms. clause_parser re-parses each sentence and
reassigns every head and deprel, so a rule that reads `unk` has to run on the tree clause_parser
leaves behind, not the whole-doc parse it discards. Running last also means the Doc rebuild cannot
drop the annotation, so no extension needs carrying across it.

Only the seven arms whose treebanks annotate idioms should get it: en, lzh, ja, fa, ar, la, sa.
zh/yue/ko/id carry no idiom annotation, so the component would emit an unvalidatable layer.

Usage:
    add_sud_idiom.py IN_MODEL OUT_MODEL
"""
import argparse
import importlib.util
import pathlib
import sys

import spacy

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))   # so sud_idiom can `import sud_misc`


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
        print(f"add_sud_idiom: skipped {name}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="pipes to remove before saving. fa/la train Subject AND Reported in one "
                         "arm but ship only Subject, so the trained sud_reported pipe has to be "
                         "stripped here -- declining to add the RULE is not enough, the weights "
                         "are already in the arm.")
    args = ap.parse_args()

    load_code("sud_misc.py")
    load_code("sud_idiom.py")
    # The incoming model may already carry the other SUD pipes (lzh gets sud_subject_rule first),
    # and spacy.load needs every factory registered or it fails with E002.
    load_code("sud_subject_frames.py", required=False)
    load_code("sud_subject_rule.py", required=False)
    load_code("sud_reported_data.py", required=False)
    load_code("sud_reported_rule.py", required=False)
    # the arm may itself need its custom tokenizer / clause_parser registered to load at all
    try:
        import seg_code  # noqa: F401
    except Exception as e:
        print(f"add_sud_idiom: seg_code not loaded ({type(e).__name__}: {e})")

    nlp = spacy.load(args.in_model)
    for name in args.drop:
        if name in nlp.pipe_names:
            nlp.remove_pipe(name)
            print(f"  dropped {name}")
    if "sud_idiom" in nlp.pipe_names:
        nlp.remove_pipe("sud_idiom")
    nlp.add_pipe("sud_idiom", last=True)
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
