#!/usr/bin/env python
"""Remove named pipes from a trained model and save it, so no dead weights ship.

A `training_<lang>_sud` arm trains one pipe per SUD feature, but which of them a wheel SHIPS is
decided per language and per feature (see the tables in scripts/package_sud.sh) -- an arm trained
for three features may ship one. Until now the dropping rode along with something else:
`add_sud_idiom.py --drop` and `add_sud_reported_rule.py` both remove pipes, but both also ADD one,
which is wrong wherever the language does not want the thing being added. Cantonese is the case in
point: it ships the trained `Subject` pipe, annotates no idioms, and must not ship `Shared`.

    drop_pipes.py IN_MODEL OUT_MODEL sud_shared [sud_reported ...]

Naming a pipe that is not there is not an error -- the script is meant to be idempotent, so a
packaging run can be repeated over an already-trimmed model.
"""
import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import spacy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("pipes", nargs="+", help="pipe names to remove")
    args = ap.parse_args()

    try:
        import seg_code  # noqa: F401  (custom tokenisers + factories, so the arm loads at all)
    except Exception as e:
        print(f"drop_pipes: seg_code not loaded ({type(e).__name__}: {e})")
    for name in ("sud_misc.py", "sud_idiom.py", "sud_subject_rule.py", "sud_shared_rule.py"):
        try:
            __import__(name[:-3])
        except Exception:
            pass

    nlp = spacy.load(args.in_model)
    for name in args.pipes:
        if name in nlp.pipe_names:
            nlp.remove_pipe(name)
            print(f"  dropped {name}")
    nlp.to_disk(args.out_model)
    print(f"{args.out_model}: pipeline {nlp.pipe_names}")


if __name__ == "__main__":
    main()
