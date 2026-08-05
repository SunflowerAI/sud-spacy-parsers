#!/usr/bin/env python
"""Copy a trained pipe from one arm into another, replacing it if already present.

Needed because a `training_<lang>_sud` arm trains several SUD features at once, and `spacy train`
picks `model-best` on the WEIGHTED MEAN of their scores -- so a feature can be checkpointed at an
epoch that was good for its neighbours and mediocre for itself. Latin is the case that forced this:
its `Shared` pipe peaked at dev F 37.34 while the saved checkpoint holds 31.90, because the epoch
that maximised the three-feature mean was chosen for `Subject`'s sake. Retraining that feature ALONE
(`SUD_FEATS=Shared SUD_SUFFIX=_shared`) lets it checkpoint on its own score; this script then puts
the result back into the multi-feature arm.

    graft_pipe.py RECIPIENT DONOR OUT sud_shared

Both arms must descend from the same base (same vocab and same frozen components) or the grafted
pipe's encoder will be reading a different model's world. That is checked, not assumed.
"""
import argparse
import filecmp
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import spacy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipient", help="arm to graft INTO")
    ap.add_argument("donor", help="arm to take the pipe FROM")
    ap.add_argument("out_model")
    ap.add_argument("pipes", nargs="+", help="pipe names to copy")
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the shared-base check (you had better know why)")
    args = ap.parse_args()

    try:
        import seg_code  # noqa: F401  (custom tokenisers + factories)
    except Exception as e:
        print(f"graft_pipe: seg_code not loaded ({type(e).__name__}: {e})")
    for name in ("sud_misc", "sud_shared_data", "sud_tagger", "sud_idiom"):
        try:
            __import__(name)
        except Exception:
            pass

    rec_dir, don_dir = pathlib.Path(args.recipient), pathlib.Path(args.donor)
    if not args.skip_check:
        # The pipes carry their OWN encoders, but they read the frozen components' predictions
        # (DEP/POS/MORPH/LEMMA and, for a tree layer, the parse itself). If those differ, the
        # grafted pipe is being fed by a model it was not trained against.
        for comp in ("tok2vec", "tagger", "parser", "morphologizer", "lemmatizer"):
            a, b = rec_dir / comp / "model", don_dir / comp / "model"
            if a.exists() and b.exists() and not filecmp.cmp(a, b, shallow=False):
                sys.exit(f"graft_pipe: {comp} differs between the two arms -- they do not share a "
                         f"base, so grafting would feed the pipe a different model's predictions")

    nlp = spacy.load(args.recipient)
    donor = spacy.load(args.donor)
    for name in args.pipes:
        if name not in donor.pipe_names:
            sys.exit(f"graft_pipe: {name} is not in the donor ({donor.pipe_names})")
        if name in nlp.pipe_names:
            nlp.remove_pipe(name)
        nlp.add_pipe(name, source=donor, last=True)
        print(f"  grafted {name}")
    nlp.to_disk(args.out_model)

    reloaded = spacy.load(args.out_model)
    print(f"{args.out_model}: pipeline {reloaded.pipe_names}")


if __name__ == "__main__":
    main()
