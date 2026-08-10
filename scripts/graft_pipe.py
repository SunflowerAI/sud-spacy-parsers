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
import json
import pathlib
import sys

# Which `performance` keys belong to which pipe.  A graft replaces a pipe but `nlp.to_disk` writes
# the RECIPIENT's meta, so without this the wheel reports the score of the model that was replaced
# -- and it does so silently, in the one field `spacy info` shows users.  Latin is the case that
# forced it: the grafted arm kept tag_acc 0.9028, measured on a 1 952-label tagset, while shipping
# a 2 342-label one.  Only the grafted pipe's own keys move; everything else is frozen and its
# scores are (verifiably) identical between the two arms.
PIPE_METRICS = {
    "tagger": ("tag_acc", "tag_micro_p", "tag_micro_r", "tag_micro_f"),
    "morphologizer": ("pos_acc", "morph_acc", "morph_micro_p", "morph_micro_r", "morph_micro_f"),
    "lemmatizer": ("lemma_acc",),
    "parser": ("dep_uas", "dep_las", "dep_las_per_type"),
    "sud_subject": ("sud_subject_f", "sud_subject_p", "sud_subject_r"),
    "sud_reported": ("sud_reported_f", "sud_reported_p", "sud_reported_r"),
    "sud_shared": ("sud_shared_f", "sud_shared_p", "sud_shared_r"),
}

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
        # ...but NOT the pipes being grafted: those are the ones that are meant to differ, which
        # is the whole point of the call.  Checking them made every same-name replacement fail.
        for comp in [c for c in ("tok2vec", "tagger", "parser", "morphologizer", "lemmatizer")
                     if c not in args.pipes]:
            a, b = rec_dir / comp / "model", don_dir / comp / "model"
            if a.exists() and b.exists() and not filecmp.cmp(a, b, shallow=False):
                sys.exit(f"graft_pipe: {comp} differs between the two arms -- they do not share a "
                         f"base, so grafting would feed the pipe a different model's predictions")

    nlp = spacy.load(args.recipient)
    donor = spacy.load(args.donor)
    for name in args.pipes:
        if name not in donor.pipe_names:
            sys.exit(f"graft_pipe: {name} is not in the donor ({donor.pipe_names})")
        # A replacement must go back where it came from.  `last=True` is right for a pipe being
        # ADDED, and silently wrong for one being SWAPPED: it would move a tagger to the end of
        # the pipeline, behind the `sud_*` pipes that read its arm's predictions -- the same
        # class of ordering bug that put `clause_parser` after `sud_shared` in the lzh wheel,
        # which built, loaded and said nothing.
        where = {"last": True}
        if name in nlp.pipe_names:
            i = nlp.pipe_names.index(name)
            nlp.remove_pipe(name)
            if i < len(nlp.pipe_names):
                where = {"before": nlp.pipe_names[i]}
        nlp.add_pipe(name, source=donor, **where)
        print(f"  grafted {name} at {nlp.pipe_names.index(name)} ({nlp.pipe_names})")
    nlp.to_disk(args.out_model)

    # carry the grafted pipes' own scores across, so meta.json describes what actually ships
    meta_path = pathlib.Path(args.out_model) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    donor_meta = json.loads((don_dir / "meta.json").read_text(encoding="utf-8"))
    donor_perf = donor_meta.get("performance", {})
    moved = {}
    for name in args.pipes:
        for key in PIPE_METRICS.get(name, ()):
            if key in donor_perf and meta.get("performance", {}).get(key) != donor_perf[key]:
                moved[key] = (meta["performance"].get(key), donor_perf[key])
                meta["performance"][key] = donor_perf[key]
    if moved:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for k, (a, b) in moved.items():
            print(f"  performance.{k}: {a} -> {b}")

    reloaded = spacy.load(args.out_model)
    print(f"{args.out_model}: pipeline {reloaded.pipe_names}")


if __name__ == "__main__":
    main()
