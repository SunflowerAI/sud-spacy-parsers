#!/usr/bin/env python3
"""Derive a *tagger-only* config whose XPOS is DOWNSTREAM of UPOS and FEATS.

Every arm here predicts XPOS from a `tagger` that sits in the BASE pipeline, beside the parser and
*before* the morphologiser was ever added -- so the one component whose target is largely a
restatement of UPOS+FEATS is the only one that cannot see them.  Measured on the treebanks
themselves (majority-class maps fitted on train, scored on test), knowing gold UPOS+FEATS on top of
the form is worth +19.6 XPOS points on ar, +13.8 on la, +14.2 on zh, +13.2 on en, +11.7 on yue,
+8.2 on id, +4.3 on fa/ko; ar and yue are nearly deterministic from UPOS+FEATS alone (99.9 / 100.0).

So: move the tagger to the END of the pipeline, behind the morphologiser, and let its encoder read
POS and MORPH alongside the token embedding it already had.  The recipe is otherwise the ordinary
freeze recipe -- source every other component and freeze it, train one fresh tagger over its OWN
encoder -- so the frozen weights come out byte-identical and no published LAS/UAS/lemma figure
needs re-measuring.  The trained tagger is grafted back with graft_pipe.py.

Three things this has to get right, each of which fails SILENTLY if it does not:

  * `annotating_components` must actually RUN the morphologiser, or POS/MORPH are absent in
    training and appear from nowhere at inference.  This is the trap CLAUDE.md records for the
    `--structural` SUD arms, where a missing `tok2vec` left the mask empty and the loss a flat
    0.00 with nothing raising.  tok2vec goes in too, for the frozen listeners.
  * The encoder must be an explicit MultiHashEmbed: `spacy.HashEmbedCNN.v2` HARD-CODES
    NORM/PREFIX/SUFFIX/SHAPE and cannot express POS or MORPH (the same reason sa's morphologiser
    is hand-maintained -- see scripts/config_guard.py).  The stack written here is exactly what
    HashEmbedCNN builds internally (rows [E, E/2, E/2, E/2]) plus the two new channels, so the
    arm differs from the tagger it replaces in those channels and nothing else.
  * The tagger must come AFTER the morphologiser in the pipeline, not merely be trained that way.
    Pipeline order is what makes UPOS/FEATS available at inference.

`--no-cond` writes the same arm WITHOUT any conditioning channel: the capacity control, since the
extra channels are also extra parameters.  A gain that survives the control is the feature.

`--feats Case,Number,...` swaps the single hashed MORPH bundle for ONE TABLE PER FEATURE
(`sud.MultiHashEmbedFeats.v1`).  This is the variant worth running: conditioning on the bundle was
measured and LOST (NEGATIVE-RESULTS.md), and the reason is that `MORPH` hashes the whole normalised
FEATS string, so `Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` are unrelated symbols and an
unseen bundle has no decomposition to fall back on.  Derive the list with
`scripts/build_feats_inventory.py`, which ranks each category by the information it carries about
XPOS *once the form is already known* -- and reports that zh, id and ko have no such category at
all, so those arms have nothing to condition on and should not be run.

    make_xpos_config.py configs/config_ar_lemma.cfg training_ar_lemma/model-best \
        --out configs/config_ar_xpos2.cfg
"""
import argparse
import json
import os
import pathlib
import sys

from thinc.api import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_guard import guard_overwrite                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", help="config of the arm to stack on (usually the lemma arm's)")
    ap.add_argument("source_model", help="model-best dir to source + freeze every other pipe from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-dir", default=None,
                    help="only for a STREAMING base config (max_epochs=-1), whose init sees 100 examples")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--embed-size", type=int, default=5000)
    ap.add_argument("--pos-rows", type=int, default=100)
    ap.add_argument("--morph-rows", type=int, default=4000)
    ap.add_argument("--feats", default=None,
                    help="comma-separated FEATS keys, one embedding table each (replaces the "
                         "hashed MORPH bundle). Derive with build_feats_inventory.py --emit")
    ap.add_argument("--feat-rows", default=None,
                    help="comma-separated row counts matching --feats")
    ap.add_argument("--no-cond", action="store_true",
                    help="capacity control: same arm, no POS/MORPH channels")
    ap.add_argument("--top", action="store_true",
                    help="inject the conditioning ABOVE the encoder: keep the released tagger's "
                         "Tok2VecListener on the frozen shared encoder and concatenate the "
                         "morphology channels just below the softmax (sud.Tok2VecPlusFeats.v1)")
    ap.add_argument("--feat-width", type=int, default=32,
                    help="--top only: width of the concatenated morphology side channel")
    ap.add_argument("--warm-start", default=None, metavar="ARM",
                    help="--top only: start AS this arm's released tagger (sud.WarmStartTagger.v1) "
                         "-- copies its head into the first columns, zeroes the new ones, and "
                         "copies its inner encoder if it has one (la, en_gum). Also fixes the new "
                         "tagger's LABEL ORDER to the released one, which the copy requires.")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    guard_overwrite(a.out, "tagger", "spacy.Tok2Vec.v2", a.force)
    cfg = Config().from_disk(a.base_config, interpolate=False)

    # 1) tagger LAST, behind the morphologiser -- this is what puts UPOS/FEATS upstream of XPOS at
    #    inference, and it is a property of the pipeline, not of how the thing was trained.
    pipe = [n for n in cfg["nlp"]["pipeline"] if n != "tagger"]
    if "morphologizer" not in pipe:
        raise SystemExit(f"{a.base_config}: no morphologizer in the pipeline -- nothing to be "
                         f"downstream OF. Point this at the arm's *_morph or *_lemma config.")
    cfg["nlp"]["pipeline"] = pipe + ["tagger"]

    # 2) source + freeze everything that is not the tagger
    frozen = [n for n in pipe]
    for name in frozen:
        cfg["components"][name] = {"source": a.source_model}
    cfg["training"]["frozen_components"] = frozen

    # 3) ...and RUN the ones whose annotation the new encoder reads. A frozen component that is not
    #    listed here does not execute, so POS/MORPH would simply be absent from every training doc.
    cfg["training"]["annotating_components"] = [n for n in ("tok2vec", "morphologizer") if n in frozen]

    # never let init_tok2vec clobber the sourced (frozen) encoder -- yue's Mandarin-init bin would
    # overwrite the trained tok2vec with the raw init.
    for section in ("paths", "initialize"):
        if section in cfg and "init_tok2vec" in cfg[section]:
            cfg[section]["init_tok2vec"] = None

    # 4) the encoder. MultiHashEmbed rows [E, E/2, E/2, E/2] reproduce HashEmbedCNN exactly; the
    #    conditioning channels are appended, so --no-cond is the identical architecture minus them.
    E = a.embed_size
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    rows = [E, E // 2, E // 2, E // 2]
    feats = [f for f in (a.feats or "").split(",") if f]
    frows = [int(r) for r in (a.feat_rows or "").split(",") if r]
    if a.no_cond:
        feats, frows = [], []
    else:
        attrs += ["POS"]
        rows += [a.pos_rows]
        if feats:
            # per-FEATURE tables REPLACE the bundle: keeping both would confound the comparison
            # with the arm that has only the bundle, which is the thing this is measured against.
            if not frows:
                raise SystemExit("--feats needs --feat-rows (build_feats_inventory.py --emit)")
            if len(frows) != len(feats):
                raise SystemExit(f"--feats has {len(feats)} keys but --feat-rows has {len(frows)}")
        else:
            attrs += ["MORPH"]
            rows += [a.morph_rows]

    # --top keeps the tagger reading EXACTLY what the released one reads -- a listener on the frozen
    # shared encoder -- and concatenates the morphology beneath the softmax instead of convolving it
    # in. That makes the comparison single-variable in the way the bottom-injection arms were not.
    if a.top:
        base_w = _upstream_width(a.source_model)
        side_attrs, side_rows = (["POS"], [a.pos_rows]) if not a.no_cond else ([], [])
        if not a.no_cond and not feats:
            side_attrs, side_rows = ["POS", "MORPH"], [a.pos_rows, a.morph_rows]
        # The inner encoder must be the SAME SHAPE as the released tagger's, or there is nothing to
        # warm-start from. Most arms read a listener on the shared encoder; la and en_gum carry
        # their own HashEmbedCNN from the XPOS-normalisation work, and it is copied verbatim.
        listener = {"@architectures": "spacy.Tok2VecListener.v1", "width": base_w,
                    "upstream": "tok2vec"}
        inner = listener
        if a.warm_start:
            rel = Config().from_disk(f"{a.warm_start}/config.cfg", interpolate=False)
            rel_tv = rel["components"]["tagger"]["model"]["tok2vec"]
            if "Listener" not in rel_tv["@architectures"]:
                inner = dict(rel_tv)      # e.g. spacy.HashEmbedCNN.v2, verbatim
        if a.no_cond:
            tok2vec_cfg = inner           # the purest control: the released tagger, retrained head
        else:
            tok2vec_cfg = {
                "@architectures": "sud.Tok2VecPlusFeats.v1",
                "tok2vec": inner,
                "feats_embed": {"@architectures": "sud.MultiHashEmbedFeats.v1",
                                "width": a.feat_width, "attrs": side_attrs, "rows": side_rows,
                                "feats": feats, "feat_rows": frows,
                                "include_static_vectors": False},
            }
        cfg["components"]["tagger"] = {
            "factory": "tagger", "neg_prefix": "!", "overwrite": False,
            "scorer": {"@scorers": "spacy.tagger_scorer.v1"},
            "model": {"@architectures": "spacy.Tagger.v2", "nO": None, "normalize": False,
                      "tok2vec": tok2vec_cfg},
        }
        if a.warm_start:
            # LABEL ORDER, not just the label set: the output layer is indexed by label id, so the
            # copy is only meaningful if the two agree position for position. Writing the released
            # arm's list and initialising from it makes that true by construction.
            labels = json.loads(pathlib.Path(f"{a.warm_start}/tagger/cfg").read_text())["labels"]
            # always the arm's OWN dir, never --labels-dir: la's labels_la_aug_xpos is a
            # checked-in artefact of the normalisation work and must not be rewritten in place.
            ldir = pathlib.Path(f"labels_{pathlib.Path(a.out).stem}")
            ldir.mkdir(parents=True, exist_ok=True)
            (ldir / "tagger.json").write_text(json.dumps(labels, indent=2))
            a.labels_dir = str(ldir)
            cfg["initialize"]["after_init"] = {"@callbacks": "sud.WarmStartTagger.v1",
                                               "source": a.warm_start}
        inner_kind = inner["@architectures"].split(".")[-2] if a.warm_start else "Tok2VecListener"
        _finish(cfg, a, feats, frows, attrs=side_attrs, rows=side_rows,
                note=f"TOP injection, inner={inner_kind} width {base_w} + side {a.feat_width}"
                     + (f", WARM-STARTED from {a.warm_start}" if a.warm_start else ""))
        return

    embed = ({"@architectures": "sud.MultiHashEmbedFeats.v1", "width": a.width, "attrs": attrs,
              "rows": rows, "feats": feats, "feat_rows": frows, "include_static_vectors": False}
             if feats else
             {"@architectures": "spacy.MultiHashEmbed.v2", "width": a.width, "attrs": attrs,
              "rows": rows, "include_static_vectors": False})

    cfg["components"]["tagger"] = {
        "factory": "tagger",
        "neg_prefix": "!",
        "overwrite": False,
        "scorer": {"@scorers": "spacy.tagger_scorer.v1"},
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "normalize": False,
            "tok2vec": {
                "@architectures": "spacy.Tok2Vec.v2",
                "embed": embed,
                "encode": {
                    "@architectures": "spacy.MaxoutWindowEncoder.v2",
                    "width": a.width,
                    "window_size": 1,
                    "maxout_pieces": 3,
                    "depth": a.depth,
                },
            },
        },
    }

    _finish(cfg, a, feats, frows, attrs=attrs, rows=rows,
            note=f"BOTTOM injection, own encoder w{a.width}/d{a.depth}")


def _upstream_width(source_model):
    """The shared encoder's output width -- what a Tok2VecListener must declare."""
    cfg = Config().from_disk(f"{source_model}/config.cfg", interpolate=False)
    model = cfg["components"]["tok2vec"]["model"]
    if "width" in model:                      # spacy.HashEmbedCNN.v2
        return int(model["width"])
    return int(model["encode"]["width"])      # spacy.Tok2Vec.v2


def _finish(cfg, a, feats, frows, attrs, rows, note):
    # the tagger is the only thing training, so it is the only thing selecting model-best. Leaving
    # the frozen components' constant scores in the mean would only blur which epoch was best FOR
    # THE TAGGER (the hazard that cost Latin's `Shared` its own checkpoint).
    sw = cfg["training"].setdefault("score_weights", {})
    for k in list(sw):
        sw[k] = None if k.endswith("_per_type") or k.endswith("_per_feat") else 0.0
    sw["tag_acc"] = 1.0

    # a streaming config (max_epochs = -1, e.g. la's augmented arm) initialises from the first 100
    # examples only, so the label set has to be handed in.
    cfg["initialize"]["components"] = (
        {"tagger": {"labels": {"@readers": "spacy.read_labels.v1",
                               "path": f"{a.labels_dir}/tagger.json", "require": True}}}
        if a.labels_dir else {})

    Config(cfg).to_disk(a.out)
    print(f"{a.out}: pipeline={cfg['nlp']['pipeline']}\n"
          f"  frozen={cfg['training']['frozen_components']}\n"
          f"  annotating={cfg['training']['annotating_components']}\n"
          f"  {note}\n"
          f"  tagger attrs={attrs} rows={rows}\n"
          f"  feats={feats or '(none)'} feat_rows={frows or '(none)'}\n"
          f"  labels={a.labels_dir + '/tagger.json' if a.labels_dir else 'collected at init'}")


if __name__ == "__main__":
    main()
