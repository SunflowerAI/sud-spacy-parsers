#!/usr/bin/env python3
"""Confirm the agreement block encodes agreement, and is neither constant nor all-unknown.

Two failure modes this catches, both of which train cleanly and score like the block's own capacity
control -- which is indistinguishable from the channel having been measured and found worthless:

  every dim ~0     the frozen morphologiser never reached the embed, so nothing declares Case at all
  dims 8-9 ~1      everything is compatible with everything, so the block carries no contrast

A synthetic check runs first, because a per-dim mean cannot tell a correct comparison from a
plausible-looking wrong one.

    .venv/bin/python scripts/check_la_agree_channel.py configs/config_la_agree.cfg
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import spacy
from spacy import util
from spacy.tokens import Doc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sud_lemmavec_embed import AGREE_DIMS, AgreeExtractor  # noqa: E402

LABELS = ["off-4", "off-3", "off-2", "off-1", "off+1", "off+2", "off+3", "off+4",
          "any-left", "any-right", "n-compat", "unknown"]


def synthetic() -> None:
    """`magnam urbem videt` -- the adjective agrees with the noun and with nothing else."""
    nlp = spacy.blank("la")
    doc = Doc(nlp.vocab, words=["magnam", "urbem", "uidet"])
    doc[0].set_morph("Case=Acc|Number=Sing|Gender=Fem")
    doc[1].set_morph("Case=Acc|Number=Sing|Gender=Fem")
    # the verb declares no Case at all -- it must come out UNKNOWN, not incompatible
    doc[2].set_morph("Number=Sing|Person=3")
    m = AgreeExtractor(20, False)
    arr = m.predict([doc])[0]
    fail = []
    if arr[0, 4] != 1.0:                     # magnam sees urbem at +1
        fail.append("adjective is not compatible with its noun at +1")
    if arr[0, 9] != 1.0 or arr[0, 8] != 0.0:
        fail.append("any-right/any-left wrong for the adjective")
    if arr[2, 11] != 1.0:
        fail.append("the caseless verb is not flagged unknown")
    if arr[2, :11].any():
        fail.append("the caseless verb has non-zero compatibility dims")
    if abs(arr[0, 10] - 1 / 8) > 1e-6:
        fail.append(f"n-compat should be 1/8 for one compatible token, got {arr[0, 10]}")
    # an incompatible neighbour must read 0 and NOT be confused with unknown
    doc2 = Doc(nlp.vocab, words=["magnam", "urbis"])
    doc2[0].set_morph("Case=Acc|Number=Sing|Gender=Fem")
    doc2[1].set_morph("Case=Gen|Number=Sing|Gender=Fem")
    a2 = m.predict([doc2])[0]
    if a2[0, 4] != 0.0 or a2[0, 11] != 0.0:
        fail.append("a genitive neighbour is not read as incompatible-but-known")
    if fail:
        raise SystemExit("SYNTHETIC CHECK FAILED:\n  " + "\n  ".join(fail))
    print("synthetic: agreement, hyperbaton reach, unknown flag and incompatibility all correct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--code", type=Path, default=Path("scripts/seg_code.py"))
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--train", default=None, help="override paths.train (it is null "
                    "in the config; the CLI supplies it at train time)")
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    synthetic()

    overrides = {"paths.train": args.train} if args.train else {}
    config = util.load_config(args.config, overrides=overrides, interpolate=False)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    resolved = nlp.config.interpolate()
    corpus = util.resolve_dot_names(resolved, [resolved["training"]["train_corpus"]])[0]
    embed = resolved["components"]["tok2vec"]["model"]["embed"]
    near = embed.get("agree_near", 20)
    const = embed.get("agree_constant", False)
    annotating = resolved["training"]["annotating_components"]
    pipes = [p for n, p in nlp.pipeline if n in annotating]

    m = AgreeExtractor(near, const)
    rows = []
    for eg in itertools.islice(corpus(nlp), args.docs):
        doc = eg.predicted
        for pipe in pipes:                      # exactly what annotating_components does
            doc = pipe(doc)
        rows.append(np.asarray(m.predict([doc])[0]))
    A = np.concatenate(rows)
    print(f"  {A.shape[0]} tokens over {args.docs} documents, agree_near={near} "
          f"agree_constant={const}")
    for i, lbl in enumerate(LABELS):
        print(f"    {lbl:<10} mean {A[:, i].mean():6.3f}")
    if const:
        print("  (capacity control: all zeros is CORRECT here)")
        return
    if not A[:, :11].any():
        raise SystemExit("FAIL: every compatibility dim is zero — the morphologiser's FEATS are "
                         "not reaching the embed, so this arm is its own control")
    if A[:, 11].mean() > 0.6:
        raise SystemExit(f"FAIL: {A[:, 11].mean():.1%} of tokens are UNKNOWN — too few carry "
                         "Case/Number/Gender for the block to say anything")
    if A[:, 8].mean() > 0.95 and A[:, 9].mean() > 0.95:
        raise SystemExit("FAIL: nearly everything is compatible with something on both sides — the "
                         "block carries no contrast at this window")
    print("  OK — the block is populated and discriminating")


if __name__ == "__main__":
    main()
