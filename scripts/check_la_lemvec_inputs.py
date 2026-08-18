#!/usr/bin/env python3
"""Confirm the parser actually RECEIVES the lemma and morphology it is configured to read.

A channel that arrives empty does not raise. The embed happily hashes ``Case=`` on every token and
looks up the empty string in the vector table, the run trains, the log looks ordinary, and the arm
scores like its own capacity control — which is indistinguishable from the channel being measured
and found worthless. NEGATIVE-RESULTS.md records the same check paying off on the ja XPOS prepass:
with ``annotating_components`` the parser saw a tag on 100 % of tokens, without it 0 %.

So this runs the config's FROZEN pipes over real training documents, exactly as
``annotating_components`` does during an update, and counts what reaches the predicted doc:

    lemma set        share of tokens the lemmatiser gave a lemma
    morph set        share it gave any FEATS at all
    per feature      share carrying each configured category
    vector hit       share whose PREDICTED lemma is in the table (the number that decides whether
                     the block is a channel or a constant, and it is NOT the gold-lemma coverage)

    .venv/bin/python scripts/check_la_lemvec_inputs.py configs/config_la_lemvec.cfg
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from spacy import util

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--code", type=Path, default=Path("scripts/seg_code.py"))
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--train", default=None, help="override paths.train")
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    overrides = {"paths.train": args.train} if args.train else {}
    config = util.load_config(args.config, overrides=overrides, interpolate=False)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    resolved = nlp.config.interpolate()
    corpus = util.resolve_dot_names(resolved, [resolved["training"]["train_corpus"]])[0]

    embed = resolved["components"]["tok2vec"]["model"]["embed"]
    feats = list(embed.get("feats") or [])
    table = {str(k) for k in np.load(embed["vectors"], allow_pickle=True)["keys"]}
    frozen = resolved["training"]["frozen_components"]
    annotating = resolved["training"]["annotating_components"]
    print(f"{args.config.name}: frozen={frozen} annotating={annotating}")
    if set(frozen) - set(annotating):
        print(f"  ⚠ frozen but NOT annotating: {sorted(set(frozen) - set(annotating))} — their "
              f"predictions will not reach the parser during training")

    pipes = [(n, p) for n, p in nlp.pipeline if n in annotating]
    n = Counter()
    total = 0
    for eg in itertools.islice(corpus(nlp), args.docs):
        doc = eg.predicted
        for _, pipe in pipes:
            doc = pipe(doc)
        for tok in doc:
            total += 1
            n["lemma set"] += bool(tok.lemma_)
            n["morph set"] += bool(str(tok.morph))
            n["vector hit"] += tok.lemma_ in table
            for f in feats:
                n[f"  {f}"] += bool(tok.morph.get(f))

    if not total:
        raise SystemExit("no tokens read — check paths.train")
    print(f"  {total} tokens over {args.docs} documents")
    for key in ("lemma set", "morph set", "vector hit"):
        print(f"    {key:<12} {n[key] / total:7.2%}")
    for f in feats:
        print(f"    {f:<12} {n['  ' + f] / total:7.2%}")
    if n["lemma set"] == 0 or n["morph set"] == 0:
        raise SystemExit("FAIL: a configured channel is empty on every token")
    if n["vector hit"] / total < 0.5:
        raise SystemExit(f"FAIL: only {n['vector hit'] / total:.2%} of predicted lemmas are in the "
                         f"table — the block is close to constant, which is its own control")
    print("  OK — both channels are populated on the predicted doc")


if __name__ == "__main__":
    main()
