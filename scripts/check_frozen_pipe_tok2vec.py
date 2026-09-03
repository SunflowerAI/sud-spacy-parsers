#!/usr/bin/env python3
"""Assert that `sud.FrozenPipeTok2Vec.v1` really froze the donor, and that the channel is LIVE.

Two failure modes, opposite to each other, and neither shows up in a training log:

  * the donor DRIFTS — a gradient reaches it and the "frozen transfer" is really a fine-tune, so
    the arm is not the experiment it claims to be;
  * the donor is INERT — its output never reaches the decision, so the arm is its own control and a
    null result means nothing. (`sud_lex_embed`'s constant-channel control was validated the same
    way: bit-identical weights proved the channel really was doing nothing.)

Usage:
    check_frozen_pipe_tok2vec.py --arm <trained model dir> --donor <donor model dir>
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def params(model):
    return {(id(n), p): np.asarray(n.get_param(p)).copy()
            for n in model.walk() for p in n.param_names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--component", default="morphologizer")
    ap.add_argument("--text", default="子曰學而時習之不亦說乎有朋自遠方來")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy
    sys.path.insert(0, "scripts")
    from sud_frozen_pipe_tok2vec import load_encoder

    nlp = spacy.load(a.arm)
    inner = [n for n in nlp.get_pipe("parser").model.walk()
             if n.name == "frozen_pipe_tok2vec"]
    if len(inner) != 1:
        sys.exit(f"  expected exactly one frozen_pipe_tok2vec in the parser, found {len(inner)}")
    trained = inner[0].layers[0]
    original = load_encoder(a.donor, a.component)

    # 1. DID IT DRIFT? every donor parameter must be bit-identical after training.
    tp, op = list(params(trained).values()), list(params(original).values())
    if len(tp) != len(op):
        sys.exit(f"  donor shape mismatch: {len(tp)} vs {len(op)} parameter arrays")
    worst = max((np.abs(x - y).max() if x.size else 0.0) for x, y in zip(tp, op))
    n_par = sum(x.size for x in tp)
    print(f"  donor drift: max |Δ| {worst:.3e} over {n_par} parameters "
          f"-> {'FROZEN' if worst == 0.0 else 'DRIFTED — this is a fine-tune, not a transfer'}")

    # 2. IS IT LIVE? perturbing the donor's output must change the parse.
    before = [(t.text, t.head.i, t.dep_) for t in nlp(a.text)]
    for n in trained.walk():
        for p in n.param_names:
            w = n.get_param(p)
            if w.size:
                n.set_param(p, np.zeros_like(w))
    after = [(t.text, t.head.i, t.dep_) for t in nlp(a.text)]
    changed = sum(1 for x, y in zip(before, after) if x != y)
    print(f"  channel liveness: zeroing the donor changes {changed} of {len(before)} arcs "
          f"-> {'LIVE' if changed else 'INERT — the arm is its own control'}")
    if worst != 0.0 or not changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
