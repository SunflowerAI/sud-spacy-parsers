#!/usr/bin/env python
"""Assert the segmented HeadDeps is EXACTLY the per-token loop it replaces.

The reference is not transcribed -- it is `git show <ref>:scripts/sud_tagger.py`, so this check
cannot drift from what was actually there. Both wrappers are given the SAME stub encoder, so the
only thing under test is the pooling itself: its output, and the gradient it hands back.

    .venv/bin/python scripts/check_head_deps.py [--ref HEAD] [--model training_en_gum_sud/model-best]
"""
import argparse
import importlib.util
import pathlib
import subprocess
import sys

import numpy
import spacy
from thinc.api import Model, NumpyOps

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODES = ["none", "deps", "closed", "deps2", "closed2"]

# The single-level modes -- including `deps`, the one the shipped `Shared` pipe uses -- must be
# EXACT, forward and backward. The two-level modes may differ in the BACKWARD pass by a float32
# rounding step and no more: there a token can receive several pooled contributions, and the loop
# summed them per-parent while the segmented form sums them in edge order. Measured against an
# exact float64 accumulation, the two implementations are EQUIDISTANT from it (4.172e-07 each) and
# differ from each other by exactly one ULP at that magnitude -- reordering, not error. The bound
# is expressed in ULPs of the largest gradient so it scales with the data rather than being a
# magic constant.
REORDERS = {"deps2", "closed2"}
ULP_BUDGET = 4


def load_ref(ref):
    src = subprocess.run(["git", "show", f"{ref}:scripts/sud_tagger.py"],
                         capture_output=True, text=True, check=True).stdout
    tmp = HERE / "_sud_tagger_ref.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("sud_tagger_ref", str(tmp))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        tmp.unlink(missing_ok=True)
    return mod


def stub_encoder(ops, arrays, sink):
    """Stands in for tok2vec: hands back fixed vectors, records the gradient it is given."""
    def forward(model, docs, is_train):
        def backprop(dXs):
            sink.append([numpy.array(d) for d in dXs])
            return []
        return list(arrays), backprop
    m = Model("stub", forward, dims={"nO": arrays[0].shape[1]})
    m.ops = ops
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref holding the reference implementation")
    ap.add_argument("--model", default="training_en_gum_sud/model-best")
    ap.add_argument("--width", type=int, default=96)
    args = ap.parse_args()

    import sud_tagger as new
    old = load_ref(args.ref)

    nlp = spacy.load(args.model)
    texts = [
        'I think -- I mean, I believe -- that she said "we are leaving now" to John.',
        "Monotheism and Holy War does have a violent interpretation, rather as others do.",
        "He went to the shop by train, and she walked, because it was raining heavily.",
        "Well, no -- the man who arrived yesterday told us that the meeting was cancelled.",
    ]
    docs = [nlp(t) for t in texts]
    docs.append(spacy.tokens.Doc(nlp.vocab, words=["unparsed", "doc", "here"]))   # the _parsed guard
    docs.append(spacy.tokens.Doc(nlp.vocab, words=[]))                            # the n == 0 path

    ops = NumpyOps()
    rng = numpy.random.default_rng(0)
    arrays = [numpy.asarray(rng.standard_normal((len(d), args.width)), dtype="f") for d in docs]
    dOuts = [numpy.asarray(rng.standard_normal((len(d), args.width * 3)), dtype="f") for d in docs]

    print(f"reference: {args.ref}   docs: {[len(d) for d in docs]}   width {args.width}")
    bad = 0
    for mode in MODES:
        for detach in (False, True):
            got = {}
            for tag, mod in (("old", old), ("new", new)):
                sink = []
                m = mod.HeadDeps(stub_encoder(ops, arrays, sink), pool=mode, detach=detach)
                m.ops = ops
                outs, bp = m(docs, is_train=True)
                bp(dOuts)
                got[tag] = ([numpy.array(o) for o in outs], sink[0])
            (fo, bo), (fn, bn) = got["old"], got["new"]
            df = max([0.0] + [float(numpy.abs(a - b).max()) for a, b in zip(fo, fn) if a.size])
            db = max([0.0] + [float(numpy.abs(a - b).max()) for a, b in zip(bo, bn) if a.size])
            scale = max([0.0] + [float(numpy.abs(a).max()) for a in bo if a.size])
            budget = (ULP_BUDGET * float(numpy.spacing(numpy.float32(scale)))
                      if (mode in REORDERS and not detach) else 0.0)
            ok = df == 0.0 and db <= budget
            bad += not ok
            note = f" (<= {budget:.1e}, {ULP_BUDGET} ulp)" if budget else " (exact)"
            print(f"  pool={mode:<8} detach={str(detach):<5} "
                  f"fwd |d|={df:.3e}  bwd |d|={db:.3e}{note}  {'OK' if ok else 'MISMATCH'}")
    print("\n" + ("within budget on every mode -- safe to swap in" if not bad
                  else f"{bad} MISMATCHES -- do not ship"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
