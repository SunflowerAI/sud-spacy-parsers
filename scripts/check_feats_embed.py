#!/usr/bin/env python3
"""Self-check for `sud.MultiHashEmbedFeats.v1` (scripts/sud_feats_embed.py).

The layer is custom code that would ship inside a wheel, so it gets a standalone check rather than
trusting the training loss. Verifies: registry resolution, forward shapes, that backprop reaches the
parameters, a to_bytes/from_bytes round-trip, EXACT equivalence with stock `spacy.MultiHashEmbed.v2`
when no feature is configured (so an arm switching to it stays single-variable), that each feature
column is exactly `hash_string("Feat=Value")`, that the DECOMPOSITION actually holds (two tokens
sharing a Case collide in the Case column while differing in Number), and -- the one that matters
most here -- that an UNSET morph and an EMPTY one produce the SAME row, which is the trap that once
cost sa 6.8 LAS.

    .venv/bin/python scripts/check_feats_embed.py
"""
import pathlib
import sys

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                            # noqa: E402
from spacy.ml.models.tok2vec import MultiHashEmbed      # noqa: E402
from spacy.strings import hash_string                   # noqa: E402
from spacy.tokens import Doc                            # noqa: E402
from spacy.util import registry                         # noqa: E402
from spacy.vocab import Vocab                           # noqa: E402
from thinc.api import Adam                              # noqa: E402

import sud_feats_embed                                  # noqa: E402,F401
from sud_feats_embed import FeatsFeatureExtractor, MultiHashEmbedFeats   # noqa: E402

# a real language's vocab, not a bare `Vocab()`: the latter has no `lex_attr_getters`, so NORM
# comes out 0 for every token -- a fixture artefact that would fail check 6 for no good reason.
V = spacy.blank("la").vocab
docs = [Doc(V, words=["puellae", "rosam", "amat", "bonus"]),
        Doc(V, words=["rex", "venit"])]
ATTRS = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
ROWS = [2000, 1000, 1000, 1000]
FEATS = ["Case", "Number", "VerbForm"]
FROWS = [32, 16, 32]
ok = True


def check(label, cond):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and cond


def build(**kw):
    return MultiHashEmbedFeats(width=64, attrs=ATTRS, rows=ROWS,
                               include_static_vectors=False, **kw)


print("1. registered in spaCy's architecture registry")
check("sud.MultiHashEmbedFeats.v1 resolvable",
      registry.architectures.get("sud.MultiHashEmbedFeats.v1") is not None)

print("\n2. forward pass shapes")
m = build(feats=FEATS, feat_rows=FROWS)
m.initialize(X=docs)
Y, bp = m(docs, is_train=True)
check("one output per doc", len(Y) == 2)
check("shapes (4,64) and (2,64)", Y[0].shape == (4, 64) and Y[1].shape == (2, 64))

print("\n3. backprop + parameter update")
dY = [m.ops.alloc2f(*y.shape) + 1.0 for y in Y]
dX = bp(dY)
opt = Adam(0.001)


def snapshot(model):
    return [(nd.id, k, nd.get_param(k).copy())
            for nd in model.walk() for k in nd.param_names if nd.has_param(k)]


before = snapshot(m)
m.finish_update(opt)
after = {(i, k): v for i, k, v in snapshot(m)}
changed = sum(1 for i, k, v in before if not numpy.allclose(v, after[(i, k)]))
check("backprop returns a list", isinstance(dX, list))
check(f"parameters updated ({changed} tensors moved)", changed > 0)

print("\n4. serialisation round-trip")
b = m.to_bytes()
m2 = build(feats=FEATS, feat_rows=FROWS)
m2.initialize(X=docs)
m2.from_bytes(b)
Y1, _ = m(docs, is_train=False)
Y2, _ = m2(docs, is_train=False)
maxdiff = max(float(numpy.abs(a - c).max()) for a, c in zip(Y1, Y2))
check(f"round-trip identical (max abs diff {maxdiff})", maxdiff == 0.0)
check(f"to_bytes non-trivial ({len(b)} bytes)", len(b) > 1000)

print("\n5. equivalence with stock MultiHashEmbed when no feature is configured")
a0 = build()
s0 = MultiHashEmbed(width=64, attrs=ATTRS, rows=ROWS, include_static_vectors=False)
a0.initialize(X=docs)
s0.initialize(X=docs)
s0.from_bytes(a0.to_bytes())          # same shapes => weights transfer
Ya, _ = a0(docs, is_train=False)
Ys, _ = s0(docs, is_train=False)
d = max(float(numpy.abs(x - y).max()) for x, y in zip(Ya, Ys))
check(f"identical output with feats=[] (max abs diff {d})", d == 0.0)

print("\n6. each feature column is exactly hash_string('Feat=Value')")
rd = Doc(V, words=["puellae", "rosam", "amat"])
rd[0].set_morph("Case=Gen|Number=Sing")
rd[1].set_morph("Case=Acc|Number=Sing")
rd[2].set_morph("Number=Sing|Person=3|VerbForm=Fin")
ex = FeatsFeatureExtractor(["NORM", "SHAPE"], FEATS)
ex.initialize()
F, _ = ex([rd], is_train=False)
check("Case col: puellae == hash('Case=Gen')", F[0][0, 2] == hash_string("Case=Gen"))
check("Case col: rosam   == hash('Case=Acc')", F[0][1, 2] == hash_string("Case=Acc"))
check("Case col: amat has no Case -> hash('Case=')", F[0][2, 2] == hash_string("Case="))
check("Number col: all three Sing collide",
      F[0][0, 3] == F[0][1, 3] == F[0][2, 3] == hash_string("Number=Sing"))
check("VerbForm col: amat == hash('VerbForm=Fin')", F[0][2, 4] == hash_string("VerbForm=Fin"))
check("standard NORM column preserved and non-zero",
      all(F[0][i, 0] == rd[i].norm for i in range(3)) and F[0][0, 0] != 0)

print("\n7. the decomposition holds -- share one feature, differ in another")
dd = Doc(V, words=["rosa", "rosae"])
dd[0].set_morph("Case=Nom|Number=Sing")
dd[1].set_morph("Case=Nom|Number=Plur")
F2, _ = ex([dd], is_train=False)
check("same Case -> same Case column", F2[0][0, 2] == F2[0][1, 2])
check("different Number -> different Number column", F2[0][0, 3] != F2[0][1, 3])
# this is the whole point: under a single hashed MORPH bundle these two are unrelated symbols
mb = Doc(V, words=["rosa", "rosae"])
mb[0].set_morph("Case=Nom|Number=Sing")
mb[1].set_morph("Case=Nom|Number=Plur")
check("...whereas their MORPH bundle keys are unrelated", mb[0].morph.key != mb[1].morph.key)

print("\n8. an UNSET morph and an EMPTY one land on the SAME row (the sa 6.8-LAS trap)")
ue = Doc(V, words=["alpha", "beta"])
ue[1].set_morph("")
check("differing morph KEYS (the trap is real)", ue[0].morph.key != ue[1].morph.key)
F3, _ = ex([ue], is_train=False)
check("identical in every feature column",
      all(F3[0][0, c] == F3[0][1, c] for c in range(2, 2 + len(FEATS))))

print("\n9. multi-valued features are kept faithfully, not truncated")
mv = Doc(V, words=["gamma"])
mv[0].set_morph("Case=Acc,Nom")
F4, _ = ex([mv], is_train=False)
check("Case=Acc,Nom hashed whole", F4[0][0, 2] == hash_string("Case=Acc,Nom"))
check("...and differs from Case=Acc", F4[0][0, 2] != hash_string("Case=Acc"))

print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)
