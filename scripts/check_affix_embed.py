#!/usr/bin/env python3
"""Self-check for `sud.MultiHashEmbedAffix.v1` (scripts/sud_affix_embed.py).\n\nThe layer is custom code that ships inside the sa wheel, so it gets a standalone check rather\nthan trusting the training loss. Verifies: registry resolution, forward shapes, that backprop\nreaches the parameters, a to_bytes/from_bytes round-trip, EXACT equivalence with stock\n`spacy.MultiHashEmbed.v2` when no affix is configured (so an arm switching to it stays\nsingle-variable), and that each affix column is exactly `hash_string(token.text[-k:])`.\n\n    .venv/bin/python scripts/check_affix_embed.py\n"""
import sys
import pathlib
import numpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                            # noqa: E402
import sa_tokenizer                                     # noqa: E402,F401  (registers the sa tokenizer)
from spacy.tokens import Doc                            # noqa: E402
from spacy.vocab import Vocab                           # noqa: E402
from spacy.ml.models.tok2vec import MultiHashEmbed      # noqa: E402
from thinc.api import Adam                              # noqa: E402
import sud_affix_embed                                  # noqa: E402,F401
from sud_affix_embed import MultiHashEmbedAffix         # noqa: E402

V = Vocab()
docs = [Doc(V, words=["devasya", "putreṇa", "gacchati", "aśvān"]),
        Doc(V, words=["rājā", "bhavati"])]
ATTRS = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
ROWS = [2000, 1000, 1000, 1000]
ok = True


def check(label, cond):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and cond


print("1. registered in spaCy's architecture registry")
from spacy.util import registry                          # noqa: E402
check("sud.MultiHashEmbedAffix.v1 resolvable", registry.architectures.get("sud.MultiHashEmbedAffix.v1") is not None)

print("\n2. forward pass shapes")
m = MultiHashEmbedAffix(width=64, attrs=ATTRS, rows=ROWS, include_static_vectors=False,
                        suffixes=[5], suffix_rows=[8000], prefixes=[2], prefix_rows=[500])
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


params_before = snapshot(m)
m.finish_update(opt)
params_after = {(i, k): v for i, k, v in snapshot(m)}
changed = sum(1 for i, k, v in params_before if not numpy.allclose(v, params_after[(i, k)]))
check("backprop returns a list", isinstance(dX, list))
check(f"parameters updated ({changed} tensors moved)", changed > 0)

print("\n4. serialisation round-trip")
b = m.to_bytes()
m2 = MultiHashEmbedAffix(width=64, attrs=ATTRS, rows=ROWS, include_static_vectors=False,
                         suffixes=[5], suffix_rows=[8000], prefixes=[2], prefix_rows=[500])
m2.initialize(X=docs)
m2.from_bytes(b)
Y2, _ = m2(docs, is_train=False)
Y1, _ = m(docs, is_train=False)
maxdiff = max(float(numpy.abs(a - c).max()) for a, c in zip(Y1, Y2))
check(f"round-trip identical (max abs diff {maxdiff})", maxdiff == 0.0)
check(f"to_bytes non-trivial ({len(b)} bytes)", len(b) > 1000)

print("\n5. equivalence with stock MultiHashEmbed when no affixes are configured")
a0 = MultiHashEmbedAffix(width=64, attrs=ATTRS, rows=ROWS, include_static_vectors=False)
s0 = MultiHashEmbed(width=64, attrs=ATTRS, rows=ROWS, include_static_vectors=False)
a0.initialize(X=docs); s0.initialize(X=docs)
s0.from_bytes(a0.to_bytes())          # same shapes => weights transfer
Ya, _ = a0(docs, is_train=False)
Ys, _ = s0(docs, is_train=False)
d = max(float(numpy.abs(x - y).max()) for x, y in zip(Ya, Ys))
check(f"identical output with suffixes=[] (max abs diff {d})", d == 0.0)

print("\n6. the affix column is exactly hash_string(text[-k:])")
from sud_affix_embed import AffixFeatureExtractor       # noqa: E402
from spacy.strings import hash_string                   # noqa: E402
# use a real model's vocab: a bare Vocab() leaves NORM unset (all zeros), which is a fixture
# artefact, not a layer bug.
rv = spacy.load(str(pathlib.Path(__file__).resolve().parent.parent / "training_sa_lemma3_noannot/model-best")).vocab
words = ["devasya", "rāmasya", "gacchati", "bhavati", "ca"]
rd = Doc(rv, words=words)
for k in (3, 4, 5, 6):
    ex = AffixFeatureExtractor(["NORM", "SHAPE"], [k], [2])
    ex.initialize()
    F, _ = ex([rd], is_train=False)
    sfx_ok = all(F[0][i, 2] == hash_string(w[-k:]) for i, w in enumerate(words))
    pfx_ok = all(F[0][i, 3] == hash_string(w[:2]) for i, w in enumerate(words))
    std_ok = all(F[0][i, 0] == rd[i].norm for i in range(len(words)))
    check(f"k={k}: suffix col == hash_string(text[-{k}:])", sfx_ok)
    check(f"k={k}: prefix col == hash_string(text[:2])", pfx_ok)
    check(f"k={k}: standard NORM column preserved and non-zero", std_ok and F[0][0, 0] != 0)

print("\n7. shared endings collide, distinct ones do not")
ex4 = AffixFeatureExtractor(["NORM"], [4], [])
ex4.initialize()
F2, _ = ex4([Doc(rv, words=["devasya", "rāmasya", "gacchati"])], is_train=False)
check("devasya / rāmasya share -asya at k=4", F2[0][0, 1] == F2[0][1, 1])
check("gacchati differs", F2[0][0, 1] != F2[0][2, 1])
check("their NORMs still differ", F2[0][0, 0] != F2[0][1, 0])
F3, _ = ex4([Doc(rv, words=["ca", "ca"])], is_train=False)
check("token shorter than k falls back to the whole word", F3[0][0, 1] == hash_string("ca"))

print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)
