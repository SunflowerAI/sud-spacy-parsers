#!/usr/bin/env python
"""`sud.HeadArgsTagger.v1` — [own | head | core arg 1 | core arg 2 | other dep 1 | other dep 2].

WHY NOT POOL THE DEPENDENTS. `sud.HeadDepsTagger.v1` averages every dependent into one vector, and
the average is exactly what destroys the signal for zero derivation: whether 歡 is "joy" or
"rejoice" turns on WHETHER IT HAS A CORE ARGUMENT AT ALL, and a mean over {a modifier, an object}
looks much like a mean over {a modifier, a modifier}. Keeping the first two CORE dependents
(subj/comp*) in their own slots, separately from the first two non-core ones, preserves the
argument/modifier contrast that the category decision actually rests on.

WHAT MOTIVATES IT. Gold arc structure is worth +4.17 on the VERB/NOUN slice for lzh (against +0.42
for predicted arcs), the only positive pre-flight among seven routes tried against zero derivation.
A hand-crafted symbolic arc channel (`sud.ArcStructureEmbed.v1`) captured only +0.05 of it over a
DEP-only channel, which is why this passes LEARNED REPRESENTATIONS of the actual tokens instead.

⚠ THE CAPACITY CONTROL IS `slots=0`, which zeroes every structural slice while keeping the identical
6x-width output layer. A gain over that control is tree information; a gain over a 1x-width plain
tagger might be nothing but parameters.

⚠ TRAIN WITH THE PARSER IN `annotating_components`, never on gold arcs: the channel would learn to
trust a signal it never meets at runtime.

⚠ `ops.scatter_add` IS CALLED BARE. Both real backends mutate in place and return None, so
assigning its result yields None (see memory `sud-scatter-add-gpu`).
"""
from typing import List, Optional

import numpy
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Model, Softmax_v2, chain, with_array

N_SLOTS = 6            # own, head, core1, core2, other1, other2
CORE_PREFIX = ("subj", "comp")


def _slot_indices(doc: Doc) -> numpy.ndarray:
    """(n, N_SLOTS) int32 of source token indices; -1 marks an empty slot."""
    n = len(doc)
    idx = numpy.full((n, N_SLOTS), -1, dtype="int32")
    core, other = {}, {}
    for t in doc:
        if t.head.i == t.i:
            continue
        bucket = core if t.dep_.split("@")[0].startswith(CORE_PREFIX) else other
        bucket.setdefault(t.head.i, []).append(t.i)
    for i, t in enumerate(doc):
        idx[i, 0] = i
        idx[i, 1] = t.head.i
        for s, lst in ((2, core.get(i, ())), (4, other.get(i, ()))):
            for k, j in enumerate(lst[:2]):          # FIRST TWO, in surface order
                idx[i, s + k] = j
    return idx


def _forward(model, docs: List[Doc], is_train: bool):
    tok2vec = model.get_ref("tok2vec")
    Xs, bp_tok2vec = tok2vec(docs, is_train)
    slots = model.attrs["slots"]
    ops = model.ops
    outs, all_idx = [], []
    for doc, X in zip(docs, Xs):
        n, w = X.shape
        idx = _slot_indices(doc)
        if slots == 0:                                # capacity control: structure removed
            idx = numpy.full_like(idx, -1)
            idx[:, 0] = numpy.arange(n)
        Y = ops.alloc2f(n, w * N_SLOTS)
        for s in range(N_SLOTS):
            col = idx[:, s]
            m = col >= 0
            if m.any():
                Y[m, s * w:(s + 1) * w] = X[ops.asarray1i(col[m])]
        outs.append(Y)
        all_idx.append(idx)

    def backprop(dYs):
        dXs = []
        for X, dY, idx in zip(Xs, dYs, all_idx):
            n, w = X.shape
            dX = ops.alloc2f(n, w)
            for s in range(N_SLOTS):
                col = idx[:, s]
                m = col >= 0
                if m.any():
                    # ⚠ BARE CALL — scatter_add mutates in place and returns None.
                    ops.scatter_add(dX, ops.asarray1i(col[m]), dY[m, s * w:(s + 1) * w])
            dXs.append(dX)
        return bp_tok2vec(dXs)

    return outs, backprop


def _init(model, X=None, Y=None):
    tok2vec = model.get_ref("tok2vec")
    tok2vec.initialize(X=X)
    if tok2vec.has_dim("nO"):
        model.set_dim("nO", tok2vec.get_dim("nO") * N_SLOTS)
    return model


def HeadArgs(tok2vec, slots: int = 1) -> Model:
    return Model("sud_head_args", _forward, init=_init, layers=[tok2vec],
                 refs={"tok2vec": tok2vec}, attrs={"slots": slots},
                 dims={"nO": (tok2vec.get_dim("nO") * N_SLOTS) if tok2vec.has_dim("nO") else None})


@registry.architectures("sud.HeadArgsTagger.v1")
def build_head_args_tagger(tok2vec, nO: Optional[int] = None, normalize: bool = False,
                           slots: int = 1) -> Model:
    t2v = HeadArgs(tok2vec, slots=slots)
    width = tok2vec.get_dim("nO") * N_SLOTS if tok2vec.has_dim("nO") else None
    output_layer = Softmax_v2(nO, width, normalize_outputs=normalize)
    model = chain(t2v, with_array(output_layer))
    model.set_ref("tok2vec", t2v)
    model.set_ref("output_layer", output_layer)
    return model
