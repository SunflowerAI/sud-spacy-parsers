#!/usr/bin/env python
"""`sud.BiaffineArcScorer.v1` — windowed biaffine arc scores for an arc-factored parser.

Dozat & Manning's decomposition: project each token twice (as a potential HEAD and as a potential
DEPENDENT), then score a candidate arc with a bilinear form plus a head-side bias. Labels are NOT
scored here — they are computed only for the arc the decoder actually selects, which keeps memory
linear (see below).

⚠ WINDOWED, AND THAT IS A SIZE DECISION, NOT AN APPROXIMATION FOR ITS OWN SAKE. Scoring every pair
is O(n^2) in the length of the CALL, and this project's arms parse whole multi-sentence documents
(`convert -n 10`) rather than sentences — standing hazard 10, where a per-sentence metric cannot see
a cost that grows with the call. Measured on Latin: at k=50, 99.99 % of all gold arcs and 100 % of
CROSSING arcs fall inside the window (max crossing distance 40, sentences averaging 12.8 tokens).
So the window costs essentially nothing and turns the cost into O(n*k) ~ 0.6 KB per token.

⚠ COLUMN 0 IS THE VIRTUAL ROOT and is always in the candidate set regardless of distance, so any
token may become a sentence root. The decoder (`sud_cle.mst`) then returns a FOREST, which is how
sentence boundaries survive: this project's parsers double as sentencisers.

Parameters at d_arc=96: ~28 k here plus ~96 k for a label scorer — against the transition parser's
440 k, so the architecture is SMALLER, which was the condition for trying it at all.
"""
from typing import List, Tuple

import numpy as np
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Model, chain, glorot_uniform_init
from thinc.types import Floats2d

NEG = -1e4          # a finite stand-in for -inf: safe under softmax and float32


def _window_mask(n: int, k: int) -> np.ndarray:
    """(n, n+1) bool over [virtual root | tokens]; True where an arc may be scored."""
    m = np.zeros((n, n + 1), dtype="bool")
    m[:, 0] = True                                   # the virtual root is always a candidate
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    m[:, 1:] = d <= k
    np.fill_diagonal(m[:, 1:], False)                # no self-arcs
    return m


@registry.architectures("sud.BiaffineArcScorer.v1")
def BiaffineArcScorer(tok2vec: Model, hidden_width: int = 96, window: int = 50) -> Model:
    head = Linear(hidden_width, init_W=glorot_uniform_init)
    dep = Linear(hidden_width, init_W=glorot_uniform_init)
    return Model(
        "sud_biaffine_arc",
        _forward,
        init=_init,
        layers=[tok2vec, head, dep],
        refs={"tok2vec": tok2vec, "head": head, "dep": dep},
        params={"W": None, "b": None},
        attrs={"hidden_width": hidden_width, "window": window},
        dims={"nI": None, "nO": hidden_width},
    )


def _init(model, X=None, Y=None):
    tok2vec, head, dep = (model.get_ref(n) for n in ("tok2vec", "head", "dep"))
    tok2vec.initialize(X=X)
    w = tok2vec.get_dim("nO")
    model.set_dim("nI", w)
    head.set_dim("nI", w); dep.set_dim("nI", w)
    head.initialize(); dep.initialize()
    h = model.attrs["hidden_width"]
    model.set_param("W", model.ops.alloc2f(h, h))
    model.set_param("b", model.ops.alloc1f(h))
    return model


def _forward(model, docs: List[Doc], is_train: bool):
    tok2vec, head, dep = (model.get_ref(n) for n in ("tok2vec", "head", "dep"))
    ops = model.ops
    k = model.attrs["window"]
    W, b = model.get_param("W"), model.get_param("b")
    Xs, bp_tok2vec = tok2vec(docs, is_train)
    Hs, bp_head = head(ops.flatten(Xs), is_train)
    Ds, bp_dep = dep(ops.flatten(Xs), is_train)
    lengths = [len(d) for d in docs]
    scores, cache = [], []
    off = 0
    for n in lengths:
        H, D = Hs[off:off + n], Ds[off:off + n]
        # a virtual-root row of zeros, so root scores come from the bias alone
        Hr = ops.xp.vstack([ops.alloc2f(1, H.shape[1]), H])
        S = (Hr @ W) @ D.T + (b @ D.T)[None, :]      # (n+1, n): head candidates x dependents
        mask = _window_mask(n, k).T                  # (n+1, n)
        S = ops.xp.where(mask, S, NEG)
        scores.append(S)
        cache.append((H, D, Hr, mask, n))
        off += n

    def backprop(dSs):
        dH_all = ops.alloc2f(sum(lengths), model.attrs["hidden_width"])
        dD_all = ops.alloc2f(sum(lengths), model.attrs["hidden_width"])
        dW = ops.alloc2f(*W.shape); db = ops.alloc1f(*b.shape)
        off = 0
        for dS, (H, D, Hr, mask, n) in zip(dSs, cache):
            # S[h,d] = (Hr[h] @ W + b) . D[d]
            dS = ops.xp.where(mask, dS, 0.0)
            HW = Hr @ W                               # (n+1, h)
            dHW = dS @ D                              # (n+1, h)
            dW += Hr.T @ dHW                          # HW = Hr @ W
            db += dS.sum(0) @ D                       # bias contracts over heads
            # ⚠ dD HAS TWO PATHS: through the bilinear form AND through the bias.
            dD = dS.T @ HW + ops.xp.outer(dS.sum(0), b)
            # ⚠ AND dHr NEEDS @ W.T -- dHW is the gradient wrt (Hr @ W), not wrt Hr.
            dHr = dHW @ W.T
            dH_all[off:off + n] = dHr[1:]             # row 0 is the virtual root: no token behind it
            dD_all[off:off + n] = dD
            off += n
        model.inc_grad("W", dW); model.inc_grad("b", db)
        dX = ops.unflatten(bp_head(dH_all) + bp_dep(dD_all), lengths)
        return bp_tok2vec(dX)

    return scores, backprop
