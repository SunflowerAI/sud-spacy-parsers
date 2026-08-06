#!/usr/bin/env python3
"""`sud.SplitTok2Vec.v1` — an encoder that is part LEARNED and part FROZEN SEMANTIC.

The released parsers encode with a 96-wide `MaxoutWindowEncoder` over hashed NORM/PREFIX/SUFFIX/
SHAPE, and know nothing about word meaning: `cooked` and `baked` are unrelated symbols. The obvious
fix -- spaCy's `include_static_vectors` -- was measured in this repo and rejected (`sud-md-static-
vectors`): no reliable LAS gain and `comp:obl` F consistently HURT (id -2.27, ko -4.75), at 9-16x
the model size.

This is a different shape of the same idea. Instead of adding a wide static block on top of a
full-width learned encoder, the 96 dimensions are SPLIT: `learned_width` trained as now, and
`semantic_width` taken straight from the vocab's vectors with NO projection and NO gradient. Total
width is unchanged, so the parser's downstream layers are untouched and the comparison against the
current arm is single-variable -- capacity held constant, information added.

Two consequences worth stating, because they are the point:

  * The semantic block is FROZEN and EXTERNAL, so nothing feeds back. A table derived from this
    encoder can be used to score types without the encoder having been trained on that table --
    which a combined table built from the parser's own tok2vec could not claim.
  * Any layer stacked on top can then share the parser's ONE representation, instead of shipping a
    second copy of the vectors in the vocab for its own encoder.

The vectors must already be the right width: PCA them to `semantic_width` before
`spacy init vectors`. A learned projection would defeat the purpose -- the whole claim is that these
dimensions are fixed, so the same numbers appear in the parser and in any table derived from it.

Rows for types with no vector are ZERO. That is deliberate and not the same as "unset": with floret
vectors there is no such type, and with a lookup table a zero row is the honest encoding of "no
semantic information", which the learned half can compensate for. (Cf. the unset-vs-empty MORPH bug
that cost sa 6.8 LAS -- the failure there was a value the encoder never met in TRAINING; here zero
rows occur in training and inference alike.)
"""
from typing import List

import numpy
from thinc.api import Model
from thinc.types import Floats2d

try:                                    # registered for `spacy train --code`
    from spacy import registry
    from spacy.tokens import Doc
except ImportError:                     # pragma: no cover
    registry = None


def _forward(model: Model, docs: List["Doc"], is_train: bool):
    learned = model.layers[0]
    width = model.attrs["semantic_width"]
    ops = model.ops
    Ys, bp_learned = learned(docs, is_train)
    out = []
    for doc, Y in zip(docs, Ys):
        S = ops.alloc2f(len(doc), width)
        vectors = doc.vocab.vectors
        if len(vectors):
            for i, token in enumerate(doc):
                # `has_vector` is False for an OOV key in a default-mode table and always True for
                # floret, which composes from character n-grams -- so floret gives a dense block.
                if token.has_vector:
                    v = token.vector
                    S[i] = v[:width] if v.shape[0] >= width else numpy.pad(v, (0, width - v.shape[0]))
        out.append(ops.xp.hstack((Y, S)))

    def backprop(dYs):
        # The semantic columns are frozen: slice them off and pass only the learned gradient back.
        return bp_learned([ops.as_contig(d[:, :d.shape[1] - width]) for d in dYs])

    return out, backprop


def _init(model, X=None, Y=None):
    learned = model.layers[0]
    learned.initialize(X=X)
    model.set_dim("nO", learned.get_dim("nO") + model.attrs["semantic_width"])
    return model


def SplitTok2Vec(learned: Model, semantic_width: int) -> Model[List["Doc"], List[Floats2d]]:
    return Model(
        "sud_split_tok2vec",
        _forward,
        init=_init,
        layers=[learned],
        attrs={"semantic_width": semantic_width},
        dims={"nO": None},
    )


if registry is not None:
    @registry.architectures("sud.SplitTok2Vec.v1")
    def split_tok2vec_v1(learned: Model, semantic_width: int):
        """`learned` is an ordinary spacy.Tok2Vec.v2 at the REDUCED width."""
        return SplitTok2Vec(learned, semantic_width)
