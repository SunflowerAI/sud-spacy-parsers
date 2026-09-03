#!/usr/bin/env python
"""`sud.ArcStructureEmbed.v1` — the parser's ARC STRUCTURE as a side channel for a tagger.

WHY IT EXISTS. lzh's dominant residual error is zero derivation (VERB<->NOUN, ~36 % of all UPOS
errors): whether 歡 is "joy" or "rejoice" HERE. Six independent routes failed on it — local form
features, type-level SikuBERT, token-level contextual SikuBERT, a random forest over all of them,
gold neighbouring TAGS (+0.00 on this class), and the existing DEP-only channel. Arc structure is
the one probe that came back positive, and by a wide margin:

    morphologiser encoder alone            91.12 %   VERB/NOUN 89.82 %
    + PREDICTED arc structure              91.55 %   VERB/NOUN 90.24 %   (+0.42)
    + GOLD arc structure                   94.51 %   VERB/NOUN 93.99 %   (+4.17)

The reason it works where the token's own DEP does not: a noun and a verb differ in what they
GOVERN, not in what they are governed by. 歡-as-verb takes an object; 歡-as-noun does not.

⚠ IT IS PARTLY CIRCULAR AND THAT CAPS IT. The parser assigned those arcs having already committed
to a reading, so predicted arcs recover only ~10 % of the gold-arc gain. The ceiling is not
reachable by a better channel — only by deciding tags and arcs together.

⚠ TRAIN WITH THE PARSER IN `annotating_components`. Trained on gold arcs, this channel would learn
to trust a signal it never meets at runtime — the classic skew, and here it is worth 3 points.
"""
from typing import List

import numpy as np
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Maxout, Model, chain, list2ragged, ragged2list, with_array
from thinc.types import Floats2d


def _featurise(deps: List[str], max_dist: int):
    di = {d: i for i, d in enumerate(deps)}
    K = len(deps)
    nF = K * 2 + 4

    def forward(model, docs: List[Doc], is_train: bool):
        out = []
        for doc in docs:
            X = np.zeros((len(doc), nF), dtype="float32")
            kids = {}
            for t in doc:
                if t.head.i != t.i:
                    kids.setdefault(t.head.i, []).append(t.dep_)
            for i, t in enumerate(doc):
                # what governs it
                X[i, di.get(t.dep_, K - 1)] = 1.0
                # ⚠ WHAT IT GOVERNS — the half that carries the noun/verb signal
                for d in kids.get(i, ()):
                    if d in di:
                        X[i, K + di[d]] = 1.0
                X[i, -4] = np.sign(t.head.i - i)
                X[i, -3] = min(abs(t.head.i - i), max_dist) / max_dist
                X[i, -2] = min(len(kids.get(i, ())), 5) / 5.0
                X[i, -1] = 1.0 if t.head.i == t.i else 0.0
            out.append(model.ops.asarray2f(X))
        # Docs are not differentiable; nothing flows back past the featuriser.
        return out, lambda dY: []

    return Model("arc_featurise", forward, dims={"nO": nF}), nF


@registry.architectures("sud.ArcStructureEmbed.v1")
def ArcStructureEmbed(width: int, deps: List[str], max_dist: int = 9) -> Model[List[Doc], List[Floats2d]]:
    if len(set(deps)) != len(deps):
        raise ValueError(f"duplicate deprel in {deps}")
    feat, nF = _featurise(deps, max_dist)
    return chain(feat, list2ragged(),
                 with_array(Maxout(width, nF, nP=3, dropout=0.0, normalize=True)),
                 ragged2list())
