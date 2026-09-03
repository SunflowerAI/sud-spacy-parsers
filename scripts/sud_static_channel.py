#!/usr/bin/env python3
"""`sud.StaticVecChannel.v1` — static vectors as a TOP-injected side channel.

WHAT IT IS FOR. The lzh parser reads a `spacy.Tok2VecListener.v1` on the shared encoder, and that
encoder's embed has `include_static_vectors = false`. Turning it on there would put the vectors in
at the BOTTOM: a `MaxoutWindowEncoder` of depth 4 then convolves them over a ±4 token window, and it
requires retraining the shared encoder, which on this arm means retraining the whole layer stack
(`seg` is a BASE recipe). This layer is the other injection point — `concatenate`d with the listener
by `sud.Tok2VecPlusFeats.v1`, so the vectors reach the parser's decision and nothing else, and the
encoder underneath stays FROZEN and byte-identical.

⚠ INJECTION POINT IS THE MEASURED VARIABLE, and it is not a small one. NEGATIVE-RESULTS.md records
the same question for the conditioned XPOS tagger: identical information in the embed cost −0.3 to
−0.6, moved above the encoder it helped everywhere — while bundle-vs-per-feature representation was
a wash at ≤ 0.10. **Where a noisy channel enters matters more than how it is represented.**

⚠ AND READ THE PRE-FLIGHT ARITHMETIC BEFORE READING ANY HEADLINE FROM AN ARM THAT USES THIS. A
static vector can only inform a decision the FORM does not already settle, and on the Kyoto test set
unseen forms are 1.15 % of tokens and forms seen twice or fewer 2.19 %. Even +15 LAS on the unseen
slice is +0.17 aggregate, against a seed spread of ~0.5 on this arm family. **Score it with
`scripts/eval_lex_slices.py`, on the slice its own rationale names**; the headline cannot resolve it
and neither can more seeds.

`spacy.StaticVectors.v2` emits `Ragged`; `Tok2VecPlusFeats` concatenates `List[Floats2d]`, so the
only work here is the conversion. `nM` (the table's own width) is inferred at initialize from the
vectors actually loaded, so this layer does not need to know it.

Config usage:

    [components.parser.model.tok2vec]
    @architectures = "sud.Tok2VecPlusFeats.v1"

    [components.parser.model.tok2vec.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = 96
    upstream = "tok2vec"

    [components.parser.model.tok2vec.feats_embed]
    @architectures = "sud.StaticVecChannel.v1"
    width = 96
"""
from typing import Optional

from spacy.ml.staticvectors import StaticVectors
from spacy.util import registry
from thinc.api import Model, chain
from thinc.layers import ragged2list


# ⚠ `dropout` MUST be Optional[float], not `float = None`. confection validates the config against
# the annotation, so a bare `float` default of None fails at `Initializing pipeline` with
# "None is not <class 'float'>" — before a single batch runs, but after the run looks started.
@registry.architectures("sud.StaticVecChannel.v1")
def StaticVecChannel(width: int, key_attr: str = "ORTH",
                     dropout: Optional[float] = None) -> Model:
    """A `List[Doc] -> List[Floats2d]` channel holding nothing but the projected static vector.

    The vector TABLE is never a parameter — spaCy does not update `vocab.vectors` — so the only
    thing this layer trains is the projection into `width`. That is deliberate and is the whole
    point of a distilled channel: frozen knowledge, one small learned adaptor.
    """
    return chain(StaticVectors(width, key_attr=key_attr, dropout=dropout), ragged2list())
