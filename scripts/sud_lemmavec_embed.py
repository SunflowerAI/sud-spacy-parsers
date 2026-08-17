#!/usr/bin/env python3
"""`sud.LemmaVecEmbed.v1` — `MultiHashEmbed` plus a DISTRIBUTIONAL lemma-vector block.

THE QUESTION IT ANSWERS. The oracle grid measured gold lemma IDENTITY, hash-embedded, at +2.22 LAS.
Identity cannot generalise: `gam`, `yā` and `vraj` are three unrelated symbols to a hash table. A
distributional vector space can, and `vectors_sa_lemma_ppmi.vec` demonstrably has that structure
(`gam -> yā, vraj, prāp`, sharing no characters). So the contrast this layer exists for is SAME GOLD
LEMMA, hash encoding vs vector encoding — which isolates whether the generalisation is worth
anything to a parser, with the lemma channel's quality held fixed.

⚠ A PRIOR ATTEMPT AT THIS QUESTION ALREADY CAME BACK NEGATIVE, on a different target. Asked whether
these vectors separate `comp:obl` from `mod` beyond the (subtype x Case) rule table, they add +0.2
to +1.2 points and NON-MONOTONICALLY in dimensionality (96-d 0.9141 > 300-d 0.9093 > 192-d 0.9045),
which is what fitting noise looks like. That tested LABELLING, not attachment, which is why this
layer exists rather than the matter being closed.

⚠ EXPERIMENT-ONLY, NOT SHIPPABLE AS WRITTEN. The table is held module-level and only its PATH goes
into the model's attrs, so a saved model depends on a file that existed on the training machine —
exactly the defect `seal_analyser_model.py` had to fix for the analyser table. That is acceptable
here because every arm using this layer is an ORACLE (it reads gold LEMMA, which no inference path
supplies). If it ever earns its way into a wheel, the table must travel in the bytes.

`constant = true` is the capacity control: same Linear, same parameter count, every token given the
zero vector and the "no entry" flag.
"""
import pathlib
import sys
from typing import Callable, List, Tuple, Union

import numpy as np
from spacy.ml.models.tok2vec import FeatureExtractor
from spacy.ml.staticvectors import StaticVectors
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import HashEmbed
from thinc.types import Floats2d, Ragged

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sud_feats_embed import FeatsFeatureExtractor  # noqa: E402

_TABLES = {}


def load_vectors(path):
    p = str(path)
    if p not in _TABLES:
        f = pathlib.Path(p)
        if not f.exists():
            raise ValueError(
                f"sud.LemmaVecEmbed.v1: lemma vectors {p} not found. Build them with "
                f"scripts/build_lemma_vectors.py. Refusing to fall back to all-zeros, which would "
                f"load cleanly and score exactly like this layer's own capacity control.")
        d = np.load(f, allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        _TABLES[p] = ({k: i for i, k in enumerate(keys)}, d["vectors"].astype("f"))
    return _TABLES[p]


def LemmaVecExtractor(path, constant: bool):
    return Model("extract_lemma_vectors", _lemmavec_forward,
                 attrs={"lv_path": str(path), "lv_constant": bool(constant)})


def _lemmavec_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    idx, V = load_vectors(model.attrs["lv_path"])
    constant = model.attrs["lv_constant"]
    dim = V.shape[1]
    out: List[Floats2d] = []
    for doc in docs:
        toks = list(doc)
        arr = np.zeros((len(toks), dim + 1), dtype="f")
        for i, tok in enumerate(toks):
            row = None if constant else idx.get(tok.lemma_)
            if row is None:
                arr[i, dim] = 1.0          # the one "no entry" flag; OOV and control share it
            else:
                arr[i, :dim] = V[row]
        out.append(model.ops.asarray2f(arr))
    backprop: Callable[[List[Floats2d]], List] = lambda d: []
    return out, backprop


@registry.architectures("sud.LemmaVecEmbed.v1")
def LemmaVecEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    vectors=None,
    constant: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if vectors is None:
        raise ValueError("sud.LemmaVecEmbed.v1 needs `vectors`")
    _, V = load_vectors(vectors)
    dim = V.shape[1]
    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(attrs))]
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, width * (len(embeddings) + 1), nP=3, dropout=0.0, normalize=True))
    hashed = chain(FeatureExtractor(list(attrs)), list2ragged(),
                   with_array(concatenate(*embeddings)))
    block = chain(LemmaVecExtractor(vectors, constant), list2ragged(),
                  with_array(Linear(width, dim + 1)))
    return chain(concatenate(hashed, block), max_out, ragged2list())


@registry.architectures("sud.LemmaVecFeatsEmbed.v1")
def LemmaVecFeatsEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    vectors=None,
    constant: bool = False,
    feats: List[str] = [],
    feat_rows: List[int] = [],
) -> Model[List[Doc], List[Floats2d]]:
    """`sud.MultiHashEmbedFeats.v1` AND the distributional lemma-vector block, in one embed.

    The two halves answer different questions and neither substitutes for the other. The per-feature
    hash tables give MORPHOLOGY a dimension per category, so that `Case=Nom|Number=Sing` and
    `Case=Nom|Number=Plur` share a case rather than arriving as two unrelated symbols — which is
    what a parser needs, since agreement and government are relations BETWEEN tokens and cannot be
    read off two opaque bundle hashes. The lemma block gives LEXICAL SEMANTICS a geometry, so that
    two verbs which govern the same case can be near each other without sharing a character.

    Latin is the case where the pair has the best claim. Its morphology is rich enough that the
    whole-bundle `MORPH` hash was measurably worthless — `configs/config_la_morphfirst.cfg` put a
    frozen morphologiser at the front of the released arm and reached LAS 0.7256 against a CAPACITY
    CONTROL's 0.7255, i.e. the entire gain was the extra embedding rows and none of it the
    morphology. Decomposition is the untried half of that experiment; the lemma block is the other.

    ``constant = true`` is the block's capacity control: identical Linear, identical parameter
    count, every token handed the zero vector and the "no entry" flag. Set ``feats = []`` as well
    and this is `spacy.MultiHashEmbed.v2` plus a dead block — the honest control for the pair.

    ⚠ THE TABLE DOES NOT TRAVEL IN THE MODEL BYTES, inherited from `sud.LemmaVecEmbed.v1` above: the
    path goes into the attrs and the array is held module-level. That is fine for an experiment and
    is NOT fine for a wheel; sealing it is the same job `seal_analyser_model.py` did for the
    analyser table, and it has to be done before this ships.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    if vectors is None:
        raise ValueError("sud.LemmaVecFeatsEmbed.v1 needs `vectors`")
    _, V = load_vectors(vectors)
    dim = V.shape[1]

    all_rows = list(rows) + list(feat_rows)
    seed = 7                       # same seeding order as MultiHashEmbed, so the columns line up

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(all_rows))]
    pieces = [chain(FeatsFeatureExtractor(attrs, feats), list2ragged(),
                    with_array(concatenate(*embeddings)))]
    if include_static_vectors:
        pieces.append(StaticVectors(width, dropout=0.0))
    pieces.append(chain(LemmaVecExtractor(vectors, constant), list2ragged(),
                        with_array(Linear(width, dim + 1))))
    # The hashed piece is itself a concatenation of one `width`-wide table per column, so the
    # Maxout's input is one `width` per TABLE plus one for the static vectors and one for the
    # lemma block -- not one per piece.
    concat_size = width * (len(embeddings) + include_static_vectors + 1)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())
