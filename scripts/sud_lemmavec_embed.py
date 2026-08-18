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


# --------------------------------------------------------------------------------------------
# AGREEMENT AS AN INPUT, the Latin counterpart of `configs/config_sa_multitask_agree.cfg`.
#
# WHY A BLOCK AND NOT JUST THE PER-FEATURE TABLES. `sud.LemmaVecFeatsEmbed.v1` already gives the
# parser a dimension per morphological category, which lets it know that this token is accusative.
# Agreement is not a property of a token: it is a RELATION between two, and a per-token embedding
# cannot express one. The parser would have to reconstruct "do these two share a case?" from two
# independently-hashed vectors read at different state positions, through a Maxout that only ever
# sees them summed. So the comparison is done here, where both tokens are in hand, and handed over
# already computed.
#
# WHAT MAKES IT WORTH DOING FOR LATIN and not merely worth copying. Measured by
# `scripts/check_la_agreement_signal.py` on the test set: gold ADJ/DET/NUM --mod|det--> nominal arcs
# are 93.5 % case/number/gender-compatible, against 13.6 % for a random nominal within three tokens
# that is NOT the head -- a gap of 79.9 points, where the Sanskrit block was built on 24.1. Under
# the PREDICTED morphology a shippable arm actually reads, the positives fall to 73.4 % but the
# negatives barely move (12.7 %), so the morphologiser's errors cost recall and not precision, and
# 60.8 points of the gap survive. That is the number this block is betting on.
#
# THE DIMENSIONS. Twelve, and the last four are the Latin-specific ones:
#   0-7   compatible with the token at offset -4..-1, +1..+4      (local agreement)
#   8,9   compatible with ANY token within `near` to the left / right
#   10    how many tokens in that window are compatible, scaled   (how ambiguous the signal is)
#   11    this token declares no Case/Number/Gender at all        (see below)
# 8-10 exist because of hyperbaton: `magnam ... urbem` is the construction Latin discontinuity is
# MADE of, and a +-4 window cannot see across one. 10 matters because a lone compatible nominal and
# nine of them are opposite evidence, and dims 8-9 report both as 1.0.
#
# DIM 11 IS NOT OPTIONAL. The frozen morphologiser sets FEATS on ~68 % of tokens, so a third of them
# have no agreement features at all -- and with dims 0-10 left at zero, "nothing agrees with me" and
# "nothing is known about me" would be the identical input. That is the same distinction CLAUDE.md
# records costing Sanskrit 6.8 LAS between an unset MORPH and an empty one, arriving by a different
# route. Sanskrit hit it too, and its comment gives the same reason: an unknown form must not be
# encoded as incompatible with everything.
#
# `agree_constant = true` is the capacity control: identical Linear, identical parameter count,
# every token handed twelve zeros.
AGREE_KEYS = ("Case", "Number", "Gender")
AGREE_DIMS = 12
_BITS: dict = {}


def _mask(tok, keys) -> tuple:
    """One integer bitmask per key, so compatibility is three ANDs rather than three set builds.

    A multi-valued feature (`Case=Nom,Acc`) sets several bits, which makes intersection the right
    operation for free: an underspecified form is compatible with either reading, and that is what
    the annotation means. A key the token does not declare gets 0, and any 0 makes the token
    UNKNOWN rather than incompatible.
    """
    out = []
    for k in keys:
        vals = tok.morph.get(k)
        if not vals:
            return ()
        m = 0
        for v in vals:
            b = _BITS.get((k, v))
            if b is None:
                b = _BITS[(k, v)] = 1 << len(_BITS)
            m |= b
        out.append(m)
    return tuple(out)


def AgreeExtractor(near: int, constant: bool):
    return Model("extract_agreement", _agree_forward,
                 attrs={"ag_near": int(near), "ag_constant": bool(constant)})


def _agree_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    near = model.attrs["ag_near"]
    constant = model.attrs["ag_constant"]
    out: List[Floats2d] = []
    for doc in docs:
        n = len(doc)
        arr = np.zeros((n, AGREE_DIMS), dtype="f")
        if not constant:
            masks = [_mask(t, AGREE_KEYS) for t in doc]
            for i, mi in enumerate(masks):
                if not mi:
                    arr[i, 11] = 1.0                 # unknown, NOT incompatible
                    continue
                lo, hi = max(0, i - near), min(n, i + near + 1)
                cnt = 0
                for j in range(lo, hi):
                    if j == i:
                        continue
                    mj = masks[j]
                    if not mj or not (mi[0] & mj[0] and mi[1] & mj[1] and mi[2] & mj[2]):
                        continue
                    cnt += 1
                    d = j - i
                    if -4 <= d <= 4:
                        arr[i, d + 4 if d < 0 else d + 3] = 1.0
                    arr[i, 8 if d < 0 else 9] = 1.0
                arr[i, 10] = min(cnt, 8) / 8.0
        out.append(model.ops.asarray2f(arr))
    backprop: Callable[[List[Floats2d]], List] = lambda d: []
    return out, backprop


@registry.architectures("sud.LemmaVecFeatsAgreeEmbed.v1")
def LemmaVecFeatsAgreeEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    vectors=None,
    constant: bool = False,
    feats: List[str] = [],
    feat_rows: List[int] = [],
    agree_near: int = 20,
    agree_constant: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    """`sud.LemmaVecFeatsEmbed.v1` plus the agreement-compatibility block documented above.

    Everything else is bit-for-bit the same layer, including the seeding order of the hash tables,
    so an arm built on this and an arm built on `sud.LemmaVecFeatsEmbed.v1` differ in the twelve
    dimensions and in nothing else.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    if vectors is None:
        raise ValueError("sud.LemmaVecFeatsAgreeEmbed.v1 needs `vectors`")
    _, V = load_vectors(vectors)
    dim = V.shape[1]

    all_rows = list(rows) + list(feat_rows)
    seed = 7

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
    pieces.append(chain(AgreeExtractor(agree_near, agree_constant), list2ragged(),
                        with_array(Linear(width, AGREE_DIMS))))
    # one `width` per hash TABLE, plus static vectors, plus the lemma block, plus this one
    concat_size = width * (len(embeddings) + include_static_vectors + 2)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())
