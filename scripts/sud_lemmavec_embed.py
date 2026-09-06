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

SEALING (`scripts/seal_la_lemvec_model.py`). The table travels in the MODEL'S OWN BYTES: the
payload sits in the extractor's `attrs`, which thinc serialises alongside the weights, and sealing
rewrites the stored config to `vectors = null` + `vector_dim = N` so the built model no longer needs
the .npz that was on the training machine. This was the defect that made the layer experiment-only,
and it is the same job `seal_analyser_model.py` did for the analyser table.

⚠ Build time and load time are DIFFERENT contracts and both must hold. `vectors` (a path) is for
building; `vector_dim` (an integer) is for a sealed model, whose architecture spaCy reconstructs
from the config BEFORE restoring the weights — so construction must not need the table, and only
`_lemmavec_forward` may refuse when it is missing. `resolve_vectors` is the single place that knows
this, and it refuses when BOTH are null rather than guessing a width.

`constant = true` is the capacity control: same Linear, same parameter count, every token given the
zero vector and the "no entry" flag.
"""
import pathlib
import sys
from typing import Callable, List, Optional, Tuple, Union

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


def _payload_from_path(path) -> dict:
    """The table as plain msgpack-safe values, so it travels in the MODEL'S OWN BYTES.

    thinc serialises `attrs` with the weights, which is what makes sealing possible at all — but it
    SKIPS AN UNSERIALISABLE ATTR WITHOUT SAYING SO, so the array goes in as raw bytes plus a shape
    and a dtype rather than as an ndarray. `_lemmavec_forward` refuses to run on an empty payload
    for exactly that reason.
    """
    idx, V = load_vectors(path)
    keys = [""] * len(idx)
    for k, i in idx.items():
        keys[i] = k
    V = np.ascontiguousarray(V, dtype="float32")
    return {"keys": keys, "shape": [int(V.shape[0]), int(V.shape[1])],
            "dtype": "float32", "data": V.tobytes()}


#: decoded payloads, keyed by id() with the payload itself held so the id cannot be recycled.
#: Decoding 1.6 MB on every forward pass would dominate the layer's cost.
_DECODED: dict = {}


def _decode(payload: dict):
    key = id(payload)
    hit = _DECODED.get(key)
    if hit is not None and hit[0] is payload:
        return hit[1], hit[2]
    n, d = payload["shape"]
    V = np.frombuffer(payload["data"], dtype=payload.get("dtype", "float32")).reshape(n, d)
    idx = {k: i for i, k in enumerate(payload["keys"])}
    _DECODED[key] = (payload, idx, V)
    return idx, V


def resolve_vectors(vectors, vector_dim, who: str):
    """(payload, dim) from EITHER a build-time path OR a sealed model's stored dimensionality.

    Two call sites, and the difference is the whole point of sealing. At BUILD time `vectors` names
    the .npz and the payload is read from it. After sealing, the stored config carries
    `vectors = null` and `vector_dim = N`: the architecture must then construct with an EMPTY
    payload, because spaCy rebuilds the architecture from the config BEFORE restoring the weights,
    and the real payload arrives moments later with them. Constructing must therefore not raise —
    only running on an empty payload may, which is what `_lemmavec_forward` does.
    """
    if vectors is not None:
        payload = _payload_from_path(vectors)
        return payload, payload["shape"][1]
    if vector_dim is None:
        raise ValueError(
            f"{who} needs either `vectors` (a path, at build time) or `vector_dim` (an integer, "
            f"in a sealed model). Both are null, so the block's width is unknown — refusing to "
            f"guess, because a wrong width fails far from here.")
    return {}, int(vector_dim)


def LemmaVecExtractor(payload: dict, constant: bool, dim: int):
    # `lv_dim` is carried explicitly rather than read off the array, because the capacity control
    # has no array at all and must still emit a block of the same width.
    return Model("extract_lemma_vectors", _lemmavec_forward,
                 attrs={"lv_payload": payload, "lv_constant": bool(constant), "lv_dim": int(dim)})


def _lemmavec_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    payload = model.attrs["lv_payload"]
    constant = model.attrs["lv_constant"]
    dim = int(model.attrs["lv_dim"])
    if constant:
        idx, V = {}, None
    else:
        if not payload:
            raise ValueError(
                "sud.LemmaVec*: the lemma-vector table is empty. Either it never reached the "
                "model, or it was dropped on serialisation (thinc skips an unserialisable attr "
                "without saying so). Refusing to run: every token would read as out-of-vocabulary, "
                "which loads cleanly and scores exactly like this layer's own capacity control.")
        idx, V = _decode(payload)
        if V.shape[1] != dim:
            raise ValueError(f"sud.LemmaVec*: stored vector_dim {dim} but the table is "
                             f"{V.shape[1]}-dimensional")
    out: List[Floats2d] = []
    for doc in docs:
        toks = list(doc)
        arr = np.zeros((len(toks), dim + 1), dtype="f")
        for i, tok in enumerate(toks):
            row = None if (constant or V is None) else idx.get(tok.lemma_)
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
    vector_dim: Optional[int] = None,
    constant: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    payload, dim = resolve_vectors(vectors, vector_dim, "sud.LemmaVecEmbed.v1")
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
    block = chain(LemmaVecExtractor(payload, constant, dim), list2ragged(),
                  with_array(Linear(width, dim + 1)))
    return chain(concatenate(hashed, block), max_out, ragged2list())


@registry.architectures("sud.LemmaVecFeatsEmbed.v1")
def LemmaVecFeatsEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    vectors=None,
    vector_dim: Optional[int] = None,
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

    The table TRAVELS IN THE MODEL BYTES once `scripts/seal_la_lemvec_model.py` has run over the
    arm — see the module docstring. An UNSEALED arm still carries `vectors = <path>` in its config
    and will raise on any machine without that file, which is the correct behaviour and not a
    regression: the alternative is a model that loads cleanly with an all-zero table and scores
    exactly like its own capacity control.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    payload, dim = resolve_vectors(vectors, vector_dim, "sud.LemmaVecFeatsEmbed.v1")

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
    pieces.append(chain(LemmaVecExtractor(payload, constant, dim), list2ragged(),
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
    vector_dim: Optional[int] = None,
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
    payload, dim = resolve_vectors(vectors, vector_dim, "sud.LemmaVecFeatsAgreeEmbed.v1")

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
    pieces.append(chain(LemmaVecExtractor(payload, constant, dim), list2ragged(),
                        with_array(Linear(width, dim + 1))))
    pieces.append(chain(AgreeExtractor(agree_near, agree_constant), list2ragged(),
                        with_array(Linear(width, AGREE_DIMS))))
    # one `width` per hash TABLE, plus static vectors, plus the lemma block, plus this one
    concat_size = width * (len(embeddings) + include_static_vectors + 2)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())


# --------------------------------------------------------------------------------------------
# VERB-DISTANCE AS AN INPUT, the transition-parser counterpart of the arc-factored decoder's
# `--clausegap` (train_arcfactored.py / NEGATIVE-RESULTS.md's "does this arc cross a clause
# boundary" entry).
#
# WHY. Checking actual conj:coord errors under the arc-factored decoder found the dominant signal
# is not arc length but whether a VERB/AUX sits between the candidate head and dependent: gold
# conj:coord arcs crossing a verb score 18.68% (n=653) vs 51.15% (n=1863) when they don't, and a
# follow-up check found the TRANSITION parser has the SAME qualitative weakness (66.13% vs 34.00%,
# an almost identical ~32-point relative gap) -- so this is not an arc-factored-specific hole the
# transition parser already closes via its stack state; it is a shared weakness worth targeting.
#
# WHY THIS IS A WEAKER, INDIRECT VERSION of --clausegap, not a straight port. `--clausegap` reads a
# genuinely PAIRWISE fact (how many verbs sit between THIS SPECIFIC candidate arc's two endpoints),
# which the arc-factored decoder's scorer can read directly because it scores every (head,
# dependent, label) triple explicitly. The transition parser has no equivalent insertion point --
# it scores ACTIONS from a state built out of a handful of stack/buffer token POSITIONS, each
# contributing only its own per-token vector, inside spaCy's compiled transition system. There is no
# clean way to hand it "does the candidate arc I am about to take cross a verb" without patching
# spaCy's own internals. What CAN be done at the ordinary config level is give each token its OWN
# distance to the nearest VERB/AUX in each direction, as part of its embedding -- from which the
# parser's state-composition MLP could, in principle, reconstruct crossing information (a verb sits
# between two tokens i < j exactly when i's distance-to-next-verb or j's distance-to-previous-verb
# is <= j - i), but only if it learns to compose two different stack positions' own features that
# way. This is real information, not zero, but a weaker bet than --clausegap's direct pairwise
# read -- worth measuring, not assuming.
#
# THE DIMENSIONS. Five:
#   0   reciprocal distance to the nearest PRECEDING VERB/AUX (1/(1+d)), 0 if none exists
#   1   reciprocal distance to the nearest FOLLOWING VERB/AUX (1/(1+d)), 0 if none exists
#   2   no VERB/AUX precedes this token AT ALL, anywhere in the sentence (the "no info" bit --
#       0.0 at dim 0 must not be confused with "a verb is infinitely close", the same unset-vs-
#       empty distinction CLAUDE.md already records costing Sanskrit 6.8 LAS by a different route)
#   3   no VERB/AUX follows this token AT ALL
#   4   this token IS a VERB/AUX itself (its own distances above measure the NEAREST OTHER verb,
#       never itself, since the running pointer only updates AFTER being read for the current token)
#
# `verbdist_constant = true` is the capacity control: identical Linear, identical parameter count,
# every token handed five zeros -- POS is never read, so this measures the block's ~480 extra
# parameters against its actual information.
VERBDIST_DIMS = 5


def VerbDistExtractor(constant: bool):
    return Model("extract_verbdist", _verbdist_forward, attrs={"vd_constant": bool(constant)})


def _verbdist_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    constant = model.attrs["vd_constant"]
    out: List[Floats2d] = []
    for doc in docs:
        n = len(doc)
        arr = np.zeros((n, VERBDIST_DIMS), dtype="f")
        if not constant:
            is_verb = [1 if t.pos_ in ("VERB", "AUX") else 0 for t in doc]
            last_verb = -1
            for i in range(n):
                if last_verb < 0:
                    arr[i, 2] = 1.0
                else:
                    arr[i, 0] = 1.0 / (1.0 + (i - last_verb))
                if is_verb[i]:
                    last_verb = i
            next_verb = -1
            for i in range(n - 1, -1, -1):
                if next_verb < 0:
                    arr[i, 3] = 1.0
                else:
                    arr[i, 1] = 1.0 / (1.0 + (next_verb - i))
                if is_verb[i]:
                    next_verb = i
            for i in range(n):
                if is_verb[i]:
                    arr[i, 4] = 1.0
        out.append(model.ops.asarray2f(arr))
    backprop: Callable[[List[Floats2d]], List] = lambda d: []
    return out, backprop


@registry.architectures("sud.LemmaVecFeatsVerbDistEmbed.v1")
def LemmaVecFeatsVerbDistEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    vectors=None,
    vector_dim: Optional[int] = None,
    constant: bool = False,
    feats: List[str] = [],
    feat_rows: List[int] = [],
    verbdist_constant: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    """`sud.LemmaVecFeatsEmbed.v1` plus the verb-distance block documented above.

    Everything else is bit-for-bit the same layer, including the seeding order of the hash tables,
    so an arm built on this and an arm built on `sud.LemmaVecFeatsEmbed.v1` differ in the five
    dimensions and in nothing else."""
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    payload, dim = resolve_vectors(vectors, vector_dim, "sud.LemmaVecFeatsVerbDistEmbed.v1")

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
    pieces.append(chain(LemmaVecExtractor(payload, constant, dim), list2ragged(),
                        with_array(Linear(width, dim + 1))))
    pieces.append(chain(VerbDistExtractor(verbdist_constant), list2ragged(),
                        with_array(Linear(width, VERBDIST_DIMS))))
    concat_size = width * (len(embeddings) + include_static_vectors + 2)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())
