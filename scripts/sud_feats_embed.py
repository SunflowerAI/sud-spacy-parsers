#!/usr/bin/env python3
"""`sud.MultiHashEmbedFeats.v1` — MultiHashEmbed plus one table per MORPHOLOGICAL FEATURE.

WHY THIS EXISTS. `MORPH` is a plain column in spaCy's feature extractor, and its value is a hash of
the WHOLE normalised FEATS bundle. So a model conditioned on `MORPH` sees `Case=Nom|Number=Sing` and
`Case=Nom|Number=Plur` as two unrelated symbols: nothing tells it they share a case, and a bundle it
never saw in training is simply a new symbol with no decomposition to fall back on. For a target
like Latin's composite XPOS -- whose morphological tail IS a restatement of FEATS, category by
category -- that is the crudest possible way to offer the information.

This layer offers it decomposed: one hash-embedded table per configured feature, each column holding
`hash_string("Case=Nom")` for that token and `hash_string("Case=")` where the token has no value.
Each feature therefore gets its own small vocabulary and its own rows, and an unseen BUNDLE still
lands on seen values in every column.

WHAT MOTIVATED IT (NEGATIVE-RESULTS.md, "Making XPOS downstream of UPOS and FEATS"). Conditioning
the tagger on a single hashed `MORPH` bundle cost 0.29-0.37 dev tag_acc on ar/zh/en against a
capacity control. But complementary information demonstrably exists: 0.6-3.2 % of test tokens are
ones the tagger gets wrong and a majority-class map on PREDICTED upos+feats gets right, and the
either-one-right ceiling sits 2-6 points above the tagger. The decomposition is the untried way to
reach it.

⚠ AN UNSET MORPH AND AN EMPTY ONE MUST NOT BE DIFFERENT INPUTS HERE. `set_morph("")` gives key 456
and an unset token key 0 (CLAUDE.md; it once cost sa 6.8 LAS). This layer reads each feature
INDIVIDUALLY, so both produce `Case=` in every column and land on the same row -- the distinction
cannot leak in. That is by construction, not by luck, and it is why the cache is keyed on
`morph.key` yet still safe.

SAFETY. With `feats=[]` this is behaviourally identical to `spacy.MultiHashEmbed.v2`, verified
byte-for-byte by `scripts/check_feats_embed.py`, so an arm can be switched over before any feature
is added and the change stays single-variable. Used in the freeze recipe it cannot regress parsing:
only the component's own standalone encoder sees it.

SIZING. Cost is `rows * width * 4` bytes. A feature's value set is TINY (Case ~8, Number ~3,
VerbForm ~6), so rows are cheap -- `scripts/build_feats_inventory.py` derives both the list and the
row counts from the treebank rather than guessing.

Config usage (drops straight into a `spacy.Tok2Vec.v2` `embed` slot):

    [components.tagger.model.tok2vec.embed]
    @architectures = "sud.MultiHashEmbedFeats.v1"
    width = 96
    attrs = ["NORM","PREFIX","SUFFIX","SHAPE","POS"]
    rows  = [5000, 2500, 2500, 2500, 100]
    feats      = ["Case","Number","Gender","VerbForm","Tense","Mood","Voice","Person"]
    feat_rows  = [32, 16, 16, 32, 32, 32, 16, 16]
    include_static_vectors = false
"""
from typing import Callable, List, Tuple, Union

from spacy.strings import hash_string
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.types import Floats2d, Ints2d, Ragged

from spacy.ml.staticvectors import StaticVectors
from thinc.layers import HashEmbed

# (morph key, feature tuple) -> per-feature hashes. Module level, not a Model attr: thinc would try
# to serialise a Model attr, and this is a pure cache. Safe across vocabs and processes for the same
# reason the affix cache is: `MorphAnalysis.key` comes from `StringStore.add` of the NORMALISED
# feats string, i.e. `hash_string`, which is a global murmur hash -- so the key determines the
# bundle globally, exactly as `token.orth` determines the string.
_FEATS_CACHE = {}


def _feat_keys(token, feats: Tuple[str, ...]):
    ck = (token.morph.key, feats)
    got = _FEATS_CACHE.get(ck)
    if got is None:
        morph = token.morph
        # `morph.get` returns a LIST -- a feature may be multi-valued (`Case=Nom,Acc`), and joining
        # keeps that faithful instead of silently taking the first value.
        got = tuple(hash_string(f"{f}={','.join(morph.get(f))}") for f in feats)
        _FEATS_CACHE[ck] = got
    return got


def FeatsFeatureExtractor(columns, feats):
    """spaCy's FeatureExtractor with extra columns read off `token.morph` at forward time.

    Column order is `columns` first, then one column per entry of `feats` -- the embedding tables
    downstream index by position, so this order is load-bearing.
    """
    return Model("extract_features_feats", _feats_forward,
                 attrs={"columns": list(columns), "feats": list(feats)})


def _feats_forward(model: Model, docs, is_train: bool) -> Tuple[List[Ints2d], Callable]:
    columns = model.attrs["columns"]
    feats = tuple(model.attrs["feats"])
    xp = model.ops.xp
    features: List[Ints2d] = []
    for doc in docs:
        # Spans arrive here too (spaCy slices the parent doc's array), same as spacy's extractor.
        # Both Doc and Span iterate over their own tokens, so the feature loop needs no special case.
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start:doc.end]
        if attrs.ndim == 1:
            attrs = attrs.reshape((attrs.shape[0], 1))
        if feats:
            extra = xp.zeros((len(doc), len(feats)), dtype="uint64")
            for i, token in enumerate(doc):
                extra[i] = _feat_keys(token, feats)
            attrs = xp.hstack((attrs.astype("uint64"), extra))
        features.append(model.ops.asarray2i(attrs, dtype="uint64"))

    backprop: Callable[[List[Ints2d]], List] = lambda d_features: []
    return features, backprop


@registry.architectures("sud.MultiHashEmbedFeats.v1")
def MultiHashEmbedFeats(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    feats: List[str] = [],
    feat_rows: List[int] = [],
) -> Model[List[Doc], List[Floats2d]]:
    """`spacy.MultiHashEmbed.v2` with one extra hash-embedded table per configured FEATS key.

    With `feats=[]` this is behaviourally identical to `MultiHashEmbed`, so an arm can be switched
    over before any feature is added and the change stays single-variable.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if any(r < 1 for r in feat_rows):
        raise ValueError("feature rows must be >= 1")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")

    all_rows = list(rows) + list(feat_rows)
    # seed 7 and the same increment order as MultiHashEmbed / MultiHashEmbedAffix, so the first
    # len(attrs) tables are seeded identically to stock and `feats=[]` is EXACTLY equivalent.
    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(all_rows))]
    concat_size = width * (len(embeddings) + include_static_vectors)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True)
    )
    extractor = FeatsFeatureExtractor(attrs, feats)
    if include_static_vectors:
        feature_extractor: Model[List[Doc], Ragged] = chain(
            extractor,
            list2ragged(),
            with_array(concatenate(*embeddings)),
        )
        model = chain(
            concatenate(feature_extractor, StaticVectors(width, dropout=0.0)),
            max_out,
            ragged2list(),
        )
    else:
        model = chain(
            extractor,
            list2ragged(),
            with_array(concatenate(*embeddings)),
            max_out,
            ragged2list(),
        )
    return model


@registry.architectures("sud.Tok2VecPlusFeats.v1")
def Tok2VecPlusFeats(
    tok2vec: Model,
    feats_embed: Model,
) -> Model[List[Doc], List[Floats2d]]:
    """Concatenate an existing tok2vec with a morphology side-channel, at the TOP.

    WHY THE POSITION MATTERS, and it is the whole point of this layer. `MultiHashEmbedFeats` above
    puts POS/MORPH in at the BOTTOM, as extra columns of the embed -- so a MaxoutWindowEncoder of
    depth 4 then convolves them over a +-4 token window, and each token's tag comes to depend on its
    NEIGHBOURS' predicted morphology as well as its own. Predicted morphology is only 0.75-0.99
    exact, so that smears an already noisy channel across the sentence, and it also means the token
    representation is rebuilt from scratch instead of reusing the co-trained shared encoder.

    Here the side channel enters AFTER the encoder, immediately below the softmax. Two consequences:
    the inner `tok2vec` can be a plain `spacy.Tok2VecListener.v1` on the FROZEN shared encoder, so
    the tagger's token representation is EXACTLY the one the released tagger already has and the
    experiment is single-variable; and a token's own morphology reaches only that token's decision.

    `feats_embed` is any Model[List[Doc], List[Floats2d]] -- in practice `sud.MultiHashEmbedFeats.v1`
    with `attrs = ["POS"]` and one table per FEATS key, which is already covered by
    scripts/check_feats_embed.py. Output width is the sum of the two, which `spacy.Tagger.v2` reads
    off `nO` when it sizes the Softmax.

    Config usage:

        [components.tagger.model.tok2vec]
        @architectures = "sud.Tok2VecPlusFeats.v1"

        [components.tagger.model.tok2vec.tok2vec]
        @architectures = "spacy.Tok2VecListener.v1"
        width = 96
        upstream = "tok2vec"

        [components.tagger.model.tok2vec.feats_embed]
        @architectures = "sud.MultiHashEmbedFeats.v1"
        width = 32
        attrs = ["POS"]
        rows  = [100]
        feats = ["Case", "Number"]
        feat_rows = [16, 16]
        include_static_vectors = false
    """
    return concatenate(tok2vec, feats_embed)
