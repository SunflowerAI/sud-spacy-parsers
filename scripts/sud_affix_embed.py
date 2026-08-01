#!/usr/bin/env python3
"""`sud.MultiHashEmbedAffix.v1` — MultiHashEmbed plus per-component affix windows.

WHY THIS EXISTS. spaCy's `PREFIX`/`SUFFIX` are entries in `lex_attr_getters` (`string[0]` and
`string[-3:]`), i.e. **lexeme** attributes: one value per vocabulary entry, shared by every
component in the language. So the only way to give the morphologiser a longer suffix window used to
be to widen `Sanskrit.Defaults` — which also widened it for the tagger, parser and lemmatiser, and
that ablation regressed LAS by 2.9 (CLAUDE.md, "NEGATIVE RESULT: do NOT widen sa PREFIX/SUFFIX").
That note ends by naming the missing piece: "it cannot be tuned per component without a custom embed
layer that hashes `token.text[-k:]` at forward time". This is that layer.

WHY IT MATTERS FOR SANSKRIT. On the sa training corpus the conditional entropy of the FEATS bundle
given the suffix falls 1.74 -> 0.75 -> 0.56 bits at k = 3 -> 5 -> 6, and a majority-class-per-suffix
lookup goes 60.0 % -> 73.3 % -> 73.9 % exact-bundle on test. The features the real morphologiser is
worst at are exactly the ones whose evidence sits outside a 3-character window: Voice (F .578;
passive `-yate`), VerbForm (.766; `-mānaḥ`, `-antaḥ`), Tense (.869; future `-ṣyati`).

A *curated inventory* of real Sanskrit inflectional endings — the intuitive fix — was simulated and
**loses**: 92-243 entries score 47-55 % against plain `form[-3:]`'s 60.0 %, and it takes ~630 entries
to draw level. The signal is window LENGTH, not linguistic curation, because real surface forms carry
stem-class and sandhi cues in the pre-desinential characters that a clean morpheme list discards.

SAFETY. Used in the morph/lemma freeze recipe, this cannot regress parsing: `frozen_components`
holds tok2vec/tagger/parser, so only the component's own standalone encoder sees the new feature,
and the lexeme-level `SUFFIX` stays at spaCy's default 3 — which is what the frozen components were
trained on, so they stay in distribution.

SIZING. Cost is `rows * width * 4` bytes in the wheel. Distinct values on sa train: k=4 7 188,
k=5 15 594, k=6 23 434 (vs the current SUFFIX table's 1 000-2 500 rows) — under-provisioning `rows`
is the most likely way to mask a real gain.

Config usage (drops straight into a `spacy.Tok2Vec.v2` `embed` slot):

    [components.morphologizer.model.tok2vec.embed]
    @architectures = "sud.MultiHashEmbedAffix.v1"
    width = 64
    attrs = ["NORM","PREFIX","SUFFIX","SHAPE","MORPH"]
    rows  = [2000, 1000, 1000, 1000, 64]
    suffixes    = [5]
    suffix_rows = [8000]
    prefixes    = []
    prefix_rows = []
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

# (orth, k, is_suffix) -> uint64. Module level, not a Model attr: thinc would try to serialise a
# Model attr, and this is a pure cache. Safe across vocabs and processes because `token.orth` IS
# `hash_string(token.text)`, so the id determines the string globally.
_AFFIX_CACHE = {}


def _affix_key(token, k, suffix):
    ck = (token.orth, k, suffix)
    got = _AFFIX_CACHE.get(ck)
    if got is None:
        text = token.text
        got = hash_string(text[-k:] if suffix else text[:k])
        _AFFIX_CACHE[ck] = got
    return got


def AffixFeatureExtractor(columns, suffixes, prefixes):
    """spaCy's FeatureExtractor with extra columns computed from the token string at forward time.

    Column order is `columns` first, then one column per entry of `suffixes`, then one per
    `prefixes` — the embedding tables downstream index by position, so this order is load-bearing.
    """
    return Model("extract_features_affix", _affix_forward,
                 attrs={"columns": list(columns), "suffixes": list(suffixes),
                        "prefixes": list(prefixes)})


def _affix_forward(model: Model, docs, is_train: bool) -> Tuple[List[Ints2d], Callable]:
    columns = model.attrs["columns"]
    suffixes = model.attrs["suffixes"]
    prefixes = model.attrs["prefixes"]
    xp = model.ops.xp
    features: List[Ints2d] = []
    for doc in docs:
        # Spans arrive here too (spaCy slices the parent doc's array), same as spacy's extractor.
        # Both Doc and Span iterate over their own tokens, so the affix loop needs no special case.
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start:doc.end]
        if attrs.ndim == 1:
            attrs = attrs.reshape((attrs.shape[0], 1))
        if suffixes or prefixes:
            extra = xp.zeros((len(doc), len(suffixes) + len(prefixes)), dtype="uint64")
            for i, token in enumerate(doc):
                col = 0
                for k in suffixes:
                    extra[i, col] = _affix_key(token, k, True)
                    col += 1
                for k in prefixes:
                    extra[i, col] = _affix_key(token, k, False)
                    col += 1
            attrs = xp.hstack((attrs.astype("uint64"), extra))
        features.append(model.ops.asarray2i(attrs, dtype="uint64"))

    backprop: Callable[[List[Ints2d]], List] = lambda d_features: []
    return features, backprop


@registry.architectures("sud.MultiHashEmbedAffix.v1")
def MultiHashEmbedAffix(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    suffixes: List[int] = [],
    suffix_rows: List[int] = [],
    prefixes: List[int] = [],
    prefix_rows: List[int] = [],
) -> Model[List[Doc], List[Floats2d]]:
    """`spacy.MultiHashEmbed.v2` with one extra hash-embedded table per configured affix length.

    With `suffixes=[]` and `prefixes=[]` this is behaviourally identical to `MultiHashEmbed`, so an
    arm can be switched over before any affix is added and the change stays single-variable.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(suffix_rows) != len(suffixes):
        raise ValueError(f"Mismatched suffix lengths: {len(suffix_rows)} vs {len(suffixes)}")
    if len(prefix_rows) != len(prefixes):
        raise ValueError(f"Mismatched prefix lengths: {len(prefix_rows)} vs {len(prefixes)}")
    if any(k < 1 for k in list(suffixes) + list(prefixes)):
        raise ValueError("affix lengths must be >= 1")

    all_rows = list(rows) + list(suffix_rows) + list(prefix_rows)
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
    extractor = AffixFeatureExtractor(attrs, suffixes, prefixes)
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
