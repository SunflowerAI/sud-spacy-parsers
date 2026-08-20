#!/usr/bin/env python3
"""`sud.KoAnalyserEmbed.v1` — `MultiHashEmbed` plus the morphemes a Korean eojeol hides.

WHAT IT ADDS, per token, from `scripts/ko_analyser.py` at RUNTIME:

  * a hash column on the FIRST morpheme      the lexical key — `잡스는`, `잡스가`, `잡스를` all
                                             become `잡스`
  * a hash column on the LAST morpheme       the functional key — the particle or ending that says
                                             what relation this token bears to its head
  * a multi-hot block over the analyser's tagset, three ways: the first tag, the last tag, and the
    BAG of every tag in the eojeol, each with its own "analyser said nothing" bit

WHY, IN ONE MEASUREMENT (`scripts/eval_ko_oov.py`, released arm, test):

    seen  7 645 tok (65.5 %)   UAS 75.95   LAS 71.84
    OOV   4 032 tok (34.5 %)   UAS 52.16   LAS 38.10

Eojeol tokenisation makes a fresh string of every stem-plus-particle combination, so a third of test
tokens are unseen and parse 33.7 LAS below the rest. The first-morpheme key is the lever:

    key                                        types in train   covers all   covers OOV eojeol
    the eojeol itself (what the parser reads)          27 752        65.5 %             0.0 %
    first morpheme                                     10 569        90.4 %            72.3 %
    last morpheme                                       5 888        94.3 %            83.5 %

⚠ THE OBJECTION, AND THE FALSIFIABLE PREDICTION. `sud_lex_embed.py` proved a per-form table
information-free: keyed on the form, it is a FUNCTION of the form, and the parser already reads the
form. That argument holds wherever the model has a trained representation of the form — and fails
exactly where this layer aims, because an unseen eojeol hashes to an untrained row and carries
nothing learnable, while its first morpheme is a different symbol with a trained row behind it
(61.3 % of OOV tokens land on a key seen at least twice in training). So the claim is testable in a
way a headline LAS cannot settle: **the gain must sit on the OOV tokens.** `eval_ko_oov.py` prints
the split for every arm, and a gain spread evenly across seen and unseen tokens refutes the
mechanism whatever the total says.

⚠ RUNTIME, NOT A SHIPPED TABLE, and here the argument is sharper than the one
`sud_analyser_embed.py` makes for Sanskrit (a frozen extract missed 6.5 % of test tokens whose forms
the analyser knew). The tokens this layer exists for are BY DEFINITION absent from any
corpus-derived key set, so a frozen table would answer for every token except the ones that need
answering — and would load cleanly while scoring like its own capacity control.

⚠ THE BACKEND IS RECORDED IN THE MODEL AND CHECKED ON LOAD (CLAUDE.md hazard 10: ask the model
rather than assuming its input regime). Two analysers do not segment alike, so an arm trained
against one and run against another is reading a channel it never saw. The fingerprint travels in
the extractor's `attrs`, which thinc serialises with the weights, and the forward pass RAISES on a
mismatch or an absent analyser rather than falling back to "unanalysed" for every token.

⚠ ONE SENTINEL, NOT TWO. A token the analyser declines and a token whose morpheme is out of the
hash table must be the SAME input, for the reason an unset MORPH and an empty one must be
(CLAUDE.md; it cost sa 6.8 LAS). Both take `_SILENT`, and the multi-hot block sets a silent bit
rather than leaving the row all-zero.

NO JACKKNIFING. The analyser is external to the treebank, so a form's training-time answer and its
inference-time answer are the same answer — unlike the corpus-harvested lexicon in
`sud_lex_embed.py`, whose folds exist to remove exactly that skew.

`constant = true` is the capacity control: same columns, same Maxout width, same parameter count,
every token given the sentinel and the silent bits. An arm switched over with `feats = []` and
`morph_rows = []` is byte-identical to stock `spacy.MultiHashEmbed.v2`, which
`check_ko_embed.py` asserts rather than argues.

Config usage:

    [components.tok2vec.model.embed]
    @architectures = "sud.KoAnalyserEmbed.v1"
    width = ${components.tok2vec.model.encode.width}
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    rows = [5000, 1000, 2500, 2500]
    morph_rows = [5000, 2000]
    feats = ["First", "Last", "Bag"]
    constant = false
    include_static_vectors = false
"""
from __future__ import annotations

import pathlib
import sys
from typing import Callable, List, Tuple, Union

import numpy
from spacy.ml.models.tok2vec import FeatureExtractor
from spacy.ml.staticvectors import StaticVectors
from spacy.strings import hash_string
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import HashEmbed
from thinc.types import Floats2d, Ints2d, Ragged

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ko_analyser  # noqa: E402

# The mecab-ko-dic (Sejong) tagset, closed and written out rather than harvested from a corpus: a
# layout derived from whatever tags happened to appear would silently change shape with the data,
# and every feature after the changed one would misalign.
KO_TAGS = [
    # 체언
    "NNG", "NNP", "NNB", "NNBC", "NR", "NP",
    # 용언
    "VV", "VA", "VX", "VCP", "VCN",
    # 관형사 · 부사 · 감탄사
    "MM", "MAG", "MAJ", "IC",
    # 조사
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX",
    # 어미
    "EP", "EF", "EC", "ETN", "ETM",
    # 접사
    "XPN", "XSN", "XSV", "XSA", "XR",
    # 기호 · 외국어 · 한자 · 숫자
    "SF", "SE", "SSO", "SSC", "SS", "SC", "SY", "SL", "SH", "SN", "UNKNOWN",
]

FEATS = ("First", "Last", "Bag")

# The one sentinel: no analysis, and any morpheme the caller cannot name, hash to this.
_SILENT = "\x00ko-analyser-silent"
_SILENT_ID = hash_string(_SILENT)


def _layout(feats: List[str]) -> Tuple[dict, int]:
    """Bit offsets. Each feature gets one bit per tag plus ONE 'the analyser said nothing' bit —
    a feature the analyser did not offer and a feature it offered no value for must not be the same
    input."""
    off, n = {}, 0
    for f in feats:
        off[f] = n
        n += len(KO_TAGS) + 1
    return off, n


_TAG_INDEX = {t: i for i, t in enumerate(KO_TAGS)}


def _check_backend(model: Model) -> None:
    want = model.attrs.get("ko_backend")
    got = ko_analyser.fingerprint()          # raises AnalyserUnavailable if none is installed
    # ⚠ The DICTIONARY is compared, not the whole fingerprint. The binding is recorded because a
    # model should say what produced its channel, but it is not the channel: measured over all
    # 31 532 distinct eojeol of the treebank, natto-py and python-mecab-ko agree on 100.00 % of tag
    # sequences and 99.99 % of lexical keys (scripts/check_ko_backends.py). Refusing on the binding
    # would reject an install that reproduces the training channel to four decimal places.
    if want and want.rsplit("/", 1)[-1] != got.rsplit("/", 1)[-1]:
        raise ValueError(
            f"sud.KoAnalyserEmbed.v1: this model's channel was built with {want!r} and the analyser "
            f"available here is {got!r} — a different DICTIONARY, not merely a different binding. "
            f"Two analysers do not segment alike, so the parser would be fed values it never saw. "
            f"Install a mecab-ko-dic backend (`pip install python-mecab-ko`), or retrain. "
            f"Refusing to run: the alternative is parsing quietly worse.")


def KoMorphIds(constant: bool) -> Model[List[Doc], List[Ints2d]]:
    return Model("extract_ko_morph_ids", _ids_forward, attrs={"ko_constant": bool(constant),
                                                              "ko_backend": None})


def _ids_forward(model: Model, docs, is_train: bool):
    constant = model.attrs["ko_constant"]
    if not constant:
        _check_backend(model)
    out = []
    for doc in docs:
        arr = numpy.zeros((len(doc), 2), dtype="uint64")
        for i, tok in enumerate(doc):
            if constant:
                arr[i, 0] = arr[i, 1] = _SILENT_ID
                continue
            ms = ko_analyser.analyse(tok.text)
            arr[i, 0] = hash_string(ms[0][0]) if ms else _SILENT_ID
            arr[i, 1] = hash_string(ms[-1][0]) if ms else _SILENT_ID
        out.append(model.ops.asarray(arr))
    return out, lambda d: []


def KoTagSets(feats: List[str], constant: bool) -> Model[List[Doc], List[Floats2d]]:
    return Model("extract_ko_tag_sets", _tags_forward,
                 attrs={"ko_feats": list(feats), "ko_constant": bool(constant), "ko_backend": None})


def _tags_forward(model: Model, docs, is_train: bool):
    feats = model.attrs["ko_feats"]
    constant = model.attrs["ko_constant"]
    if not constant:
        _check_backend(model)
    off, n_dims = _layout(feats)
    xp = model.ops.xp
    out = []
    for doc in docs:
        arr = xp.zeros((len(doc), n_dims), dtype="f")
        for i, tok in enumerate(doc):
            ms = [] if constant else ko_analyser.analyse(tok.text)
            for f in feats:
                base = off[f]
                if not ms:
                    arr[i, base + len(KO_TAGS)] = 1.0        # the silent bit
                    continue
                if f == "First":
                    idx = [_TAG_INDEX.get(ms[0][1])]
                elif f == "Last":
                    idx = [_TAG_INDEX.get(ms[-1][1])]
                else:
                    idx = [_TAG_INDEX.get(t) for _, t in ms]
                idx = sorted({j for j in idx if j is not None})
                if idx:
                    for j in idx:
                        arr[i, base + j] = 1.0
                else:
                    arr[i, base + len(KO_TAGS)] = 1.0
        out.append(model.ops.asarray2f(arr))
    return out, lambda d: []


@registry.architectures("sud.KoAnalyserEmbed.v1")
def KoAnalyserEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    morph_rows: List[int] = [],
    feats: List[str] = [],
    constant: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if morph_rows and len(morph_rows) != 2:
        raise ValueError(
            f"morph_rows is [first-morpheme rows, last-morpheme rows]; got {morph_rows}")
    bad = [f for f in feats if f not in FEATS]
    if bad:
        raise ValueError(f"unknown feature(s) {bad}; known: {list(FEATS)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")

    # The fingerprint is taken at BUILD time and travels in the bytes. On a machine with no analyser
    # it stays None and the forward pass raises with the installation instructions, rather than the
    # architecture failing to construct — build time and load time are different contracts, the
    # lesson `seal_la_lemvec_model.py` records.
    backend = None
    if not constant and (feats or morph_rows):
        try:
            backend = ko_analyser.fingerprint()
        except ko_analyser.AnalyserUnavailable:
            backend = None

    # seed 7 and the same increment order as MultiHashEmbed, so an arm switched over with no extra
    # channels is seeded identically to stock.
    seed = 7

    def make_hash_embed(index, all_rows):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i, rows) for i in range(len(rows))]
    extras: List[Model] = []
    if include_static_vectors:
        extras.append(StaticVectors(width, dropout=0.0))
    if morph_rows:
        ids = KoMorphIds(constant)
        ids.attrs["ko_backend"] = backend
        morph_embeds = [HashEmbed(width, morph_rows[i], column=i, seed=100 + i, dropout=0.0)
                        for i in range(2)]
        extras.append(chain(ids, list2ragged(), with_array(concatenate(*morph_embeds))))
    if feats:
        tagsets = KoTagSets(feats, constant)
        tagsets.attrs["ko_backend"] = backend
        _, n_dims = _layout(list(feats))
        extras.append(chain(tagsets, list2ragged(), with_array(Linear(width, n_dims))))

    n_blocks = len(embeddings) + include_static_vectors + bool(morph_rows) * 2 + bool(feats)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, width * n_blocks, nP=3, dropout=0.0, normalize=True)
    )
    if not extras:
        # The flat chain, node for node as spacy.MultiHashEmbed.v2 builds it — not merely an
        # equivalent nesting, so `check_ko_embed.py` can assert byte identity.
        return chain(
            FeatureExtractor(list(attrs)),
            list2ragged(),
            with_array(concatenate(*embeddings)),
            max_out,
            ragged2list(),
        )
    hashed: Model[List[Doc], Ragged] = chain(
        FeatureExtractor(list(attrs)),
        list2ragged(),
        with_array(concatenate(*embeddings)),
    )
    return chain(concatenate(hashed, *extras), max_out, ragged2list())
