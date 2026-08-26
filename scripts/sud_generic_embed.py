#!/usr/bin/env python3
"""`sud.GenericEmbed.v1` — the language-agnostic token representation: UPOS + decomposed FEATS +
one cross-lingually aligned vector. No wordform, no affix, no shape, no script, no language id.

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER EMBED IN THIS REPO. All of them start from
`MultiHashEmbed`'s `["NORM", "PREFIX", "SUFFIX", "SHAPE"]`, i.e. from the token STRING, and add
channels beside it. A string is exactly what cannot be shared across thirteen writing systems, so
this layer removes it and keeps only channels that mean the same thing in every language:

    UPOS            17 universal categories
    FEATS           one hash table per morphological CATEGORY, not one per bundle
    aligned vector  128 d, one shared space for all thirteen languages, + an OOV flag

⚠ THE NEGATIVE RESULT ON STATIC VECTORS DOES NOT TRANSFER HERE, and it is worth being explicit
because it looks as though it should. `NEGATIVE-RESULTS.md` records kanripo vectors as a parser
input for lzh at **+0.04 LAS over three seeds**, and fastText `md` on yue/id/ko as +0.2-0.9 inside
seed noise. Both measured a vector channel added BESIDE the wordform, for a parser that already read
the string and had already learned that string's syntax from the same treebank. Here the vector is
the ONLY lexical channel there is, and its job is not to add information to a language the parser
knows -- it is to make a Tamil noun and a Latin noun land in the same place. Same asset class,
different question; the earlier results neither support nor refute this one.

WHY FEATS IS DECOMPOSED. `MORPH` as a spaCy column is a hash of the WHOLE normalised bundle, so
`Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` arrive as two unrelated symbols. Across thirteen
treebanks that is fatal rather than merely crude: no two of them share a bundle inventory, so a
whole-bundle hash would make every language's morphology a private vocabulary and there would be
nothing cross-lingual left in the channel. `sud.MultiHashEmbedFeats.v1` already solves this and is
reused verbatim -- one table per category, `Case=Acc` the same symbol whether it came from Latin,
Sanskrit or Tamil.

HOW THE LANGUAGE IS CHOSEN, AND WHY THAT IS NOT A LANGUAGE FEATURE. The thirteen tables live in one
space but are separate row-sets, so a lookup needs to know which language a token is in. That is
read off `Doc._.tb_lang`, used ONLY to pick the row, and never embedded: the model has no parameter
that varies with it, and two identical (UPOS, FEATS, vector) triples get identical treatment
whatever language they came from. `lang_id = true` is the control that measures what actually
knowing the language is worth -- it adds a thirteen-row embedding table and is off by default.

THREE REFUSALS, each for a failure this project has already paid for:

  * **`Doc._.tb_lang` unset** -> raise. A default language would silently look up every token in one
    table, miss nearly all of them, and score exactly like the layer's own capacity control.
  * **table absent** -> raise, naming the build script. `sud.LemmaVecEmbed.v1` records why: an
    all-zeros fallback loads cleanly and is indistinguishable from a dead channel.
  * **fingerprint mismatch** -> raise. A table built from a different asset release has the same
    shape and the wrong rows, which is CLAUDE.md hazard 10 exactly (zh's traditional jieba
    dictionary silently replaced by jieba's own simplified one).

TWO CONTROLS, because adding a 128-d block also adds parameters and the two must be separable
(NEGATIVE-RESULTS.md, "always run a capacity control"):

    constant = true    same Linear, same parameter count, every token handed the zero vector and
                       the OOV flag. Isolates the PARAMETERS.
    shuffle = true     the same rows with the key-to-row correspondence destroyed WITHIN each
                       language: identical shapes, identical norms, identical parameter count,
                       zero information. Isolates the ALIGNMENT. This is the stronger control and
                       the one `make_vec_config.py` argues for.

Config usage:

    [components.tok2vec.model.embed]
    @architectures = "sud.GenericEmbed.v1"
    width = 128
    table = "assets_vec/generic_vec.npz"
    fingerprint = "9d0499f1bf1e0016"
    feats = ["Case", "Number", ...]
    feat_rows = [40, 12, ...]
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import HashEmbed
from thinc.types import Floats2d, Ragged

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sud_feats_embed import FeatsFeatureExtractor                               # noqa: E402

#: The language a doc's tokens are in -- the row-set selector, never a model input. Set by
#: `sud.GenericCorpus.v1` at training time and by the caller at inference.
if not Doc.has_extension("tb_lang"):
    Doc.set_extension("tb_lang", default=None)

_TABLES: dict = {}


def load_table(path, shuffle: bool = False):
    """`({(lang, key): row}, vectors, meta)`, loaded once per process per (path, shuffle)."""
    ck = (str(path), bool(shuffle))
    if ck in _TABLES:
        return _TABLES[ck]
    f = pathlib.Path(path)
    if not f.exists():
        raise ValueError(
            f"sud.GenericEmbed.v1: aligned vector table {path} not found. Build it with "
            f"scripts/build_generic_vectors.py. Refusing to fall back to all-zeros, which would "
            f"load cleanly and score exactly like this layer's own capacity control.")
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    keys = bytes(z["keys"]).decode("utf-8").split("\n")
    V = z["vectors"].astype("float32")
    idx = {}
    for i, k in enumerate(keys):
        lang, _, key = k.partition("\t")
        idx[(lang, key)] = i
    if shuffle:
        # Permute WITHIN each language, so the control keeps every property of the real table
        # except the one being tested: same rows, same norms, same per-language geometry, and the
        # key-to-row correspondence destroyed. A global shuffle would additionally scramble which
        # language's region a token lands in, which is a second change and a weaker control.
        rng = np.random.default_rng(0)
        bounds: dict = {}
        for i, k in enumerate(keys):
            lang = k.partition("\t")[0]
            lo, hi = bounds.get(lang, (i, i))
            bounds[lang] = (min(lo, i), max(hi, i))
        V = V.copy()
        for lang, (lo, hi) in bounds.items():
            perm = rng.permutation(hi - lo + 1) + lo
            V[lo:hi + 1] = V[perm]
    _TABLES[ck] = (idx, V, meta)
    return _TABLES[ck]


def resolve_table(table, vector_dim: Optional[int], fingerprint: Optional[str],
                  constant: bool, shuffle: bool, who: str):
    """A `_Table` and its width, or -- for the capacity control -- `(None, width)`.

    `constant = true` must construct WITHOUT the table: the control's whole point is that it runs on
    a machine where the channel is dead, and requiring the file would make the control depend on the
    thing it is controlling for.
    """
    if constant:
        if vector_dim is None:
            raise ValueError(f"{who}: `constant = true` needs `vector_dim` (an integer) -- the "
                             f"control must emit a block of the same width as the arm it controls, "
                             f"and with no table there is nothing to read the width off.")
        return None, int(vector_dim)
    if table is None:
        raise ValueError(f"{who} needs `table` (the .npz from scripts/build_generic_vectors.py), "
                         f"or `constant = true` for the capacity control.")
    idx, V, meta = load_table(table, shuffle=shuffle)
    got = meta.get("fingerprint")
    if fingerprint and got != fingerprint:
        raise ValueError(
            f"{who}: table {table} has fingerprint {got!r} but this model was built on "
            f"{fingerprint!r}. Same shape, different rows -- a table from another asset release "
            f"loads cleanly and is wrong only on the vocabulary the two disagree about. Rebuild "
            f"with scripts/build_generic_vectors.py or point `table` at the right file.")
    return _Table(idx, V, meta), int(meta["dim"])




# --------------------------------------------------------------------------------------------
# The lookup key.
#
# Folding is PER LANGUAGE and cannot be done once for all of them: sa is keyed by LEMMA and the rest
# by FORM, eleven of the thirteen are lowercased and two are not, and la alone carries an
# orthography fold. The rules are not reimplemented here -- `aligned_vectors.KEY_NORM` is the one
# place that knows them, imported rather than copied, exactly as `align_vectors.py` imports it.
#
# The (lang, raw) -> row result is memoised because the fold is a Python string transform and would
# otherwise run once per token per epoch instead of once per type. Same reasoning as the feats
# cache in `sud_feats_embed.py`, and safe for the same reason: the fold is a pure function of the
# string and the language.

from aligned_vectors import KEY_NORM                                            # noqa: E402


class _Table:
    """The merged table plus the per-language fold each lookup needs."""

    __slots__ = ("idx", "V", "meta", "langs", "_fold", "_attr", "_cache")

    def __init__(self, idx, V, meta):
        self.idx, self.V, self.meta = idx, V, meta
        self.langs = sorted(meta["languages"])
        self._fold, self._attr, self._cache = {}, {}, {}
        for lang, m in meta["languages"].items():
            norm = KEY_NORM.get(m.get("key_norm") or "")
            if norm is not None:
                self._fold[lang] = norm
            elif m.get("lowercased"):
                self._fold[lang] = str.lower
            else:
                self._fold[lang] = None
            self._attr[lang] = m.get("key_attr", "form")

    def row(self, lang, token):
        raw = token.lemma_ if self._attr.get(lang) == "lemma" else token.text
        ck = (lang, raw)
        if ck in self._cache:
            return self._cache[ck]
        f = self._fold.get(lang)
        folded = f(raw) if f is not None else raw
        r = self.idx.get((lang, folded))
        self._cache[ck] = r
        return r


def AlignedVecExtractor(table: Optional[_Table], dim: int):
    return Model("extract_aligned_vectors", _alignedvec_forward,
                 attrs={"av_table": table, "av_dim": int(dim)})


def _alignedvec_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    table: Optional[_Table] = model.attrs["av_table"]
    dim = int(model.attrs["av_dim"])
    out: List[Floats2d] = []
    for doc in docs:
        arr = np.zeros((len(doc), dim + 1), dtype="f")
        if table is None:                              # the capacity control: a dead channel
            arr[:, dim] = 1.0
            out.append(model.ops.asarray2f(arr))
            continue
        # Spans reach here too -- spaCy slices the parent doc's array -- and an extension set on
        # the Doc is not readable off a Span, so go to `.doc` for it.
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        if not lang:
            raise ValueError(
                "sud.GenericEmbed.v1: Doc._.tb_lang is unset. This layer holds one row-set per "
                "language and cannot guess which to look a token up in -- defaulting to one would "
                "miss nearly every token and score exactly like the dead-channel control. Set it "
                "(`doc._.tb_lang = 'ta'`); training corpora get it from sud.GenericCorpus.v1.")
        if lang not in table.langs:
            raise ValueError(
                f"sud.GenericEmbed.v1: Doc._.tb_lang = {lang!r}, which this table has no rows "
                f"for. Known: {' '.join(table.langs)}.")
        for i, tok in enumerate(doc):
            r = table.row(lang, tok)
            if r is None:
                arr[i, dim] = 1.0        # OOV gets its OWN dimension, never a zero vector -- the
            else:                        # same distinction an unset MORPH needs from an empty one
                arr[i, :dim] = table.V[r]
        out.append(model.ops.asarray2f(arr))
    backprop: Callable[[List[Floats2d]], List] = lambda d: []
    return out, backprop


# --------------------------------------------------------------------------------------------
# The language-id control. OFF by default, and the default is the claim: the headline arm has no
# parameter that varies with the language, so it cannot have learned "Latin puts the verb last" as
# a language fact rather than as a fact about Latin's UPOS/FEATS/vector distribution. Turning it on
# measures what the shortcut is worth, which is the only honest way to say the shortcut was not
# taken.

def LangIdExtractor(langs: List[str]):
    return Model("extract_lang_id", _langid_forward, attrs={"li_langs": list(langs)})


def _langid_forward(model: Model, docs, is_train: bool):
    langs = model.attrs["li_langs"]
    out = []
    for doc in docs:
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        arr = np.zeros((len(doc), len(langs)), dtype="f")
        if lang in langs:
            arr[:, langs.index(lang)] = 1.0
        out.append(model.ops.asarray2f(arr))
    return out, lambda d: []


# --------------------------------------------------------------------------------------------
# The TYPOLOGY channel: a graded word-order profile per language, two dims per parameter
# (value, known). Off by default.
#
# ⚠ WHY GRADED AND NOT BINARY, and why this is not simply the language-id control renamed.
# Binarised at 50 %, 12 of the 13 profiles are DISTINCT -- so binary features are a language
# identifier in disguise, and the thirteen-row language embedding they would be equivalent to was
# measured at +0.34 macro LAS, i.e. nothing. Graded values are different in two ways that matter:
# they say "free order" where a bit must lie (Latin's O-V 49 / Obl-V 53 / Adj-N 49 / Num-N 52), and
# they let languages SHARE structure rather than each getting a private row -- which is the only
# mechanism by which this could help the low-resource end.
#
# ⚠ AND IT IS AIMED AT ZERO-SHOT, NOT IN-SAMPLE. In-sample the parser observes word order directly
# in every sentence's UPOS sequence, so a sentence-independent prior on it is largely redundant --
# which is what the language-id null already suggests. A HELD-OUT language is the case a language
# embedding structurally cannot serve (no row exists) and a typological profile can.

_TYPOLOGY: dict = {}


def load_typology(path, shuffle: bool = False):
    """`(langs, {lang: [v, known, v, known, ...]}, n_dims)`, loaded once per process."""
    ck = (str(path), bool(shuffle))
    if ck in _TYPOLOGY:
        return _TYPOLOGY[ck]
    f = pathlib.Path(path)
    if not f.exists():
        raise ValueError(
            f"sud.GenericEmbed.v1: typology table {path} not found. Build it with "
            f"scripts/build_typology.py. Refusing to fall back to a constant vector, which would "
            f"load cleanly and score exactly like this channel's own capacity control.")
    blob = json.loads(f.read_text(encoding="utf-8"))
    params = blob["meta"]["params"]
    langs = sorted(blob["languages"])
    vecs = {}
    for lang in langs:
        row = []
        for k in params:
            e = blob["languages"][lang][k]
            row.extend([float(e["value"]), float(e["known"])])
        vecs[lang] = row
    if shuffle:
        # The capacity control: same vectors, same parameter count, attached to the WRONG
        # languages. Anything a permuted profile also achieves was never the typology.
        # A DERANGEMENT, not just a permutation: a plain shuffle left one language holding its own
        # profile, and a control that is partly not a control is the weaker kind of mistake -- it
        # can only ever narrow the gap it exists to measure.
        rng = np.random.default_rng(0)
        n = len(langs)
        while True:
            perm = list(rng.permutation(n))
            if all(perm[i] != i for i in range(n)):
                break
        vecs = {langs[i]: vecs[langs[perm[i]]] for i in range(n)}
    out = (langs, vecs, len(params) * 2)
    _TYPOLOGY[ck] = out
    return out


def TypologyExtractor(langs, vecs, dim):
    return Model("extract_typology", _typology_forward,
                 attrs={"ty_langs": list(langs), "ty_vecs": dict(vecs), "ty_dim": int(dim)})


def _typology_forward(model: Model, docs, is_train: bool):
    vecs = model.attrs["ty_vecs"]
    dim = int(model.attrs["ty_dim"])
    out = []
    for doc in docs:
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        row = vecs.get(lang)
        if row is None:
            raise ValueError(
                f"sud.GenericEmbed.v1: no typological profile for Doc._.tb_lang={lang!r}. "
                f"Known: {' '.join(sorted(vecs))}. Refusing to substitute a neutral vector, which "
                f"would be indistinguishable from a measured one.")
        arr = np.tile(np.asarray(row, dtype="f"), (len(doc), 1))
        out.append(model.ops.asarray2f(arr))
    return out, lambda d: []


# --------------------------------------------------------------------------------------------


@registry.architectures("sud.GenericEmbed.v1")
def GenericEmbed(
    width: int,
    feats: List[str] = [],
    feat_rows: List[int] = [],
    upos_rows: int = 64,
    table: Optional[str] = None,
    fingerprint: Optional[str] = None,
    vector_dim: Optional[int] = None,
    constant: bool = False,
    shuffle: bool = False,
    lang_id: bool = False,
    typology: Optional[str] = None,
    typology_shuffle: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    """UPOS + one table per FEATS category + the aligned-vector block. No string channel at all.

    `upos_rows` is 64 for a 17-value inventory, i.e. deliberately over-provisioned: a hash collision
    between two of seventeen universal categories would be the single most damaging thing that could
    happen in this layer, and the table costs 64 * width * 4 bytes.
    """
    if len(feat_rows) != len(feats):
        raise ValueError(f"Mismatched feature lengths: {len(feat_rows)} vs {len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    if any(r < 1 for r in feat_rows):
        raise ValueError("feature rows must be >= 1")
    if constant and shuffle:
        raise ValueError("`constant` and `shuffle` are two different controls -- a dead channel and "
                         "an uninformative one. Pick one; together they are just the dead channel "
                         "with a misleading name.")

    tbl, dim = resolve_table(table, vector_dim, fingerprint, constant, shuffle,
                             "sud.GenericEmbed.v1")

    all_rows = [upos_rows] + list(feat_rows)
    seed = 7                      # same seeding order as MultiHashEmbed, so columns line up

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(all_rows))]
    pieces = [chain(FeatsFeatureExtractor(["POS"], feats), list2ragged(),
                    with_array(concatenate(*embeddings)))]
    pieces.append(chain(AlignedVecExtractor(tbl, dim), list2ragged(),
                        with_array(Linear(width, dim + 1))))
    n_blocks = len(embeddings) + 1
    if typology is not None:
        ty_langs, ty_vecs, ty_dim = load_typology(typology, shuffle=typology_shuffle)
        pieces.append(chain(TypologyExtractor(ty_langs, ty_vecs, ty_dim), list2ragged(),
                            with_array(Linear(width, ty_dim))))
        n_blocks += 1
    if lang_id:
        langs = tbl.langs if tbl is not None else []
        if not langs:
            raise ValueError("`lang_id = true` needs the table, to know the language inventory.")
        pieces.append(chain(LangIdExtractor(langs), list2ragged(),
                            with_array(Linear(width, len(langs)))))
        n_blocks += 1

    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, width * n_blocks, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())
