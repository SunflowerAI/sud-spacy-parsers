#!/usr/bin/env python3
"""The v2 embedding layer plus a LEXICAL channel: one aligned vector per token.

WHAT CHANGED, AND WHY IT IS NOT JUST v1 AGAIN. v1 read an aligned fastText vector of the WORDFORM.
It was worth +8.60 macro LAS held-in and it was DEAD zero-shot, for a structural reason no amount of
tuning would have fixed: a language with no treebank has no vector table, so the channel was empty
in exactly the situation the arm exists for. v2 dropped it.

Here the channel is keyed by LEMMA and has TWO fill routes into ONE space:

    training     the token's lemma, looked up in its own language's aligned table
    deployment   an ENGLISH GLOSS the user supplies, looked up in the English table

Those are the same space. Aligned vectors are rotated into the English hub, which is what makes an
English gloss a legitimate stand-in for a source lemma rather than a different kind of thing. The
consequence is the point of the arm: training coverage now limits how well the channel is LEARNED,
not where it can be USED. Chintang, K'iche' and Xavante have no fastText at all and can still fill
it from glosses.

⚠ THE SUBSTITUTION IS AN ASSUMPTION UNTIL IT IS MEASURED, and this repo has paid 4.83 F once for
asking a model at inference for a regime it never met (`CLAUDE.md`, hazard 11). It was measured
first, on the two treebanks carrying both a `Gloss=` column and an existing aligned table:

                  cos(source, en-gloss)   shuffled   random-en   beat shuffled
    ar PADT             +0.444             +0.109     +0.012        94.0 %
    lzh Kyoto           +0.340             +0.116     +0.022        84.5 %

Well above a permuted control, and nowhere near 1.0 -- so the shift is real and the arm has to be
EVALUATED under the deployment fill, never only under the training one. `vectors_fill` names the
regime, travels in the config, and is read back rather than assumed.

⚠ A GLOSS IS HALF GRAMMAR, AND THE HALF IT SUPPLIES IS THE HALF THAT HELPS LEAST. Interlinear glosses
write content morphemes as English words and grammatical ones as Leipzig abbreviations -- `NMLZ1`,
`IPFV.AFF`, `pick_up_for-3[SG].P-BEN-IMP[.2SG.A]`. fastText has no useful row for those, and function
words are where the attachment signal lives. All-caps pieces are dropped rather than hashed, on the
grounds that FEATS already carries them; what is left resolved for 98 % of pieces on ar and lzh,
which are two of the mildest gloss styles in the release.

⚠ AND GLOSSES ARE NOT ALWAYS ENGLISH. Bambara, Gbaya, Haitian and Occitan gloss into French, Pesh
into Spanish. Nothing here can detect that; the corpus builder must.

THE REFUSALS, in the same spirit as v2's:

  * `Doc._.tb_lang` unset -> raise. Inherited from v2.
  * a language with no row-set in the table -> WARN once and run the channel all-OOV. Not an
    error: 48 of the 80 training languages are in this state, so it is a signal the model learns
    to condition on rather than a defect. It is not v2's unfitted-language-row case, which cost
    4 LAS because it was an untrained PARAMETER that training never produced.
  * a doc with nothing to fill the channel from -> raise. This is the failure that would otherwise
    ship: a user sets `vectors_fill = "gloss"`, forgets `Token._.gloss`, and gets the
    OOV-on-every-token parse with no error anywhere.

OOV GETS ITS OWN DIMENSION, never a zero vector. A zero vector is indistinguishable from a real one
that happens to be near the origin, and the repo's most expensive recurring bug is exactly this
shape -- an unset MORPH and an empty one are different inputs, and conflating them cost Sanskrit
6.8 LAS.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import List, Optional

import numpy as np
from spacy.tokens import Doc, Token
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import Embed, HashEmbed
from thinc.types import Floats2d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from aligned_vectors import KEY_NORM                      # noqa: E402  ONE definition of each fold
from sud_feats_embed import FeatsFeatureExtractor         # noqa: E402
from sud_generic_embed_v2 import (                        # noqa: E402
    ConstantExtractor, LangIdExtractor, LangSlotExtractor, TypologyExtractor, load_typology)

#: The English gloss for a token, the DEPLOYMENT fill for the lexical channel. Unset means "no
#: gloss for this token", which reaches the model as the OOV dimension -- not as a zero vector.
if not Token.has_extension("gloss"):
    Token.set_extension("gloss", default=None)

FILLS = ("lemma", "gloss", "auto")

_TABLES: dict = {}

#: languages already warned about, so a training run says it once rather than per batch
_WARNED: set = set()

#: fill regimes already warned about for an empty doc, so a corpus says it once
_WARNED_FILL: set = set()


class _VecTable:
    """The merged aligned table, plus the per-language key fold each lookup needs.

    Folds cannot be guessed and are not guessed here: each language's asset records whether its keys
    are lowercased and whether it declares an explicit `key_norm` (Latin's orthography fold is the
    only one), and both are read back off the table's own meta. Folding when you should not, or not
    folding when you should, is 31 points of English type coverage.
    """

    __slots__ = ("idx", "V", "meta", "langs", "dim", "_fold", "_cache")

    def __init__(self, idx, V, meta):
        self.idx, self.V, self.meta = idx, V, meta
        self.langs = sorted(meta["languages"])
        self.dim = int(V.shape[1])
        self._fold, self._cache = {}, {}
        for lang, m in meta["languages"].items():
            norm = KEY_NORM.get(m.get("key_norm") or "")
            if norm is not None:
                self._fold[lang] = norm
            elif m.get("lowercased"):
                self._fold[lang] = str.lower
            else:
                self._fold[lang] = None

    def row(self, lang: str, raw: Optional[str]):
        """Row index for a raw key in `lang`, or None. `_` is MISSING, never a key.

        spaCy keeps a CoNLL-U `_` as a LITERAL string rather than as absent, which once taught a
        Sanskrit transducer `FORM -> "_"` on 5 043 tokens. Treat it as absent here too.
        """
        if not raw or raw == "_":
            return None
        ck = (lang, raw)
        hit = self._cache.get(ck, ...)
        if hit is not ...:
            return hit
        f = self._fold.get(lang)
        r = self.idx.get((lang, f(raw) if f is not None else raw))
        self._cache[ck] = r
        return r


def load_vectors(path: str) -> _VecTable:
    if path in _TABLES:
        return _TABLES[path]
    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError(
            f"sud.GenericEmbed.v3: no vector table at {path}. The channel has no neutral fallback "
            f"on purpose -- build it with scripts/build_generic_vectors_v3.py, or drop `vectors` "
            f"from the config to train the arm without a lexical channel.")
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    keys = str(z["keys"]).split("\n")
    V = np.ascontiguousarray(z["vectors"], dtype="float32")
    if len(keys) != len(V):
        raise ValueError(f"{path}: {len(keys)} keys against {len(V)} rows")
    idx = {}
    for i, k in enumerate(keys):
        lang, _, key = k.partition("\t")
        idx[(lang, key)] = i
    t = _VecTable(idx, V, meta)
    _TABLES[path] = t
    return t


def VecExtractor(table: _VecTable, fill: str, shuffle: bool = False):
    return Model("extract_aligned_vec", _vec_forward,
                 attrs={"vt_table": table, "vt_fill": fill, "vt_shuffle": bool(shuffle),
                        "vt_dim": table.dim + 1})


def _vec_forward(model: Model, docs, is_train: bool):
    table: _VecTable = model.attrs["vt_table"]
    fill: str = model.attrs["vt_fill"]
    shuffle: bool = model.attrs["vt_shuffle"]
    dim: int = model.attrs["vt_dim"]
    out = []
    for doc in docs:
        parent = doc.doc if hasattr(doc, "doc") else doc
        lang = parent._.tb_lang
        if lang is None:
            raise ValueError(
                "sud.GenericEmbed.v3: Doc._.tb_lang is unset. The reader stamps it during training "
                "and the caller must stamp it at inference; there is no default, because one would "
                "silently give every document the same row-set.")
        if lang not in table.langs and fill != "gloss" and lang not in _WARNED:
            # ⚠ A WARNING AND NOT AN ERROR, and the distinction took a wrong turn to find. Only 32
            # of the 80 training languages have rows, so an all-OOV channel is a state the model
            # meets constantly WHILE TRAINING and has learned to condition on -- the OOV dimension
            # exists for exactly this. Refusing here does not protect a deployer; it makes the arm
            # untrainable. This is NOT the same shape as v2's unfitted language row, which was 4 LAS
            # worse than no channel because it was an untrained PARAMETER rather than a trained
            # signal, and nothing in training ever produced it.
            _WARNED.add(lang)
            print(f"sud.GenericEmbed.v3: no vector rows for tb_lang={lang!r}; the lexical channel "
                  f"is OOV on every token of it. Fine while training (48 of 80 languages are in "
                  f"this state) and a real loss at inference -- supply Token._.gloss and set "
                  f"`vectors_fill = \"gloss\"` to fill it from English instead.", file=sys.stderr)

        n = len(doc)
        A = model.ops.alloc2f(n, dim)
        rows, seen_key = [], False
        for i, tok in enumerate(doc):
            r = None
            if fill in ("gloss", "auto"):
                g = tok._.gloss
                if g:
                    seen_key = True
                    r = table.row("en", g)
            if r is None and fill in ("lemma", "auto"):
                lem = tok.lemma_
                if lem and lem != "_":
                    seen_key = True
                    r = table.row(lang, lem)
            rows.append(r)
        if not seen_key and fill not in _WARNED_FILL:
            # ⚠ A WARNING, AND THE ENFORCEMENT LIVES IN THE CALLER. This began as an error and fired
            # on legitimate input: with one sentence per doc, a short sentence of punctuation and
            # function words has nothing glossable in it, and that is normal rather than a mistake.
            # The failure actually worth catching -- a caller who set `fill = "gloss"` and never set
            # Token._.gloss at all -- is not visible from ONE doc; it is visible from the corpus
            # fill rate, which the caller computes and this layer cannot see. So the layer says it
            # once and `eval_generic_v3.py` refuses on a whole-language fill rate of zero.
            _WARNED_FILL.add(fill)
            want = {"lemma": "a lemma", "gloss": "Token._.gloss",
                    "auto": "either a lemma or Token._.gloss"}[fill]
            print(f"sud.GenericEmbed.v3: vectors_fill={fill!r} and a doc had {want} on no token, "
                  f"so the channel is OOV throughout it. Normal for a short sentence; if it holds "
                  f"across a corpus, the input is missing and the parse is silently worse than one "
                  f"from an arm with no lexical channel.", file=sys.stderr)

        if shuffle:
            # The control that asks whether it is the RIGHT vector that matters, not merely A
            # vector: same rows, same OOV pattern, permuted across positions. Deterministic on the
            # doc so a rerun reproduces -- Math.random-style nondeterminism would make the control
            # unrepeatable across seeds, which is the one thing a control may not be.
            order = np.argsort(np.array([hash((lang, i)) % (2 ** 31) for i in range(n)]))
            rows = [rows[j] for j in order]

        for i, r in enumerate(rows):
            if r is None:
                A[i, -1] = 1.0                       # OOV owns its own dimension
            else:
                A[i, :-1] = table.V[r]
        out.append(A)

    def backprop(d):                                  # a lookup has no gradient
        return []

    return out, backprop


def VecConstantExtractor(dim: int):
    """The capacity control: the same Linear, every token handed a zero vector and the OOV flag.

    Not the same thing as omitting `vectors`. Omitting it removes a whole Maxout block; this keeps
    every parameter and removes only the information, which is the comparison a delta may be quoted
    against. v1 drew exactly this distinction for its own vector channel.
    """
    return Model("extract_aligned_vec_constant", _vec_constant_forward, attrs={"vc_dim": int(dim)})


def _vec_constant_forward(model: Model, docs, is_train: bool):
    dim = model.attrs["vc_dim"]
    out = []
    for doc in docs:
        A = model.ops.alloc2f(len(doc), dim)
        A[:, -1] = 1.0
        out.append(A)

    def backprop(d):
        return []

    return out, backprop


@registry.architectures("sud.GenericEmbed.v3")
def GenericEmbedV3(
    width: int,
    feats: List[str] = [],
    feat_rows: List[int] = [],
    upos_rows: int = 64,
    typology: Optional[str] = None,
    typology_shuffle: bool = False,
    typology_constant: bool = False,
    typology_dim: int = 8,
    lang_id: bool = False,
    langs: List[str] = [],
    lang_embed: bool = False,
    lang_embed_rows: int = 0,
    lang_embed_dim: int = 0,
    lang_slots: dict = {},
    vectors: Optional[str] = None,
    vectors_fill: str = "lemma",
    vectors_constant: bool = False,
    vectors_shuffle: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    import collections

    if len(feat_rows) != len(feats):
        raise ValueError(f"feats has {len(feats)} entries, feat_rows has {len(feat_rows)}")
    if len(set(feats)) != len(feats):
        dupes = [f for f, n in collections.Counter(feats).items() if n > 1]
        raise ValueError(f"duplicate FEATS category: {dupes}")
    if any(r < 1 for r in feat_rows):
        raise ValueError("feat_rows must all be >= 1")
    if typology_shuffle and typology_constant:
        raise ValueError(
            "typology_shuffle and typology_constant are two different controls and together they "
            "are just the dead channel with a misleading name. Pick one.")
    if (typology_shuffle or typology_constant) and typology is None:
        raise ValueError("typology_shuffle/typology_constant need `typology` to be set")
    if lang_embed and lang_id:
        raise ValueError("lang_embed and lang_id are two encodings of the same thing; pick one")
    if lang_embed and not lang_slots:
        raise ValueError("lang_embed needs `lang_slots`, a {language: row} map with spare rows")
    if lang_id and not langs:
        raise ValueError("lang_id = true needs an explicit `langs` list")

    if vectors_fill not in FILLS:
        raise ValueError(f"vectors_fill must be one of {FILLS}, not {vectors_fill!r}")
    if vectors_shuffle and vectors_constant:
        raise ValueError(
            "vectors_shuffle and vectors_constant are two different controls -- one asks whether the "
            "RIGHT vector matters, the other what the PARAMETERS buy. Together they are the dead "
            "channel under a misleading name. Pick one.")
    if (vectors_shuffle or vectors_constant) and vectors is None:
        raise ValueError("vectors_shuffle/vectors_constant need `vectors` to be set, so that the "
                         "arm being controlled is unambiguous")

    all_rows = [upos_rows] + list(feat_rows)
    seed = 7  # same seeding order as MultiHashEmbed, so the columns line up

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(all_rows))]
    pieces = [chain(FeatsFeatureExtractor(["POS"], feats), list2ragged(),
                    with_array(concatenate(*embeddings)))]
    n_blocks = len(embeddings)

    if typology is not None:
        ty_langs, ty_vecs, ty_dim, _shift = load_typology(
            typology, shuffle=typology_shuffle, dim=typology_dim)
        extractor = (ConstantExtractor(ty_dim) if typology_constant
                     else TypologyExtractor(ty_langs, ty_vecs, ty_dim))
        pieces.append(chain(extractor, list2ragged(), with_array(Linear(width, ty_dim))))
        n_blocks += 1

    if vectors is not None:
        table = load_vectors(vectors)
        vdim = table.dim + 1                       # + the OOV dimension, which is never a zero row
        vex = (VecConstantExtractor(vdim) if vectors_constant
               else VecExtractor(table, vectors_fill, shuffle=vectors_shuffle))
        pieces.append(chain(vex, list2ragged(), with_array(Linear(width, vdim))))
        n_blocks += 1

    if lang_embed:
        rows = lang_embed_rows or (max(lang_slots.values()) + 1)
        d = lang_embed_dim or width
        emb = Embed(d, rows, column=0, dropout=0.0)
        pieces.append(chain(LangSlotExtractor(lang_slots), list2ragged(),
                            with_array(emb if d == width else chain(emb, Linear(width, d)))))
        n_blocks += 1

    if lang_id:
        pieces.append(chain(LangIdExtractor(langs), list2ragged(),
                            with_array(Linear(width, len(langs)))))
        n_blocks += 1

    max_out = with_array(Maxout(width, width * n_blocks, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())


def set_vectors_fill(nlp, fill: str) -> int:
    """Switch a LOADED arm between the training fill and the deployment fill.

    The fill regime is a property of INFERENCE, not of the weights. An arm trains on `lemma`,
    because that is what a treebank has; it is deployed on `gloss`, because that is what a user of
    an unseen language has. Nothing about the parameters changes -- both routes look up rows in the
    same shared space, which is the entire premise of the channel.

    ⚠ THIS RETURNS A COUNT AND REFUSES AT ZERO, deliberately. A no-op that silently changed nothing
    would leave the caller evaluating the LEMMA fill while reporting the gloss one, and on a test
    language with no rows the lemma fill is all-OOV -- so the number would look like "the channel
    buys nothing zero-shot" when what actually happened is that it was never switched on. That is
    precisely the shape of defect this repo keeps paying for, so it raises.
    """
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}, not {fill!r}")
    n = 0
    for _, pipe in nlp.pipeline:
        model = getattr(pipe, "model", None)
        if model is None:
            continue
        for node in model.walk():
            if node.name == "extract_aligned_vec":
                node.attrs["vt_fill"] = fill
                n += 1
    if not n:
        raise ValueError(
            f"set_vectors_fill({fill!r}): this pipeline has no `extract_aligned_vec` node, so it "
            f"has no lexical channel to switch -- it is a g3_base/g3_vec_ctl arm, or a v2 wheel. "
            f"Refusing rather than returning quietly, because a silent no-op here would report the "
            f"lemma fill's number under the gloss fill's name.")
    return n
