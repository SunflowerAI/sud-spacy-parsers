#!/usr/bin/env python3
"""`sud.GenericEmbed.v2` — UPOS + decomposed FEATS + four typological features. No lexical channel.

v1 read a cross-lingually aligned 128-d vector as well, and that channel was worth +8.60 macro LAS
on held-in languages. It is gone, deliberately: an aligned vector has to be trained per language
from a large monolingual corpus, which is exactly what a language with no treebank also does not
have. What is left is what a linguist can supply for a language with nothing at all --

    UPOS         17 universal categories
    FEATS        one hash table per morphological CATEGORY, not one per bundle
    typology     four 2-bit fields, 8 dims, constant across a document

-- and the point of the experiment is what that is worth zero-shot, not what it costs in-sample.

**THE TYPOLOGY ENCODING.** Four fields, two bits each, in the order
`[OV VO SV VS HM DM SEX NOSEX]`:

    11   both attested (a genuinely mixed language: German OV+VO, Arabic SV+VS, Latin double-marking)
    10   the first only          01   the second only
    00   UNKNOWN -- no evidence, not "in the middle"

The `00` convention replaces v1's separate `known` flag and keeps the same distinction, which this
repo has paid for once already: an unmeasured value that renders like a measured one cost Sanskrit
6.8 LAS through `set_morph("")`. It has one known overload -- a genuinely isolating language is
neither head- nor dependent-marking and also lands on `00` -- and `typology_dim = 12` measures what
that costs by adding one `measured` flag per field.

⚠ **WHY BINARY HERE WHEN v1 ARGUED FOR GRADED.** v1's nine graded word-order parameters binarised to
12 distinct profiles out of 13 languages -- a language identifier in disguise, and a language
identifier was measured at −0.02 macro LAS, i.e. nothing. Eight bits over ~80 training languages
collide heavily by construction (the sampler reports the rate), so languages must SHARE profile
rows rather than each getting a private one. That sharing is the only mechanism by which the channel
can transfer to a language it has never seen, so the collision is the point rather than a defect.

THREE REFUSALS, each for a failure already paid for in this repo:

  * **`Doc._.tb_lang` unset** -> raise. A default would silently give every document one profile.
  * **no profile for a language** -> raise. A neutral vector is indistinguishable from a measured one.
  * **`lang_id` with an unlisted language** -> raise. v1 emitted an all-zero row instead, which
    means `generic_langid` would run happily on a zero-shot language and score as though a language
    embedding had been consulted when no row for it exists.

TWO CONTROLS, because adding a block adds parameters and the two must be separable
(NEGATIVE-RESULTS.md, "always run a capacity control"):

    typology_constant = true   same Linear, same parameter count, every token handed zeros.
                               Isolates the PARAMETERS.
    typology_shuffle = true    the real profiles attached to the WRONG languages. Isolates whether
                               the RIGHT profile matters. This is the gate on the whole experiment.

⚠ **THE SHUFFLE IS BIT-DISTINCT, NOT MERELY A DERANGEMENT.** v1 checked only that no language kept
its own profile by INDEX, which was sufficient there because its 18-d graded profiles were all
distinct. Here they are not: with 8 bits many languages share a profile, so an index-derangement can
hand a language a bit-identical row and quietly stop being a control -- and a control that is partly
not a control can only narrow the gap it exists to measure. `derange_bits` groups languages by
profile and rotates by the largest group, which makes a bit-identical assignment impossible rather
than unlikely, and needs no RNG.

Config usage:

    [components.tok2vec.model.embed]
    @architectures = "sud.GenericEmbed.v2"
    width = ${components.tok2vec.model.encode.width}
    upos_rows = 64
    feats = ["Case", "Number", ...]
    feat_rows = [64, 16, ...]
    typology = "assets_typ/typology_v2.json"
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import List, Optional

import numpy as np
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import Embed, HashEmbed
from thinc.types import Floats2d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sud_feats_embed import FeatsFeatureExtractor  # noqa: E402

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]

#: The language a doc's tokens are in. A row-set selector for the typology lookup, never a model
#: input in the headline arm -- no parameter varies with it unless `lang_id` is on.
if not Doc.has_extension("tb_lang"):
    Doc.set_extension("tb_lang", default=None)

_TYPOLOGY: dict = {}


def derange_bits(langs, vecs):
    """Reassign profiles so that no language keeps a BIT-IDENTICAL one. Deterministic, no RNG.

    Group the languages by profile so that identical profiles are contiguous, then rotate the
    assignment by the size of the largest group. Positions `i` and `i + k` cannot fall in the same
    contiguous group when every group is at most `k` long, so the result is bit-distinct by
    construction. Impossible only if one profile is shared by more than half the languages, which
    is reported rather than worked around.
    """
    groups = collections.defaultdict(list)
    for lg in langs:
        groups[tuple(vecs[lg])].append(lg)
    order = [lg for key in sorted(groups) for lg in sorted(groups[key])]
    k = max(len(v) for v in groups.values())
    n = len(order)
    if k > n // 2:
        raise ValueError(
            f"sud.GenericEmbed.v2: cannot build a bit-distinct derangement -- {k} of {n} languages "
            f"share one profile, so at least one language must keep its own. The control would be "
            f"partly not a control; widen the profile or drop the arm.")
    out = {order[i]: vecs[order[(i + k) % n]] for i in range(n)}
    moved = sum(1 for lg in langs if tuple(out[lg]) != tuple(vecs[lg]))
    if moved != len(langs):
        raise ValueError("sud.GenericEmbed.v2: derangement left a language with its own profile")
    return out, k


def load_typology(path, shuffle: bool = False, dim: int = 8):
    """`(langs, {lang: [float, ...]}, n_dims)`, loaded once per process.

    `dim = 8` is the four 2-bit fields. `dim = 12` appends one `measured` flag per field, which is
    what separates "this language is isolating" from "nobody has told us" -- the `00` overload.
    """
    ck = (str(path), bool(shuffle), int(dim))
    if ck in _TYPOLOGY:
        return _TYPOLOGY[ck]
    f = pathlib.Path(path)
    if not f.exists():
        raise ValueError(
            f"sud.GenericEmbed.v2: typology table {path} not found. Build it with "
            f"scripts/prep_generic_v2.py. Refusing to fall back to a constant vector, which would "
            f"load cleanly and score exactly like this channel's own capacity control.")
    blob = json.loads(f.read_text(encoding="utf-8"))
    langs = sorted(blob["languages"])
    vecs = {}
    for lg in langs:
        bits = [float(b) for b in blob["languages"][lg]["bits"]]
        if len(bits) != 8:
            raise ValueError(f"sud.GenericEmbed.v2: {lg} has {len(bits)} bits, expected 8")
        if dim == 12:
            # One flag per FIELD, not per bit: a field is measured when either of its bits is set.
            bits += [float(bits[i] or bits[i + 1]) for i in (0, 2, 4, 6)]
        elif dim != 8:
            raise ValueError(f"sud.GenericEmbed.v2: typology_dim must be 8 or 12, got {dim}")
        vecs[lg] = bits
    shift = None
    if shuffle:
        vecs, shift = derange_bits(langs, vecs)
    out = (langs, vecs, dim, shift)
    _TYPOLOGY[ck] = out
    return out


def TypologyExtractor(langs, vecs, dim):
    return Model("extract_typology", _typology_forward,
                 attrs={"ty_langs": list(langs), "ty_vecs": dict(vecs), "ty_dim": int(dim)})


def _typology_forward(model: Model, docs, is_train: bool):
    vecs = model.attrs["ty_vecs"]
    out = []
    for doc in docs:
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        if not lang:
            raise ValueError(
                "sud.GenericEmbed.v2: Doc._.tb_lang is unset. The reader stamps it during training "
                "and `generic_corpus.annotate` stamps it at inference; a default would hand every "
                "document the same profile and the arm would score like its own control.")
        row = vecs.get(lang)
        if row is None:
            raise ValueError(
                f"sud.GenericEmbed.v2: no typological profile for Doc._.tb_lang={lang!r}. "
                f"Known: {' '.join(sorted(vecs))}. Refusing to substitute a neutral vector, which "
                f"would be indistinguishable from a measured one.")
        out.append(model.ops.asarray2f(np.tile(np.asarray(row, dtype="f"), (len(doc), 1))))
    return out, lambda d: []


def ConstantExtractor(dim: int):
    """The capacity control: the same width of input, carrying nothing."""
    return Model("extract_typology_constant", _constant_forward, attrs={"c_dim": int(dim)})


def _constant_forward(model: Model, docs, is_train: bool):
    dim = int(model.attrs["c_dim"])
    out = [model.ops.asarray2f(np.zeros((len(doc), dim), dtype="f")) for doc in docs]
    return out, lambda d: []


def LangSlotExtractor(slots):
    """Emit the ROW INDEX of a document's language, for a trainable per-language embedding.

    A one-hot into a Linear -- what `lang_id` does -- fixes the input width at the number of
    TRAINING languages, so a language seen for the first time at inference has nowhere to go and the
    layer can only refuse. A lookup table with SPARE ROWS can be handed a new language: assign it an
    unused slot and fit that row alone on a small annotated sample, leaving every other parameter
    frozen. That is the difference between a diagnostic and something deployable.
    """
    return Model("extract_lang_slot", _langslot_forward, attrs={"ls_slots": dict(slots)})


def _langslot_forward(model: Model, docs, is_train: bool):
    slots = model.attrs["ls_slots"]
    out = []
    for doc in docs:
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        if lang not in slots:
            raise ValueError(
                f"sud.GenericEmbed.v2: no embedding slot for {lang!r}. Assign one of the spare "
                f"rows with scripts/adapt_lang_embed.py before parsing an unseen language; a "
                f"default row would silently give it some training language's vector.")
        idx = slots[lang]
        out.append(model.ops.asarray2i(np.full((len(doc), 1), idx, dtype="i")))
    return out, lambda d: []


def LangIdExtractor(langs: List[str]):
    return Model("extract_lang_id", _langid_forward, attrs={"li_langs": list(langs)})


def _langid_forward(model: Model, docs, is_train: bool):
    langs = model.attrs["li_langs"]
    out = []
    for doc in docs:
        parent = doc if isinstance(doc, Doc) else doc.doc
        lang = parent._.tb_lang
        if lang not in langs:
            # v1 emitted an all-zero row here. That let the language-id arm run on a HELD-OUT
            # language and be reported beside the typology arm, when the honest answer is that a
            # language embedding has no row for a language it never saw. Raising makes
            # `g2_langid` a train-side control by construction.
            raise ValueError(
                f"sud.GenericEmbed.v2: lang_id is on and Doc._.tb_lang={lang!r} is not one of the "
                f"{len(langs)} training languages. A language embedding cannot serve a zero-shot "
                f"language; score this arm on held-in data only.")
        arr = np.zeros((len(doc), len(langs)), dtype="f")
        arr[:, langs.index(lang)] = 1.0
        out.append(model.ops.asarray2f(arr))
    return out, lambda d: []


@registry.architectures("sud.GenericEmbed.v2")
def GenericEmbed(
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
) -> Model[List[Doc], List[Floats2d]]:
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
        # The constant control must still be *declared*, so the arm it controls is unambiguous.
        raise ValueError("typology_shuffle/typology_constant need `typology` to be set")
    if lang_embed and lang_id:
        raise ValueError("lang_embed and lang_id are two encodings of the same thing; pick one")
    if lang_embed and not lang_slots:
        raise ValueError("lang_embed needs `lang_slots`, a {language: row} map with spare rows")
    if lang_id and not langs:
        raise ValueError(
            "lang_id = true needs an explicit `langs` list. v1 read the inventory off the vector "
            "table, which no longer exists; make_generic_config_v2.py writes it from the manifest.")

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
        ty_langs, ty_vecs, ty_dim, shift = load_typology(
            typology, shuffle=typology_shuffle, dim=typology_dim)
        if typology_constant:
            extractor = ConstantExtractor(ty_dim)
        else:
            extractor = TypologyExtractor(ty_langs, ty_vecs, ty_dim)
        pieces.append(chain(extractor, list2ragged(), with_array(Linear(width, ty_dim))))
        n_blocks += 1

    if lang_embed:
        rows = lang_embed_rows or (max(lang_slots.values()) + 1)
        # A BOTTLENECK, when `lang_embed_dim` is set: the per-language vector is `d`-dimensional and
        # a Linear widens it to the block width. Every block feeding the Maxout must be `width`
        # wide, so a narrow embedding cannot simply be concatenated.
        #
        # ⚠ A trained d-dimensional space is NOT the same thing as a 128-d space truncated to d.
        # Truncating the fitted Basque row to its top 8 principal components kept only half the
        # adaptation gain (+2.27 of +4.44), because the benefit is spread across the spectrum --
        # top-8 PCs hold 51 % of the variance. Training under the constraint forces the model to
        # compress language identity while it learns, which is the thing worth testing.
        d = lang_embed_dim or width
        # `column=0` because the extractor emits a single index column.
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


@registry.architectures("sud.GenericTagEmbed.v1")
def GenericTagEmbed(
    width: int,
    attrs: List[str] = ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
    rows: List[int] = [5000, 2500, 2500, 2500],
    lang_embed: bool = True,
    lang_embed_rows: int = 0,
    lang_embed_dim: int = 0,
    lang_slots: dict = {},
) -> Model[List[Doc], List[Floats2d]]:
    """The tagging counterpart of `sud.GenericEmbed.v2`: WORDFORM + a per-language embedding.

    The parser can be language-agnostic because UPOS and FEATS already are. A tagger cannot: its
    only input is the string, and strings do not transfer -- an English tagger reaches 17-50 % UPOS
    on the twenty held-out languages, against the ~95 % the parser needs. So this reads the form the
    ordinary way (NORM/PREFIX/SUFFIX/SHAPE) and adds the same spare-row language table, on the bet
    that eighty languages of shared orthographic evidence plus a fitted language vector does what
    one language cannot.

    ⚠ FEATS IS AN OUTPUT HERE, NOT AN INPUT. `sud.GenericEmbed.v2` takes a `feats` list and reads
    morphology off the token; this must not, or the morphologiser would be predicting its own input.
    The two architectures are deliberately separate rather than one with a flag.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"attrs has {len(attrs)} entries, rows has {len(rows)}")
    if lang_embed and not lang_slots:
        raise ValueError("lang_embed needs `lang_slots`, a {language: row} map with spare rows")

    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(attrs))]
    pieces = [chain(FeatsFeatureExtractor(attrs, []), list2ragged(),
                    with_array(concatenate(*embeddings)))]
    n_blocks = len(embeddings)

    if lang_embed:
        n_rows = lang_embed_rows or (max(lang_slots.values()) + 1)
        d = lang_embed_dim or width
        # `column=0`: the slot extractor produces its OWN single-column array in a
        # separate concat branch, so the index is not offset by the string attrs.
        emb = Embed(d, n_rows, column=0, dropout=0.0)
        pieces.append(chain(LangSlotExtractor(lang_slots), list2ragged(),
                            with_array(emb if d == width else chain(emb, Linear(width, d)))))
        n_blocks += 1

    max_out = with_array(Maxout(width, width * n_blocks, nP=3, dropout=0.0, normalize=True))
    return chain(concatenate(*pieces), max_out, ragged2list())


# --------------------------------------------------------------------------------------------
# UPOS is an INPUT to this arm, never an output.
# --------------------------------------------------------------------------------------------
#
# spaCy's morphologizer predicts a JOINT `POS=X|Feat=Val` label and, in `set_annotations`, writes
# BOTH halves: `doc.c[j].pos = labels_pos[...]` fires whenever `overwrite` is on. The bundled arm
# shipped with `overwrite = true`, so the morphologiser silently replaced the UPOS the user had
# supplied -- with its own guess -- and the parser then read that guess rather than the input. The
# whole premise of the arm is that UPOS is the one column the user provides, so this is the wheel
# discarding its only lexical evidence.
#
# The fix is `overwrite = false` (see `fix_generic_pos_write.py`), which makes the POS write
# unreachable for any token that already has one. That leaves one hole, and this component closes
# it: a token with NO UPOS would then simply be parsed on the "POS=" absent-feature row -- silently,
# and on the single input the arm cannot do without. It refuses instead (CLAUDE.md hazard 8: a
# component that silently loses an input must refuse to load).
#
# ⚠ REGISTERED ONLY ONCE. Inside the wheel this module is imported twice under two names --
# `xx_sud_generic.sud_generic_embed_v2` and, via the `sys.path` insertion above, bare
# `sud_generic_embed_v2` -- so the decorator runs twice. `registry.architectures` tolerates that
# silently; `Language.factory` raises E004 and the wheel will not import at all. Caught only by
# installing the built wheel into a clean target, never by running from the repo.
def _make_require_upos(nlp, name: str, strict: bool):
    def require_upos(doc: Doc) -> Doc:
        if not strict:
            return doc
        missing = [i for i, t in enumerate(doc) if t.pos == 0]
        if missing:
            head = ", ".join(f"{i}:{doc[i].text!r}" for i in missing[:5])
            raise ValueError(
                f"sud_require_upos: {len(missing)} of {len(doc)} tokens have no UPOS "
                f"({head}{', ...' if len(missing) > 5 else ''}). This arm READS UPOS and does not "
                f"predict it -- tagging does not transfer across languages (32-39 % on held-out "
                f"languages, no better than one English tagger), which is why the column is yours "
                f"to supply. Set `token.pos_` on every token, or "
                f"`nlp.disable_pipe('sud_require_upos')` to parse without it and accept that every "
                f"untagged token is read as category-unknown.")
        return doc
    return require_upos


try:
    from spacy.language import Language
except ImportError:                      # pragma: no cover -- thinc-only use of this module
    pass
else:
    if not Language.has_factory("sud_require_upos"):
        Language.factory("sud_require_upos",
                         default_config={"strict": True})(_make_require_upos)
