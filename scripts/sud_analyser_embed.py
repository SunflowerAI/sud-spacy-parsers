#!/usr/bin/env python3
"""`sud.AnalyserFeatsEmbed.v1` — `MultiHashEmbed` plus a MULTI-HOT block of morphological
CANDIDATE SETS, read from a shipped analyser table rather than predicted by a component.

WHY THIS LAYER EXISTS. `eval_sa_oracle_noise.py` located the parser's headroom precisely: gold FEATS
is worth **+5.76 LAS** over the morphologiser's FEATS, and the loss is not the morphologiser being
unhelpful — it is the morphologiser being *confidently wrong* on `Case` 14.6 % of the time, which
inverts the attachment the parser is deciding. A rule-based analyser cannot make that error in the
same way: it returns the SET of analyses a form can have, so it is either right, or honest about not
knowing. On the same tokens the intersected vidyut ∩ Heritage table is confidently wrong 1.5 % of
the time and silent 2.5 %, with the gold value in the set 95.4 % of the time.

WHY MULTI-HOT AND NOT A HASH OF THE SET. `sud.MultiHashEmbedFeats.v1` would read a candidate set
today with no code change — it joins multi-valued features with commas, so a token stamped
`Case=Nom,Voc` already works. But a hash makes `Nom,Voc` an unrelated symbol to `Nom`, discarding
the subset structure that is the whole content of a constraint. That is the block-vs-decomposed
mistake this project already paid for, one level up. Here `Case` is eight bits plus a silent bit, so
`{Nom,Voc}` is literally `Nom` + `Voc` and a set the model never met still lands on values it knows.

WHY NO JACKKNIFING, unlike `sud_lex_embed.py`. That table is derived FROM the treebank, so a form
seen once would be answered in training and `<OOV>` at inference, and its folds exist to remove the
skew. This table comes from outside the treebank, so the training-time and inference-time answers
for a form are the same answer. There is no leakage to fold away.

⚠ THE KEY IS `token.norm_`, THE PREDICTED PADAPĀṬHA. Both analysers store pre-pausal forms, so the
lookup key is what `sud_unsandhi` produces, which `scripts/make_norm_corpus.py` writes into NORM.
An arm using this layer is therefore only deployable once the TOKENISER also writes `token.norm_` —
the same prerequisite `configs/config_sa_mwt_norm.cfg` carries. If it is not written, every token
falls to the silent bit and the layer scores like its own capacity control.

⚠ THE TABLE MUST TRAVEL INSIDE THE MODEL. thinc's `Model.to_dict` serialises `attrs` through
`serialize_attr` and skips anything raising TypeError **silently**, so an unserialisable table would
vanish on save and the layer would reload answering "silent" for every token — loading cleanly and
scoring like a control. It is therefore a plain dict of lists of ints, the forward pass RAISES
rather than degrading when it is empty, and `check_analyser_embed.py` verifies the round trip.

`constant = true` is the capacity control: identical parameter count, every token given the silent
pattern, so an arm can be switched over and the gain attributed to the INFORMATION rather than to
the extra projection.

Config usage:

    [components.tok2vec.model.embed]
    @architectures = "sud.AnalyserFeatsEmbed.v1"
    width = 96
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "MORPH"]
    rows  = [5000, 1000, 2500, 2500, 64]
    table = "scripts/sa_analyser_lut.json.gz"
    feats = ["Case", "Number", "Gender", "Person"]
    include_static_vectors = false
"""
import gzip
import json
import os
import pathlib
from typing import Callable, List, Tuple, Union

from spacy.ml.staticvectors import StaticVectors
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Linear, Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import HashEmbed
from thinc.types import Floats2d, Ragged

from spacy.ml.models.tok2vec import FeatureExtractor
from sud_affix_embed import AffixFeatureExtractor


def load_table(table) -> dict:
    if table is None:
        return {}
    if isinstance(table, dict):
        return table
    p = pathlib.Path(table)
    if not p.exists():
        raise ValueError(
            f"sud.AnalyserFeatsEmbed.v1: analyser table {p} not found. Build it with "
            f"scripts/build_analyser_lexicon.py. The layer will NOT fall back to 'silent' for "
            f"every token, because that loads cleanly and scores like its own capacity control.")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return json.load(f)



# ---------------------------------------------------------------------------------------------
# RUNTIME MODE: ask vidyut per token instead of shipping a frozen extract of it.
#
# The frozen table has a structural ceiling. Its key set is whatever vocabulary happened to be
# probed, so on the Vedic test 6.5 % of tokens miss it — and the analysers recognise 75.6 % of those
# forms perfectly well, they were simply never asked. Widening the extract does not fix this:
# adding DCS's 108 888 forms (4x the table) moved honest coverage 86.7 % -> 87.5 %, because DCS is
# classical and the test is Vedic. It is a vocabulary MISMATCH, not a vocabulary shortage.
#
# Calling vidyut live removes the key set entirely, and removes the train/deploy skew with it:
# training and inference then run the same lookup, so there is no frozen artefact to drift.
# The trade is Heritage, which cannot be a runtime dependency (`sanskrit_parser` pins
# werkzeug==2.1.2 and pulls flask, pandas and sqlalchemy). Measured cost of dropping it, on Case:
# pinned-and-right 0.493 -> 0.314, in-set recall 0.951 -> 0.945, confidently-wrong 0.014 -> 0.010.
# Looser sets, no less safe.
#
# The data bundle is NOT redistributed: `vidyut` is declared as a requirement and the user runs
# `vidyut.download_data(...)` once. VIDYUT_DATA points at it.

_KOSHA = {}
_BITS_CACHE = {}

VIBHAKTI = {"praTamA": "Nom", "dvitIyA": "Acc", "tftIyA": "Ins", "caturTI": "Dat",
            "paYcamI": "Abl", "zazWI": "Gen", "saptamI": "Loc", "samboDanam": "Voc"}
LINGA = {"puM": "Masc", "strI": "Fem", "napuMsaka": "Neut"}
VACANA = {"eka": "Sing", "dvi": "Dual", "bahu": "Plur"}
# ⚠ Purusha.praTama is the THIRD person; Vibhakti.praTamA is the NOMINATIVE. Same name, different
# category — one shared dict would stamp Case=Nom on every finite verb and nothing would raise.
PURUSHA = {"praTama": "3", "maDyama": "2", "uttama": "1"}

DEFAULT_VALUES = {"Case": ["Nom", "Acc", "Ins", "Dat", "Abl", "Gen", "Loc", "Voc"],
                  "Number": ["Sing", "Dual", "Plur"],
                  "Gender": ["Masc", "Fem", "Neut"],
                  "Person": ["1", "2", "3"]}


def kosha_path(path=None):
    return path or os.environ.get("VIDYUT_DATA") or "vidyut-data/kosha"


def get_kosha(path=None):
    p = kosha_path(path)
    if p not in _KOSHA:
        try:
            from vidyut.kosha import Kosha
        except ImportError as e:
            raise ValueError(
                "sud.AnalyserFeatsEmbed.v1 is in runtime mode but `vidyut` is not installed. "
                "Install it (`pip install vidyut`) and fetch its data once with "
                "`python -c \"import vidyut; vidyut.download_data('vidyut-data')\"`. Refusing to "
                "continue: without it every token reads 'silent' and the model quietly parses "
                "worse instead of failing.") from e
        if not pathlib.Path(p).exists():
            raise ValueError(
                f"sud.AnalyserFeatsEmbed.v1: vidyut data not found at {p!r}. Set VIDYUT_DATA or run "
                f"`python -c \"import vidyut; vidyut.download_data('vidyut-data')\"`.")
        _KOSHA[p] = Kosha(p)
    return _KOSHA[p]


def _variants(slp1):
    """Kosha stores pre-pausal forms: a visarga-final word lives under `s` or `r`."""
    out = [slp1]
    if slp1.endswith("H"):
        out += [slp1[:-1] + "s", slp1[:-1] + "r"]
    if slp1.endswith("M"):
        out += [slp1[:-1] + "m"]
    return out


def analyse(form_iast, values, feats, path=None):
    """IAST padapāṭha -> {feature: [bit indices]}. Cached; the corpus has ~37k distinct norms, so
    after one epoch this is pure dict access."""
    ck = (kosha_path(path), form_iast)
    got = _BITS_CACHE.get(ck)
    if got is not None:
        return got
    from vidyut import lipi
    kosha = get_kosha(path)
    sets = {f: set() for f in feats}
    for v in _variants(lipi.transliterate(form_iast, lipi.Scheme.Iast, lipi.Scheme.Slp1)):
        for e in kosha.get(v):
            if "Case" in sets and getattr(e, "vibhakti", None) is not None:
                sets["Case"].add(VIBHAKTI.get(str(e.vibhakti)))
            if "Gender" in sets and getattr(e, "linga", None) is not None:
                sets["Gender"].add(LINGA.get(str(e.linga)))
            if "Number" in sets and getattr(e, "vacana", None) is not None:
                sets["Number"].add(VACANA.get(str(e.vacana)))
            if "Person" in sets and getattr(e, "purusha", None) is not None:
                sets["Person"].add(PURUSHA.get(str(e.purusha)))
    got = {f: sorted(values[f].index(v) for v in s if v in values[f]) for f, s in sets.items()}
    got = {f: idx for f, idx in got.items() if idx}
    _BITS_CACHE[ck] = got
    return got


def _layout(values: dict, feats: List[str]):
    """Bit offsets. Each feature gets len(values[f]) value bits plus ONE 'analyser silent' bit —
    a feature the analyser did not offer and a feature it offered every value for must not be the
    same input, for the same reason an unset MORPH and an empty one must not be (CLAUDE.md)."""
    off, n = {}, 0
    for f in feats:
        off[f] = n
        n += len(values[f]) + 1
    return off, n


def AnalyserExtractor(payload: dict, feats: List[str], constant: bool):
    return Model("extract_analyser_sets", _analyser_forward,
                 attrs={"an_payload": payload, "an_feats": list(feats),
                        "an_constant": bool(constant)})


def _analyser_forward(model: Model, docs, is_train: bool) -> Tuple[List[Floats2d], Callable]:
    payload = model.attrs["an_payload"]
    feats = model.attrs["an_feats"]
    constant = model.attrs["an_constant"]
    xp = model.ops.xp

    table = payload.get("table") or {}
    values = payload.get("values") or {}
    live = bool(payload.get("kosha_mode"))
    if not constant and not live and not table:
        raise ValueError(
            "sud.AnalyserFeatsEmbed.v1: the analyser table is empty. Either it never reached the "
            "model, or it was dropped on serialisation (thinc skips an unserialisable attr without "
            "saying so). Refusing to run: every token would read 'silent'.")
    off, n_dims = _layout(values, feats) if values else ({f: i for i, f in enumerate(feats)}, len(feats))

    out: List[Floats2d] = []
    for doc in docs:
        toks = list(doc)
        arr = xp.zeros((len(toks), n_dims), dtype="f")
        for i, tok in enumerate(toks):
            if constant:
                bits = None
            elif live:
                bits = analyse(tok.norm_, values, feats, payload.get("kosha"))
            else:
                bits = table.get(tok.norm_)
            for f in feats:
                base = off[f]
                idx = (bits or {}).get(f)
                if idx:
                    for j in idx:
                        arr[i, base + j] = 1.0
                else:
                    arr[i, base + len(values.get(f, []))] = 1.0     # the silent bit
        out.append(model.ops.asarray2f(arr))

    backprop: Callable[[List[Floats2d]], List] = lambda d: []
    return out, backprop


@registry.architectures("sud.AnalyserFeatsEmbed.v1")
def AnalyserFeatsEmbed(
    width: int,
    attrs: Union[List[str], List[int], List[Union[str, int]]],
    rows: List[int],
    include_static_vectors: bool,
    table=None,
    feats: List[str] = [],
    values: dict = {},
    kosha=None,
    runtime: bool = False,
    constant: bool = False,
    suffixes: List[int] = [],
    suffix_rows: List[int] = [],
    prefixes: List[int] = [],
    prefix_rows: List[int] = [],
) -> Model[List[Doc], List[Floats2d]]:
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"duplicate feature in {feats}")
    # `runtime = true` is the SHIPPING switch, and `kosha` stays null in a published config: the
    # path is resolved per machine (VIDYUT_DATA, else ./vidyut-data/kosha) at forward time, so a
    # wheel does not carry a path that only existed on the training machine — the same failure the
    # frozen `table` path had.
    live = runtime or kosha is not None
    payload = {"kosha_mode": True, "kosha": kosha} if live else load_table(table)
    if live and not values:
        values = DEFAULT_VALUES
    # GEOMETRY FROM CONFIG, DATA FROM BYTES. On a user's machine spaCy rebuilds the architecture
    # from config and only then loads the weights, so `table` — a path that existed on the training
    # machine — is gone. `values` carries the bit layout (a few closed lists, tens of bytes) so the
    # layer can always be CONSTRUCTED; the 32 507-form table itself arrives via from_bytes, and the
    # forward pass still refuses to run on an empty one. Packaging sets `table = null` and keeps
    # `values`, so the same config works before and after.
    if values:
        payload = dict(payload)
        payload["values"] = {f: list(v) for f, v in values.items()}
    values = payload.get("values") or {}
    if feats and not values:
        raise ValueError(
            "sud.AnalyserFeatsEmbed.v1: neither `values` nor a readable `table` was given, so the "
            "multi-hot block has no bit layout. Set `values` in the config — it is what makes the "
            "layer loadable away from the machine that built the table.")
    missing = [f for f in feats if f not in values]
    if feats and missing:
        raise ValueError(
            f"sud.AnalyserFeatsEmbed.v1: {missing} absent from the table's value inventory "
            f"{sorted(values)}. A feature with no value list has no bit layout, and silently "
            f"widening the block would misalign every feature after it.")
    _, n_dims = _layout(values, feats) if feats else ({}, 0)

    # seed 7 and the same increment order as MultiHashEmbed, so the hash columns are seeded
    # identically to stock and an arm switched over with feats=[] stays single-variable.
    seed = 7

    def make_hash_embed_row(index, all_rows):
        nonlocal seed
        seed += 1
        return HashEmbed(width, all_rows[index], column=index, seed=seed, dropout=0.0)

    # The shipped sa arm uses per-component affix windows (suffix 5 / 8 000 rows, worth +1.19
    # morph and +1.60 lemma). Dropping them to make room for the analyser channel would confound
    # the two changes, so this layer carries both and delegates the affix columns to the module
    # that already implements and verifies them.
    if len(suffix_rows) != len(suffixes) or len(prefix_rows) != len(prefixes):
        raise ValueError("affix lengths and row counts must match")
    all_rows = list(rows) + list(suffix_rows) + list(prefix_rows)
    embeddings = [make_hash_embed_row(i, all_rows) for i in range(len(all_rows))]
    n_blocks = len(embeddings) + include_static_vectors + bool(feats)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, width * n_blocks, nP=3, dropout=0.0, normalize=True)
    )
    extras = []
    if include_static_vectors:
        extras.append(StaticVectors(width, dropout=0.0))
    if feats:
        extras.append(chain(
            AnalyserExtractor(payload, feats, constant),
            list2ragged(),
            with_array(Linear(width, n_dims)),
        ))
    if not extras:
        # The flat chain, node for node as spacy.MultiHashEmbed.v2 builds it — not merely an
        # equivalent nesting. `check_analyser_embed.py` asserts byte identity, so the claim that
        # switching an arm over with feats=[] is a no-op is checkable rather than argued.
        return chain(
            FeatureExtractor(list(attrs)) if not (suffixes or prefixes)
            else AffixFeatureExtractor(list(attrs), suffixes, prefixes),
            list2ragged(),
            with_array(concatenate(*embeddings)),
            max_out,
            ragged2list(),
        )
    hashed: Model[List[Doc], Ragged] = chain(
        FeatureExtractor(list(attrs)) if not (suffixes or prefixes)
        else AffixFeatureExtractor(list(attrs), suffixes, prefixes),
        list2ragged(),
        with_array(concatenate(*embeddings)),
    )
    return chain(concatenate(hashed, *extras), max_out, ragged2list())
