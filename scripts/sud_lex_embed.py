#!/usr/bin/env python3
"""`sud.LexFieldEmbed.v1` — one hash-embedded table per COMMA-SEPARATED XPOS FIELD, read from a
per-form lexicon rather than from a tagger.

WHY THIS EXISTS. lzh's XPOS is a four-level ontology (`v,動詞,行為,設置`) and the parser cannot see
it. Two things stand in the way of the obvious fix:

  * **The parser runs before any tagger in the arm that ships.** `package_sud.sh` refuses to package
    a pipeline whose tagger precedes its morphologiser -- that guard is what stops a pre-graft arm
    going out -- and the grafted tagger sits at the END, conditioned on UPOS+FEATS. So there is no
    predicted TAG available at the point the parser needs one, and manufacturing a first-pass tagger
    to supply it would trip the guard for real reasons.
  * **A predicted feature is not the feature.** NEGATIVE-RESULTS.md records nine languages in which
    conditioning on PREDICTED morphology lost, and the resolution: where a noisy channel enters the
    network matters more than how it is represented. Hence this layer is a SIDE CHANNEL, wrapped by
    `sud.Tok2VecPlusFeats.v1` ABOVE the encoder, so a token's own class reaches that token's
    decision and is not convolved over its neighbours.

The lexicon sidesteps both -- and is worthless for a third reason that neither of them shares.

⚠ WHAT IT ACTUALLY BUYS: NOTHING, BY CONSTRUCTION. A per-form majority IS the tag for practical
purposes, which is precisely the problem: it is a FUNCTION OF THE FORM, so conditioning on
(form, f(form)) is conditioning on form, and the parser already reads the form. Measured: 0.0000
bits about the relation beyond `NORM`, against gold XPOS's 0.2222 and a predicted tagger's 0.1475;
-0.19 LAS against its capacity control. The useful part of XPOS is the WITHIN-form variation
(58.8 % of lzh tokens have an XPOS-ambiguous form) and a majority table destroys exactly that.
The `tag` source below exists because only a per-TOKEN value can carry any of it -- and that was
measured too, at -0.25 LAS, for the separate reason in the second bullet above.
See NEGATIVE-RESULTS.md, "XPOS as a parser input, and kanripo vectors, for lzh".

⚠ THE TABLE MUST TRAVEL INSIDE THE MODEL. thinc's `Model.to_dict` serialises `attrs` through
`serialize_attr` and **skips anything that raises TypeError, silently** -- so a table that is not
msgpack-clean would vanish on save and the layer would come back answering `<OOV>` for every token,
loading cleanly and scoring like a capacity control. That is standing hazard 8 (a component that
silently loses an input must refuse to load) in its exact shape. The table is therefore a plain
dict of lists of ints, and the forward pass RAISES rather than degrading when it is empty.
`check_lex_embed.py` verifies the round trip.

⚠ AN OOV FORM AND A FORM WITH NO MAJORITY MUST BE THE SAME INPUT. Both take code -1 and therefore
the same hashed symbol, for the same reason an unset MORPH and an empty one must agree (CLAUDE.md;
it cost sa 6.8 LAS). There is no second sentinel to get wrong.

Config usage -- the side channel goes on the PARSER, above its listener:

    [components.parser.model.tok2vec]
    @architectures = "sud.Tok2VecPlusFeats.v1"

    [components.parser.model.tok2vec.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = 96
    upstream = "tok2vec"

    [components.parser.model.tok2vec.feats_embed]
    @architectures = "sud.LexFieldEmbed.v1"
    width = 32
    table = "models/lzh_xpos_lex.json"
    fields = [0, 1, 2]
    rows = [8, 16, 64]
"""
import json
import pathlib
from typing import Callable, List, Optional, Tuple

from spacy.strings import hash_string
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Maxout, Model, chain, concatenate, list2ragged, ragged2list, with_array
from thinc.layers import HashEmbed
from thinc.types import Floats2d, Ints2d, Ragged

# (field index, code) -> hash. Module level and not a Model attr, for the same reason the feats
# cache is: thinc would try to serialise a Model attr, and this is a pure function of its key.
_SYM_CACHE = {}

# The one sentinel. A form absent from the table and a form with no majority both land here.
OOV = -1

# Where the field values come from.
#   "lexicon"  a per-form majority table  -- PROVEN INFORMATION-FREE, kept for the record
#   "tag"      the token's own predicted TAG, split on commas
#
# ⚠ THE LEXICON SOURCE CANNOT WORK, AND THE REASON IS AN IDENTITY, NOT A MEASUREMENT. A per-form
# majority table is a DETERMINISTIC FUNCTION OF THE FORM, so conditioning on (form, f(form)) is
# conditioning on form -- and the parser already holds the form in NORM. Measured on lzh test with
# one estimator throughout: gold XPOS is worth 0.2222 bits about the relation beyond the form, a
# predicted tagger 0.1475, and the lexicon 0.0000. Exactly zero, as it must be.
#
# The useful part of XPOS is the part a majority table destroys: 58.8 % of lzh tokens have a form
# whose XPOS field 2 varies (H(XPOS field2 | form) = 0.2808 bits), and ALL of the parsing signal
# lives in that within-form variation. Hence "tag": only a per-TOKEN source can carry any of it.
SOURCES = ("lexicon", "tag")


def _sym(field: int, code: int) -> int:
    ck = (field, code)
    got = _SYM_CACHE.get(ck)
    if got is None:
        got = hash_string(f"xpos{field}=<OOV>" if code == OOV else f"xpos{field}={code}")
        _SYM_CACHE[ck] = got
    return got


def _load_table(table) -> dict:
    """`table` is a path, an already-loaded dict, or None (constant/control mode)."""
    if table is None:
        return {}
    if isinstance(table, dict):
        return table
    p = pathlib.Path(table)
    if not p.exists():
        raise ValueError(
            f"sud.LexFieldEmbed.v1: lexicon {p} not found. Build it with "
            f"scripts/build_xpos_lexicon.py; the layer will not fall back to <OOV> for every "
            f"token, because that loads cleanly and scores like its own capacity control.")
    return json.loads(p.read_text(encoding="utf-8"))


def LexFeatureExtractor(table: dict, fields: List[int], constant: bool, source: str):
    return Model("extract_lex_fields", _lex_forward,
                 attrs={"lex_table": table, "lex_fields": list(fields),
                        "lex_constant": bool(constant), "lex_source": source})


def _tag_sym(field: int, tag: str) -> int:
    """The token's own TAG, split on commas, one column per field.

    An unset TAG, a TAG with too few fields, and a token the tagger declined to tag must all be the
    SAME input -- there is one sentinel and no second one to get wrong, for the reason an unset
    MORPH and an empty one must agree (it cost sa 6.8 LAS)."""
    parts = tag.split(",") if tag else ()
    if field >= len(parts):
        return _sym(field, OOV)
    ck = (field, parts[field])
    got = _SYM_CACHE.get(ck)
    if got is None:
        got = hash_string(f"xpos{field}={parts[field]}")
        _SYM_CACHE[ck] = got
    return got


def _lex_forward(model: Model, docs, is_train: bool) -> Tuple[List[Ints2d], Callable]:
    table = model.attrs["lex_table"]
    fields = model.attrs["lex_fields"]
    constant = model.attrs["lex_constant"]
    source = model.attrs.get("lex_source", "lexicon")
    xp = model.ops.xp

    if not constant and source == "tag":
        features = []
        for doc in docs:
            toks = list(doc)
            arr = xp.zeros((len(toks), len(fields)), dtype="uint64")
            for i, tok in enumerate(toks):
                arr[i] = [_tag_sym(f, tok.tag_) for f in fields]
            features.append(model.ops.asarray2i(arr, dtype="uint64"))
        return features, lambda d_features: []

    if not constant and not table.get("full"):
        raise ValueError(
            "sud.LexFieldEmbed.v1: the lexicon is empty. Either the table never reached the "
            "model, or it was dropped on serialisation (thinc skips an unserialisable attr "
            "without saying so). Refusing to run: every token would read <OOV>.")

    k = int(table.get("k", 0) or 0)
    lut = table.get("full", {})
    folds = table.get("folds", [])
    const_row = None
    if constant:
        # The capacity control: identical column count, identical parameter count, zero information.
        const_row = xp.asarray([_sym(f, 0) for f in fields], dtype="uint64")

    features: List[Ints2d] = []
    for doc in docs:
        toks = list(doc)
        arr = xp.zeros((len(toks), len(fields)), dtype="uint64")
        if constant:
            if toks:
                arr[:] = const_row
            features.append(model.ops.asarray2i(arr, dtype="uint64"))
            continue
        # Jackknifing: while training, read the table built from the OTHER folds, so a form seen
        # once is <OOV> to the parser exactly as it will be to a reader of unseen text. At
        # inference the full table is used -- the model is then served a CLEANER channel than it
        # was trained on, which is the safe direction for the asymmetry to run.
        diff = None
        if is_train and k and folds:
            diff = folds[hash_string("".join(t.text for t in toks)) % k]
        for i, tok in enumerate(toks):
            codes = diff.get(tok.text) if diff is not None else None
            if codes is None:
                codes = lut.get(tok.text)
            if codes is None:
                arr[i] = [_sym(f, OOV) for f in fields]
            else:
                arr[i] = [_sym(f, codes[f]) for f in fields]
        features.append(model.ops.asarray2i(arr, dtype="uint64"))

    backprop: Callable[[List[Ints2d]], List] = lambda d_features: []
    return features, backprop


@registry.architectures("sud.LexFieldEmbed.v1")
def LexFieldEmbed(
    width: int,
    fields: List[int],
    rows: List[int],
    table: Optional[str] = None,
    constant: bool = False,
    source: str = "lexicon",
) -> Model[List[Doc], List[Floats2d]]:
    """One hash-embedded table per configured XPOS field, looked up per form.

    `constant = true` is the capacity control: same columns, same rows, same Maxout, every token
    given one symbol per column. Without it a gain cannot be told from the extra parameters -- the
    la arm in NEGATIVE-RESULTS.md scored 0.5 apart from an architecturally IDENTICAL twin.
    """
    if len(rows) != len(fields):
        raise ValueError(f"Mismatched lengths: {len(rows)} rows vs {len(fields)} fields")
    if not fields:
        raise ValueError("sud.LexFieldEmbed.v1 needs at least one field")
    if any(r < 1 for r in rows):
        raise ValueError("rows must be >= 1")
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate field in {fields}")

    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    tbl = {} if (constant or source == "tag") else _load_table(table)
    if not constant and source == "lexicon":
        n = int(tbl.get("n_fields", 0))
        bad = [f for f in fields if not 0 <= f < n]
        if bad:
            raise ValueError(f"fields {bad} outside the table's 0..{n - 1} ({tbl.get('source')})")

    # seed 7 and the same increment order as spacy.MultiHashEmbed, so a table here is seeded like
    # the corresponding table there and the two layers stay comparable.
    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, rows[index], column=index, seed=seed, dropout=0.0)

    embeddings = [make_hash_embed(i) for i in range(len(fields))]
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, width * len(embeddings), nP=3, dropout=0.0, normalize=True)
    )
    return chain(
        LexFeatureExtractor(tbl, fields, constant, source),
        list2ragged(),
        with_array(concatenate(*embeddings)),
        max_out,
        ragged2list(),
    )
