"""The SUD annotation slot: `Token._.sud_misc`, plus the helpers every SUD component shares.

CoNLL-U column 10 (MISC) has no home in a spaCy `Doc`. `spacy convert --converter conllu` reads
field 10 for exactly two things -- `SpaceAfter=No` and the NER pattern -- and discards the rest
(see `spacy/training/converters/conllu_to_docs.py`), so SUD's own MISC layer (`Subject`,
`Reported`, `Idiom`, `InIdiom`) never reaches the corpora at all, let alone the models.

This module opens one slot for it. The slot is a spaCy extension, NOT `token.morph`: the
morphologiser's output stays what it was, and a predicted SUD feature never has to compete for
room with a morphological one.

WHICH COLUMN A KEY BELONGS TO IS A PROPERTY OF THE KEY, not of this slot. `Idiom`/`InIdiom`/
`Reported`/`Subject` are MISC features in every treebank here; `Shared` is a FEATS one (10 178
tokens in SUD_English-EWT train alone, all in field 6, none in field 10). The rule throughout this
project is to follow the data rather than the prose -- SUD's own guidelines list `Subject` among
the morpho-syntactic features, and the data does not -- so the two groups are declared separately
and serialised to different columns. `misc_string`/`feats_string` are what a CoNLL-U writer should
use; at runtime both groups live in the one dict.

    from sud_misc import set_misc, get_misc, misc_string, feats_string
    set_misc(token, "Idiom", "Yes")
    get_misc(token, "Idiom")      # -> "Yes" or None
    misc_string(token)            # -> "Idiom=Yes|Subject=SubjRaising", column 10
    feats_string(token)           # -> "Shared=Yes",                   column 6

The extension holds a plain `dict`. It is registered at import with a `has_extension` guard,
because loading two models in one process -- or reloading one -- imports this module twice and
`set_extension` raises on a duplicate. `force=` is deliberately not used: it would stomp a
caller's own extension of the same name.

CAVEAT for `clause_parser` arms (lzh/sa): that component rebuilds the `Doc` from scratch, so any
extension it does not explicitly copy is dropped. `sud_misc` is listed in its carry-over tuple; if
a new extension is added here, add it there too.
"""
from spacy.tokens import Token

# The SUD keys this project predicts, grouped by the CoNLL-U column they are written to.
# Order within each group is the CoNLL-U convention: alphabetical.
SUD_MISC_KEYS = ("Idiom", "InIdiom", "Reported", "Subject")     # column 10
SUD_FEATS_KEYS = ("Shared",)                                    # column 6 -- see the docstring
SUD_KEYS = tuple(sorted(SUD_MISC_KEYS + SUD_FEATS_KEYS))

# Prefix used to smuggle these keys through `spacy convert` in the FEATS column at TRAINING time
# (see scripts/hoist_sud_gold.py). It exists so a hoisted key can never be mistaken for a genuine
# morphological feature; at inference the components write to MISC, never to FEATS.
HOIST_PREFIX = "Sud"


if not Token.has_extension("sud_misc"):
    # `default=None` rather than `default={}`: a mutable default is shared by every token in the
    # process, so writing to one token's dict would write to all of them. The dict is created
    # lazily, per token, by set_misc().
    Token.set_extension("sud_misc", default=None)


def set_misc(token, key, value):
    """Record `key=value` in the token's MISC slot. A falsy value clears the key.

    Clearing never allocates: components typically call this for every token and set a value on
    very few, so the common path must not leave an empty dict on each one.
    """
    d = token._.sud_misc
    if not value:
        if d is not None:
            d.pop(key, None)
        return
    if d is None:
        d = {}
        token._.sud_misc = d
    d[key] = value


def get_misc(token, key, default=None):
    d = token._.sud_misc
    return default if d is None else d.get(key, default)


def misc_string(token, keys=SUD_MISC_KEYS):
    """Render the slot as a CoNLL-U column-10 fragment, so it can be written back out.

    Returns "" when nothing is set, which is what a caller should splice out rather than emit as
    an empty `|` field.
    """
    d = token._.sud_misc
    if not d:
        return ""
    return "|".join(f"{k}={d[k]}" for k in keys if d.get(k))


def feats_string(token, keys=SUD_FEATS_KEYS):
    """The same, for the keys that belong in column 6 (FEATS) rather than column 10.

    A writer merges this with `token.morph` -- these keys are deliberately kept OUT of `morph`
    (see the docstring), so serialising a full FEATS cell means joining the two.
    """
    d = token._.sud_misc
    if not d:
        return ""
    return "|".join(f"{k}={d[k]}" for k in keys if d.get(k))


def deprel_base(token):
    """The deprel without its `@`-suffixed deep feature (`comp:obj@x` -> `comp:obj`).

    SUD decouples the surface relation from a deep-feature layer suffixed with `@`, so any rule
    matching on a relation name has to strip it first. `unk` is bare in every treebank here, but
    the parser's label set carries subtyped relations generally, so never compare `dep_` directly.
    """
    return token.dep_.split("@", 1)[0]
