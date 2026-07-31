"""The MISC slot: `Token._.sud_misc`, plus the helpers every SUD-MISC component shares.

CoNLL-U column 10 (MISC) has no home in a spaCy `Doc`. `spacy convert --converter conllu` reads
field 10 for exactly two things -- `SpaceAfter=No` and the NER pattern -- and discards the rest
(see `spacy/training/converters/conllu_to_docs.py`), so SUD's own MISC layer (`Subject`,
`Reported`, `Idiom`, `InIdiom`) never reaches the corpora at all, let alone the models.

This module opens one slot for it and keeps it strictly separate from FEATS. Nothing here touches
`token.morph`: the morphologiser's output stays byte-for-byte what it was, and a MISC feature never
masquerades as a morphological one. The treebanks in this project put all four keys in MISC, and
that is what the released models emit -- note that SUD's own guidelines list `Subject` among the
morpho-syntactic (FEATS) features, so the data and the prose disagree here; we follow the data.

    from sud_misc import set_misc, get_misc, misc_string
    set_misc(token, "Idiom", "Yes")
    get_misc(token, "Idiom")      # -> "Yes" or None
    misc_string(token)            # -> "Idiom=Yes|Subject=SubjRaising", CoNLL-U order

The extension holds a plain `dict`. It is registered at import with a `has_extension` guard,
because loading two models in one process -- or reloading one -- imports this module twice and
`set_extension` raises on a duplicate. `force=` is deliberately not used: it would stomp a
caller's own extension of the same name.

CAVEAT for `clause_parser` arms (lzh/sa): that component rebuilds the `Doc` from scratch, so any
extension it does not explicitly copy is dropped. `sud_misc` is listed in its carry-over tuple; if
a new extension is added here, add it there too.
"""
from spacy.tokens import Token

# The SUD MISC keys this project predicts. Order is the CoNLL-U convention: alphabetical.
SUD_MISC_KEYS = ("Idiom", "InIdiom", "Reported", "Subject")

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


def deprel_base(token):
    """The deprel without its `@`-suffixed deep feature (`comp:obj@x` -> `comp:obj`).

    SUD decouples the surface relation from a deep-feature layer suffixed with `@`, so any rule
    matching on a relation name has to strip it first. `unk` is bare in every treebank here, but
    the parser's label set carries subtyped relations generally, so never compare `dep_` directly.
    """
    return token.dep_.split("@", 1)[0]
