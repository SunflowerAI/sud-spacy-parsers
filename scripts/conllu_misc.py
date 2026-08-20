#!/usr/bin/env python3
"""Read and write the CoNLL-U MISC column when a VALUE can be the field separator itself.

WHY THIS EXISTS. The Sanskrit daṇḍa IS the character `|`, and CoNLL-U separates MISC attributes
with `|`. A daṇḍa token's padapāṭha form is therefore written `Unsandhied=|`, and the obvious
reader — `col.split("|")` — hands back the EMPTY string, because `"Unsandhied=|".split("|")` is
`['Unsandhied=', '']`. Since `Unsandhied` sorts last in these treebanks, the daṇḍa is almost always
the final attribute on the line, so the pipe that swallows it is the one immediately before the
newline. Every sa script parsed MISC the naive way, so the daṇḍa's value was silently lost on the
way in — and `make_unsandhi_corpus.py`, which parks the value in the LEMMA column, wrote it out as
an EMPTY FIELD: a malformed CoNLL-U row, not merely a missing annotation.

WHY THE VALUE IS RECOVERABLE. An empty attribute is not legal MISC — UD requires each
`|`-separated item to be non-empty — so an empty item can only have come from a literal `|` at the
end of the PRECEDING value. Folding each empty item back into its predecessor inverts the ambiguity
exactly:

    Unsandhied=|                 -> {'Unsandhied': '|'}
    Unsandhied=||SpaceAfter=No   -> {'Unsandhied': '|', 'SpaceAfter': 'No'}
    Gloss=.|Unsandhied=|         -> {'Gloss': '.', 'Unsandhied': '|'}

Checked against every sa, UFAL, DCS and SUD corpus in the repo: not one MISC field contains an
empty item today, so the rule can only ever fire on a value that genuinely ends in a daṇḍa. It does
NOT recover a `|` in the MIDDLE of a value (`Unsandhied=|x` is indistinguishable from a bare item
`x`, and no such value exists), and it deliberately does not invent an escape convention — these
files have to stay readable by tools that are not this one.

WRITING MATTERS AS MUCH AS READING. `misc_set` re-emits a `|` value verbatim, in the same encoding
the treebanks use. The reason to route writes through here is that the naive
`split("|")` + `"|".join(...)` rewrite DELETES a daṇḍa value — it drops the empty item — so a
script that only meant to touch `SpaceAfter` would destroy gold it never looked at.
"""


def misc_items(col):
    """The MISC (or FEATS) attributes of `col`, in order, with a `|`-valued attribute kept whole.

    Returns `[]` for the `_` / empty column. A leading empty item has no predecessor to attach to,
    so it is malformed either way and is dropped.
    """
    if col in ("_", ""):
        return []
    items = []
    for part in col.split("|"):
        if part == "":
            if items:
                items[-1] += "|"      # the separator was a literal daṇḍa ending the previous value
        else:
            items.append(part)
    return items


def misc_dict(col):
    """`col` as a key -> value mapping. Items with no `=` are dropped, as every caller expects."""
    out = {}
    for item in misc_items(col):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def misc_get(col, key, default=None):
    """The value of one MISC key, or `default` when the key is absent."""
    for item in misc_items(col):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return default


def misc_join(items):
    """Serialise attributes back to a MISC column, keeping CoNLL-U's `_` for "nothing here"."""
    return "|".join(items) if items else "_"


def misc_set(col, key, value):
    """Set — or, with `value=None`, remove — one MISC key, keeping every other attribute intact."""
    parts = [i for i in misc_items(col) if not i.startswith(key + "=")]
    if value is not None:
        parts.append(f"{key}={value}")
    return misc_join(parts)


def misc_has(col, item):
    """Whether a bare attribute (`SpaceAfter=No`) is present, without tripping over a daṇḍa value."""
    return item in misc_items(col)
