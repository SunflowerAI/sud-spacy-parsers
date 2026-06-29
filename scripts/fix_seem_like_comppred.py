#!/usr/bin/env python3
"""Re-annotate predicative `like`/`as if`/`as though` complements of appearance
(linking) verbs in SUD English-EWT as comp:pred.

Motivation
----------
SUD-EWT labels the predicative slot of seem/look/appear/sound/feel/smell/taste
as comp:pred everywhere it is an adjective ("seem interested") or a `like`-PP
under the copula `be` ("it's like paradise") -- but it leaves the *same* slot
under the appearance verb itself as `udep` (the `like`+NP case) or `mod` (the
clausal `like S` / `as if S` / `as though S` case). Those are complements, not
modifiers. This script corrects them to comp:pred.

Scope (only genuine predicative complements; conditional `if` and concessive
`though` adjuncts are deliberately left as `mod`):
  A. FORM=like, UPOS=ADP    under an appearance VERB   (was udep)  -> comp:pred
  B. FORM=like, UPOS=SCONJ  under an appearance VERB   (was mod)   -> comp:pred
  C. FORM=as,   UPOS=SCONJ  under an appearance VERB AND has an
     `if`/`though` child  (i.e. `as if`/`as though`)  (was mod)   -> comp:pred

The match keys on FORM/UPOS/head-lemma/children, all of which are invariant
under the udep relabel pipeline, so the same correction lands identically on the
base, *.relabeled and *.relabeled_ext files. Idempotent. Byte-identical except
the DEPREL cell of matched tokens.
"""
import sys

APPEARANCE = {"seem", "look", "appear", "sound", "feel", "smell", "taste"}
TARGET = "comp:pred"


def is_token(cols):
    return len(cols) >= 8 and "-" not in cols[0] and "." not in cols[0]


def fix_block(block):
    lines = block.split("\n")
    rows = [(i, l.split("\t")) for i, l in enumerate(lines)]
    toks = [(i, c) for i, c in rows if is_token(c)]
    idx = {c[0]: c for _, c in toks}
    children = {}
    for _, c in toks:
        children.setdefault(c[6], []).append(c)

    changes = []  # (category, old_deprel, form)
    for line_i, c in toks:
        head = idx.get(c[6])
        if not head or head[3] != "VERB" or head[2].lower() not in APPEARANCE:
            continue
        form, upos = c[1].lower(), c[3]
        cat = None
        if form == "like" and upos == "ADP":
            cat = "A:like+NP"
        elif form == "like" and upos == "SCONJ":
            cat = "B:like+clause"
        elif form == "as" and upos == "SCONJ":
            kid_forms = {k[1].lower() for k in children.get(c[0], [])}
            if kid_forms & {"if", "though"}:
                cat = "C:as if/though"
        if cat is None or c[7] == TARGET:
            continue
        changes.append((cat, c[7], c[1]))
        c[7] = TARGET
        lines[line_i] = "\t".join(c)
    return "\n".join(lines), changes


def fix_file(path):
    text = open(path, encoding="utf-8").read()
    blocks = text.split("\n\n")
    all_changes = []
    for bi, b in enumerate(blocks):
        new_b, ch = fix_block(b)
        blocks[bi] = new_b
        all_changes.extend(ch)
    if all_changes:
        open(path, "w", encoding="utf-8").write("\n\n".join(blocks))
    return all_changes


if __name__ == "__main__":
    from collections import Counter

    for path in sys.argv[1:]:
        ch = fix_file(path)
        by_cat = Counter(c[0] for c in ch)
        by_old = Counter(c[1] for c in ch)
        print(f"{path}: {len(ch)} changed  {dict(by_cat)}  from {dict(by_old)}")
