#!/usr/bin/env python3
"""Hoist SUD's own annotation keys into the FEATS column so they survive `spacy convert`.

`spacy convert --converter conllu` reads MISC for exactly two things -- `SpaceAfter=No` and the NER
pattern -- and discards the rest (`spacy/training/converters/conllu_to_docs.py`). FEATS, by
contrast, goes straight into `token.morph`. So the only way to get `Subject=`/`Reported=` gold into
a `.spacy` corpus without writing a replacement converter is to move it to field 6 first.

THIS IS A TRAINING-TIME TRANSPORT ONLY, and the `Sud` prefix is what makes it unmistakable:
`Subject=SubjRaising` becomes `SudSubject=SubjRaising`. A prefixed key can never be read as a
genuine morphological feature, and `sud_tagger` reads only prefixed keys, so it cannot train on
real FEATS by accident. At inference the components write to `Token._.sud_misc`, and `sud_misc`
decides which CoNLL-U column each key is serialised back to.

TWO SOURCE COLUMNS, because the treebanks use both. `Idiom`/`InIdiom`/`Reported`/`Subject` are MISC
features; **`Shared` is already a FEATS one** -- so for it this script is a RENAME within field 6,
not a move between fields. Each key is looked for in MISC first and then in FEATS, and a key found
in FEATS is REMOVED from it: leaving `Shared` beside `SudShared` would make the reference carry the
same gold twice, and would score the frozen morphologiser against a feature the arm has taken over.
(No leakage is possible either way -- under `annotating_components` the PREDICTED doc's FEATS come
from the frozen morphologiser, not from this file.)

The rewrite is block-based and touches nothing but the FEATS cell of rows that have a hoistable
key, so the output is byte-identical to the input everywhere else -- verify before long runs, as
with the other CoNLL-U rewriters in this project.

    hoist_sud_gold.py IN.conllu OUT.conllu [--keys Subject Reported]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sud_misc import HOIST_PREFIX, SUD_KEYS  # noqa: E402


def parse_col(col):
    d = {}
    if col == "_":
        return d
    for item in col.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
    return d


def hoist_row(fields, keys):
    """Return the row with any wanted key copied into FEATS under the Sud prefix.

    A key is looked for in MISC first, then in FEATS. One found in FEATS is CONSUMED -- the arm
    takes the feature over, so it must not remain under its bare name as well (see the docstring).
    """
    misc = parse_col(fields[9])
    feats = parse_col(fields[5])
    found, consumed = {}, []
    for k in keys:
        hoisted = f"{HOIST_PREFIX}{k}"
        if k in misc:
            found[hoisted] = misc[k]
        elif k in feats:
            found[hoisted] = feats[k]
            consumed.append(k)
        elif hoisted in feats:
            # Already hoisted on an earlier pass, and its bare source was consumed then. Carrying
            # it forward is what keeps this script IDEMPOTENT for the FEATS-sourced keys: without
            # this branch a second run would strip `SudShared` as stale and find no `Shared` left
            # to re-derive it from, silently deleting the gold.
            found[hoisted] = feats[hoisted]
    if not found:
        return fields, 0
    # Drop any stale hoisted key so re-running is idempotent rather than duplicating.
    feats = {k: v for k, v in feats.items()
             if not k.startswith(HOIST_PREFIX) and k not in consumed}
    feats.update(found)
    fields = list(fields)
    fields[5] = "|".join(f"{k}={feats[k]}" for k in sorted(feats)) or "_"
    return fields, len(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--keys", nargs="+", default=list(SUD_KEYS),
                    help=f"SUD keys to hoist (default: {' '.join(SUD_KEYS)})")
    args = ap.parse_args()

    keys = list(args.keys)
    n_rows = n_hoisted = 0
    out = []
    for line in open(args.infile, encoding="utf-8"):
        stripped = line.rstrip("\n")
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        fields = stripped.split("\t")
        if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
            out.append(line)          # MWT range / empty node: FEATS is not ours to touch
            continue
        n_rows += 1
        fields, n = hoist_row(fields, keys)
        n_hoisted += n
        out.append("\t".join(fields) + "\n")

    pathlib.Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    print(f"{args.outfile}: {n_hoisted} values hoisted over {n_rows} tokens "
          f"({', '.join(keys)})")


if __name__ == "__main__":
    main()
