#!/usr/bin/env python3
"""Hoist SUD's MISC keys into the FEATS column so they survive `spacy convert`.

`spacy convert --converter conllu` reads MISC for exactly two things -- `SpaceAfter=No` and the NER
pattern -- and discards the rest (`spacy/training/converters/conllu_to_docs.py`). FEATS, by
contrast, goes straight into `token.morph`. So the only way to get `Subject=`/`Reported=` gold into
a `.spacy` corpus without writing a replacement converter is to move it to field 6 first.

THIS IS A TRAINING-TIME TRANSPORT ONLY. At inference the components write to MISC
(`Token._.sud_misc`), never to FEATS -- these are MISC features in every treebank here, and the
released models emit them as such. To make that impossible to misread, each key is renamed with a
`Sud` prefix on the way in: `Subject=SubjRaising` becomes `SudSubject=SubjRaising`. A prefixed key
can never be mistaken for a genuine morphological feature, and `sud_tagger` reads only prefixed
keys, so it cannot accidentally train on real FEATS.

The rewrite is block-based and touches nothing but the FEATS cell of rows that have a hoistable
key, so the output is byte-identical to the input everywhere else -- verify before long runs, as
with the other CoNLL-U rewriters in this project.

    hoist_sud_gold.py IN.conllu OUT.conllu [--keys Subject Reported]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sud_misc import HOIST_PREFIX, SUD_MISC_KEYS  # noqa: E402


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
    """Return the row with any wanted MISC key copied into FEATS under the Sud prefix."""
    misc = parse_col(fields[9])
    found = {f"{HOIST_PREFIX}{k}": misc[k] for k in keys if k in misc}
    if not found:
        return fields, 0
    feats = parse_col(fields[5])
    # Drop any stale hoisted key so re-running is idempotent rather than duplicating.
    feats = {k: v for k, v in feats.items() if not k.startswith(HOIST_PREFIX)}
    feats.update(found)
    fields = list(fields)
    fields[5] = "|".join(f"{k}={feats[k]}" for k in sorted(feats)) or "_"
    return fields, len(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--keys", nargs="+", default=list(SUD_MISC_KEYS),
                    help=f"MISC keys to hoist (default: {' '.join(SUD_MISC_KEYS)})")
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
