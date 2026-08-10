#!/usr/bin/env python
"""Normalise sa's XPOS column: field 5 is the UPOS, on every token, in all three sources.

Sanskrit is the third arm trained on more than one treebank (Vedic + UFAL + DCS), and unlike
Latin and English it needed no conversion: field 5 arrives `_` in all three and is filled from
UPOS downstream, so the convention was already uniform. It was uniform with ONE exception --
a single token (`dharm'` in sa_ufal, present in both the train and test files) carried
`Compound=Yes` in the XPOS column, a FEATS value that had shifted one field left. It was not
harmless: `Compound=Yes` reached the RELEASED sa tagger as a label.

The fix is one cell, and this script exists only so it is not a manual edit to gitignored data
that the next rebuild silently loses. FEATS is deliberately NOT touched -- sa reads `Compound`
as an INPUT feature, and the tokeniser and `sa_compound` supply it at runtime, which is how the
arm is designed. Idempotent; XPOS column only; verified line for line.

⚠ sa was NOT retrained for this. The junk label sits unused in the shipped model until the arm
is next rebuilt.

    normalise_sa_xpos.py <file.conllu> [more.conllu ...] [--dry-run]
"""
import argparse


def normalise(path, dry_run):
    raw = open(path, encoding="utf-8").read()
    out, n = [], 0
    for line in raw.split("\n"):
        c = line.split("\t")
        if len(c) == 10 and c[0].isdigit() and c[4] != "_" and c[4] != c[3] and "=" in c[4]:
            c[4] = c[3]
            line = "\t".join(c)
            n += 1
        out.append(line)
    new = "\n".join(out)
    if len(new.split("\n")) != len(raw.split("\n")):
        raise SystemExit(f"{path}: line count changed")
    if not dry_run:
        open(path, "w", encoding="utf-8").write(new)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for path in a.files:
        print(f"{path}: {normalise(path, a.dry_run)} XPOS cell(s) normalised to the UPOS")


if __name__ == "__main__":
    main()
