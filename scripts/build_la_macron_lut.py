#!/usr/bin/env python3
"""Harvest the macronisation lookup table used by the ``la_macronise`` component.

Input is a plain CoNLL-U file and its macronised twin (``macronise_la.py`` output), which are
token-aligned by construction -- that script only ever rewrites FORM. For every token we record
the vowel-length PATTERN as a bitmask over character positions, keyed at three levels of
specificity:

    L1  (form, upos, feats)   -- what the morphologiser lets us disambiguate
    L2  (form, upos)
    L3  (form)                -- what a bare word list can do

NO SUFFIX LEVELS. This used to emit two more, indexed from the RIGHT, "which generalise to forms
never seen in training" -- S4 ``(form[-4:], upos, feats)`` and S3 ``(form[-3:], upos, feats)``. They
have been removed because they are not worth their size: measured on the held-out ITTB+PROIEL test
split, they agree with Alatius on **52.46 %** of the tokens they answer, against **90.42 %** for the
Morpheus table `la_macronise` now falls through to instead. An ending generalises; a stem does not,
and a stem is what those tokens were missing. See `la_macronise.fetch_morpheus`.

Patterns are case-insensitive (the key lowercases the form and the mask is positional), so a
sentence-initial capital is not mistaken for a distinct macronisation.

**Backoff pruning.** A more specific level is stored ONLY where it disagrees with the level below
it, which is what makes the table small enough to ship: 152 443 raw entries collapse to 42 817
(0.87 MB JSON, 0.23 MB gzipped). It also makes the morphology's actual contribution legible --
only ~1 150 (form, upos, feats) keys override the bare-form default, and those are the genuine
homographs (fōrma Nom vs fōrmā Abl, hōc Acc vs hoc Nom).

CAVEAT, and it is the important one: this table records what the **Alatius macroniser** produced,
not gold vowel length. Alatius is ~98-99% accurate on vowels, and its own RFTagger sometimes
disagrees with the treebank's gold morphology -- which is why some keys are "ambiguous" at all
(``forma`` tagged Case=Nom gets ``fōrmā`` 68 times, which is simply wrong). Accuracy figures
measured against it are AGREEMENT WITH ALATIUS, never ground truth.

    build_la_macron_lut.py assets_la/la_ittbproiel-sud-train.conllu \
                           assets_la/la_ittbproiel-sud-train.macron.conllu \
                           scripts/la_macron_lut.json.gz
"""
import argparse
import gzip
import json
import unicodedata
from collections import Counter, defaultdict


def rows(path):
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" in line and line.split("\t", 1)[0].isdigit():
            yield line.split("\t")


def strip_macron(s):
    n = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in n if c != "̄"))


def load(plain_path, macron_path):
    out = []
    for p, m in zip(rows(plain_path), rows(macron_path)):
        form, macd = p[1], m[1]
        if strip_macron(macd) != form:
            continue  # the macroniser changed more than vowel length -- skip the token
        if not any(c.isalpha() for c in form):
            continue
        mask = 0
        for i, c in enumerate(macd):
            if strip_macron(c) != c:
                mask |= 1 << i
        out.append((form.lower(), mask, p[3], p[5]))
    return out


def majority(table):
    return {k: v.most_common(1)[0][0] for k, v in table.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plain")
    ap.add_argument("macron")
    ap.add_argument("out")
    args = ap.parse_args()

    data = load(args.plain, args.macron)
    l1, l2, l3 = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for form, mask, upos, feats in data:
        l1[(form, upos, feats)][mask] += 1
        l2[(form, upos)][mask] += 1
        l3[form][mask] += 1

    b1, b2, b3 = majority(l1), majority(l2), majority(l3)

    # backoff pruning: store a level only where it differs from the next-general one
    p2 = {k: v for k, v in b2.items() if b3.get(k[0]) != v}
    p1 = {k: v for k, v in b1.items()
          if p2.get((k[0], k[1]), b3.get(k[0])) != v}

    blob = {
        "L1": [[f, u, x, m] for (f, u, x), m in p1.items()],
        "L2": [[f, u, m] for (f, u), m in p2.items()],
        "L3": [[f, m] for f, m in b3.items()],
    }   # no S4/S3 -- see the module docstring. `la_macronise` still READS them, for tables built
        # before this change, but only when no Morpheus table is available to beat them.
    raw = json.dumps(blob, ensure_ascii=False).encode("utf-8")
    with gzip.open(args.out, "wb", compresslevel=9) as fh:
        fh.write(raw)

    print(f"{args.out}")
    print(f"  tokens read      {len(data)}")
    print(f"  L1 {len(b1):6d} -> {len(p1):6d} after pruning "
          f"(morphology overrides the bare form on {len(p1)} keys)")
    print(f"  L2 {len(b2):6d} -> {len(p2):6d}")
    print(f"  L3 {len(b3):6d}")
    print("  (no suffix levels: la_macronise falls through to Morpheus instead)")
    print(f"  json {len(raw)/1e6:.2f} MB, gzipped {len(gzip.compress(raw, 9))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
