#!/usr/bin/env python3
"""Harvest `ar_vocalise`'s table from SUD_Arabic-PADT's own ``Vform`` annotation.

Every PADT token carries the fully vocalised word form in MISC beside the gold UPOS and FEATS, so
the pairing this needs is already in the treebank -- no external tool has to generate it, which is
the one respect in which Arabic is easier here than Latin was. Three rungs, most specific first:

    L1  (skeleton, upos, feats)   the morphologiser settling the case ending
    L2  (skeleton, upos)
    L3  (skeleton)                a bare word list, the last resort

Each key keeps its MAJORITY Vform. The table is then BACKOFF-PRUNED: an L1 entry whose answer L2
would have given anyway is dropped, and likewise L2 against L3. That is not only a size win -- it
makes the stored table say what it means, namely "this key answers DIFFERENTLY from the more
general one", so reading it tells you where morphology is actually doing work.

⚠ LICENCE, and it is not a formality. SUD_Arabic-PADT is CC BY-NC-SA 3.0. This table is a direct
extract of its annotation, so it carries that licence; and by the argument this project already
made for en_gum -- the annotations are what a trained model absorbs -- so does any model trained
on PADT. The released `ar_sud_padt` wheel declares **CC BY-SA 4.0**, which appears to be wrong
already, before this component exists. Resolve that before packaging this table into a wheel.

Usage:
    python scripts/build_ar_vocalise_lut.py            # -> scripts/ar_vocalise_lut.json.gz
    python scripts/build_ar_vocalise_lut.py --stats    # report only, write nothing
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_vocalise import canon, strip_diac   # noqa: E402

DEFAULT_TRAIN = "assets_ar/SUD_Arabic-PADT/ar_padt-sud-train.conllu"
DEFAULT_OUT = Path(__file__).resolve().parent / "ar_vocalise_lut.json.gz"
PASS_THROUGH = {"X", "PUNCT", "SYM"}


def read_conllu(path):
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        # range lines (n-m) hold the undivided orthographic word and no annotation of their own;
        # empty nodes (n.m) are not surface tokens. Neither carries a Vform to harvest.
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            continue
        misc = dict(kv.split("=", 1) for kv in f[9].split("|") if "=" in kv)
        v = misc.get("Vform")
        if v:
            yield f[1], f[3], f[5], v, f[2]


def build(paths):
    c1 = collections.defaultdict(collections.Counter)
    c2 = collections.defaultdict(collections.Counter)
    c3 = collections.defaultdict(collections.Counter)
    n = 0
    lex = collections.Counter()
    for p in paths:
        for form, upos, feats, v, lemma in read_conllu(p):
            # PADT's LEMMA is a vocalised lemma in the same convention as calima's `lex`, so this
            # counter ranks the analyser's rival readings by how common the LEXEME actually is.
            if lemma and lemma != "_":
                lex[canon(lemma)] += 1
            n += 1
            k = strip_diac(form)
            c1[(k, upos, feats)][v] += 1
            c2[(k, upos)][v] += 1
            # `X`/PUNCT/SYM are harvested into the UPOS-keyed rungs but deliberately kept OUT of
            # the skeleton-only one, which the component never consults for them: their fallback is
            # the bare form, and a majority taken across the other parts of speech would be a
            # confident wrong answer for a foreign word that merely shares a skeleton.
            if upos not in PASS_THROUGH:
                c3[k][v] += 1
    maj = lambda c: c.most_common(1)[0][0]
    l3 = {k: maj(c) for k, c in c3.items()}
    l2 = {k: maj(c) for k, c in c2.items()}
    l1 = {k: maj(c) for k, c in c1.items()}
    before = (len(l1), len(l2), len(l3))

    # Prune against whatever the component would have answered had the entry been absent -- which
    # is NOT simply the next rung down, because the pass-through classes skip L3 and fall back to
    # the bare form. Pruning them against L3 would delete entries that are then never recovered.
    # Compare with `canon`: two spellings differing only by a free convention are one answer.
    def fallback(f, u):
        if u in PASS_THROUGH:
            return f                      # the component returns the token unchanged
        return l3.get(f)                  # None = nothing below, so the entry must be kept

    l2p = {}
    for (f, u), m in l2.items():
        fb = fallback(f, u)
        if fb is None or canon(fb) != canon(m):
            l2p[(f, u)] = m
    l1p = {}
    for (f, u, x), m in l1.items():
        fb = l2p[(f, u)] if (f, u) in l2p else fallback(f, u)
        if fb is None or canon(fb) != canon(m):
            l1p[(f, u, x)] = m
    return l1p, l2p, l3, before, n, lex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[DEFAULT_TRAIN])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stats", action="store_true", help="report and write nothing")
    a = ap.parse_args()
    l1, l2, l3, before, n, lex = build(a.train)
    print(f"harvested {n} tokens carrying Vform")
    print(f"  L1 (skeleton, upos, feats) {before[0]:7d} -> {len(l1):7d} after pruning")
    print(f"  L2 (skeleton, upos)        {before[1]:7d} -> {len(l2):7d}")
    print(f"  L3 (skeleton)              {before[2]:7d} -> {len(l3):7d}")
    if a.stats:
        return
    blob = {"L1": [[f, u, x, m] for (f, u, x), m in l1.items()],
            "L2": [[f, u, m] for (f, u), m in l2.items()],
            "L3": [[f, m] for f, m in l3.items()],
            # LEX ranks the analyser's rival readings of one skeleton. Without it `للمدرسة` came
            # back as لِلمُدَرِّسَة "for the teacher" -- a legitimate vocalisation of that skeleton,
            # but calima lists it first and the component took the first surviving reading.
            "LEX": [[k, c] for k, c in lex.most_common() if c > 1]}
    with gzip.open(a.out, "wb", compresslevel=9) as fh:
        fh.write(json.dumps(blob, ensure_ascii=False).encode("utf-8"))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
