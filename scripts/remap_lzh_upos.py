#!/usr/bin/env python3
"""Split 之's UPOS by what it links: nominal genitive -> PART, clausal nominaliser -> SCONJ.

WHY ONLY 之. A survey of every UPOS/XPOS cell over 1 500 tokens looked for a split DEPREL
distribution — the signature of one label covering two functions. Four cells qualified, and three
are not conflations at all: a VERB is `root` 42 % and `mod` 17 % because a verb can head a main or
a modifier clause; a NOUN is `subj` 40 % and `comp:obj` 35 % because that is what nouns do.
Rewriting UPOS by deprel there would encode syntax into the part of speech, which is precisely what
UPOS is meant not to carry.

The one real case is `SCONJ p,助詞,接続,属格` — 之, 8 097 tokens, `mod` 68 % with a NOUN head against
`subj` 28 % with a VERB head. One lexeme, two jobs:

    君之臣      genitive linker between two nominals            -> PART
    吾之不遇    nominaliser turning a clause into an argument   -> SCONJ

THE RULE is what 之 LINKS, read off its head's UPOS — not its deprel, which reports the role of the
whole constituent rather than its internal category. Head in {VERB, AUX} -> clausal -> SCONJ;
otherwise -> PART. That separates 97.6 % of tokens cleanly (the PART side is 92 % `mod`, the SCONJ
side 90 % `subj`); the 2.4 % residue is mostly 之 under a PART head such as 也/者.

⚠ **XPOS IS NOT TOUCHED.** Both keep `p,助詞,接続,属格` — Kyoto's own tagset does not make this
distinction (SCONJ, CCONJ and PART all come from `p,助詞`; only fields 3-4 separate 属格 "genitive"
from 並列 "coordinative"). So after this remap one XPOS maps to two UPOS, which is already true of
26 of the 121 codes, and `build_lzh_xpos_tables.py` must be re-run so the UPOS mask admits both.

⚠ **THIS REWRITES THE GOLD**, so UPOS accuracy after it is not comparable to the 93.13 before it —
the moving-denominator trap that `docs/udep-relabel.md` records for every relabelling in this repo.
The parser is unaffected: it reads no UPOS and its heads and deprels do not change.

Rationale for PART rather than ADP: zh GSD tags the direct counterpart 的 as PART for the nominal
use and SCONJ for the relative/clausal one, so this keeps lzh consistent with its sibling treebank.

Usage:
    remap_lzh_upos.py --in <conllu> --out <conllu>
"""
import argparse
import collections
import pathlib

CLAUSAL_HEAD = {"VERB", "AUX"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--form", default="之")
    ap.add_argument("--from-upos", default="SCONJ")
    ap.add_argument("--to-upos", default="PART")
    a = ap.parse_args()

    out, block, changed, kept = [], [], 0, 0
    dist = collections.Counter()

    def flush():
        nonlocal changed, kept
        rows = [l for l in block if l and not l.startswith("#") and "\t" in l]
        cols = [r.split("\t") for r in rows]
        idx = {c[0]: i for i, c in enumerate(cols)}
        for c in cols:
            if c[1] != a.form or c[3] != a.from_upos:
                continue
            h = c[6]
            hp = cols[idx[h]][3] if h != "0" and h in idx else "ROOT"
            if hp in CLAUSAL_HEAD:
                kept += 1
                dist[(a.from_upos, hp)] += 1
            else:
                c[3] = a.to_upos
                changed += 1
                dist[(a.to_upos, hp)] += 1
        it = iter("\t".join(c) for c in cols)
        for l in block:
            out.append(next(it) if (l and not l.startswith("#") and "\t" in l) else l)

    for line in pathlib.Path(a.inp).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            flush()
            out.append("")
            block = []
            continue
        block.append(line)
    if block:
        flush()
    pathlib.Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{a.out}: {a.form} {a.from_upos}->{a.to_upos} on {changed}, kept {a.from_upos} on {kept}")
    print("   by head UPOS:", ' '.join(f'{u}/{h}:{n}' for (u, h), n in dist.most_common(6)))


if __name__ == "__main__":
    main()
