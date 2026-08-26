#!/usr/bin/env python3
"""Extract a plain-text corpus from DCS for training Sanskrit vectors.

DCS is IAST throughout, with FORM = the unsandhied word and LEMMA = the dictionary stem, which is
the same representation our sa arm works in (`docs/sanskrit.md`). Two regimes are emitted and the
choice is RECORDED rather than inferred downstream (CLAUDE.md hazard 10):

  --key lemma   one line per sentence of LEMMAS.  Cross-lingual alignment is a meaning-level
                operation and Apte -- the only Sanskrit-English anchor source we have -- is keyed by
                stems, so the anchors and the vector table agree only in this regime.
  --key form    one line per sentence of unsandhied FORMS, for a form-keyed table.
"""
import argparse, pathlib, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcs", default="assets_dcs/dcs/data/conllu/files")
    ap.add_argument("--key", choices=["lemma", "form"], default="lemma")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    col = 2 if a.key == "lemma" else 1
    nsent = ntok = nfile = 0
    types = set()
    with open(a.out, "w", encoding="utf-8") as out:
        for p in sorted(pathlib.Path(a.dcs).rglob("*.conllu")):
            nfile += 1
            sent = []
            for line in p.open(encoding="utf-8", errors="replace"):
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    if sent:
                        out.write(" ".join(sent) + "\n"); nsent += 1; ntok += len(sent); sent = []
                    continue
                fs = line.split("\t")
                if len(fs) < 3 or not fs[0].isdigit():
                    continue          # skips the n-m multiword ranges, whose columns are all "_"
                w = fs[col]
                if w and w != "_":
                    sent.append(w); types.add(w)
            if sent:
                out.write(" ".join(sent) + "\n"); nsent += 1; ntok += len(sent)
            if nfile % 2000 == 0:
                print(f"  {nfile} files, {nsent} sentences, {ntok} tokens", file=sys.stderr, flush=True)
    print(f"{a.out}: {nfile} files, {nsent} sentences, {ntok} tokens, {len(types)} types")

main()
