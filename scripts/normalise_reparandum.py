#!/usr/bin/env python3
"""Normalise the UD-carry-over deprel ``reparandum`` to SUD's ``conj:dicto`` in CoNLL-U.

SUD annotates disfluency/repair as a subtype of ``conj`` (``conj:dicto`` — the sibling of
``conj:coord``/``conj:appos``), not with UD's bare ``reparandum`` relation. A few upstream SUD
releases (Latin-ITTB, Cantonese-HK, Chinese-GSD/GSDSimp) still carry the un-converted UD
``reparandum``; this rewrites it. It is a **pure label rename** — the head/attachment is
unchanged, so the dependency tree is identical.

ONLY the DEPREL column (field 8) is touched: ``reparandum`` is also a genuine Latin word form
(the gerundive of *reparō*), so FORM/LEMMA occurrences must be left alone. Rewrites in place.

    normalise_reparandum.py file1.conllu [file2.conllu ...]
"""
import sys


def convert_line(line):
    if not line or line[0] == "#" or line == "\n":
        return line, 0
    cols = line.rstrip("\n").split("\t")
    if len(cols) != 10:
        return line, 0
    dep = cols[7]
    if dep == "reparandum":
        cols[7] = "conj:dicto"
    elif dep.startswith("reparandum:") or dep.startswith("reparandum@"):  # subtyped (not seen, but safe)
        cols[7] = "conj:dicto" + dep[len("reparandum"):]
    else:
        return line, 0
    return "\t".join(cols) + "\n", 1


def main():
    total = 0
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        out, n = [], 0
        for l in lines:
            nl, c = convert_line(l)
            out.append(nl); n += c
        if n:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(out)
        print(f"  {path}: {n} reparandum -> conj:dicto")
        total += n
    print(f"total: {total}")


if __name__ == "__main__":
    main()
