#!/usr/bin/env python
"""Revert the CSL-marked sandhi in a sandhied-CSL CoNLL-U file, producing the *CSL-reverted*
representation the parser trains on.

Applies `sa_tokenizer.desandhi_csl` (the single source of truth, shared with the runtime
tokeniser) to each sentence's ordered token forms: vowel coalescence and avagraha are undone,
while the unmarked consonant/visarga sandhi (visarga → -o/-r, m → ṃ, t/n assimilation) is kept
on the surface. Only the FORM column changes; lemmas, deprels, heads, feats and MISC are
untouched. MWT range forms are rebuilt by concatenating their reverted members, and `# text =`
is regenerated from the reverted forms + SpaceAfter so the file still round-trips.

Usage: .venv/bin/python scripts/revert_csl_sandhi.py IN.conllu OUT.conllu
"""
import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("sa_tokenizer", os.path.join(_HERE, "sa_tokenizer.py"))
sa_tokenizer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa_tokenizer)
desandhi_csl = sa_tokenizer.desandhi_csl


def _rebuild_text(tok_cols):
    parts = []
    for cols in tok_cols:
        parts.append(cols[1])
        misc = cols[9].split("|") if len(cols) > 9 else []
        if "SpaceAfter=No" not in misc:
            parts.append(" ")
    return "".join(parts).strip()


def process(in_path, out_path):
    n_sent = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as out:
        block = []
        def flush():
            nonlocal n_sent
            if not block:
                return
            n_sent += 1
            comments = [l for l in block if l.startswith("#")]
            rows = [l.rstrip("\n").split("\t") for l in block if not l.startswith("#") and l.strip()]
            toks = [c for c in rows if "-" not in c[0] and "." not in c[0]]
            forms = [c[1] for c in toks]
            rev = desandhi_csl(forms)
            id2form = {}
            for c, nf in zip(toks, rev):
                c[1] = nf
                id2form[int(c[0])] = nf
            for c in rows:                                   # rebuild MWT range surfaces
                if "-" in c[0]:
                    a, b = (int(x) for x in c[0].split("-"))
                    c[1] = "".join(id2form[i] for i in range(a, b + 1))
            text = _rebuild_text(toks)
            for l in comments:
                out.write(("# text = " + text + "\n") if l.startswith("# text") else l)
            for c in rows:
                out.write("\t".join(c) + "\n")
            out.write("\n")
        for line in fin:
            if line.strip() == "":
                flush()
                block = []
            else:
                block.append(line)
        flush()
    print(f"{in_path}: {n_sent} sentences -> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    process(a.inp, a.out)


if __name__ == "__main__":
    main()
