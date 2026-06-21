#!/usr/bin/env python
"""Coarsen the Indonesian treebank: merge each multiword-token range (host+enclitic,
e.g. penghuni+nya -> penghuninya) into a single whitespace token, re-projecting the
tree. This makes id tokens = whitespace+punctuation, which spaCy's rule tokeniser
reproduces deterministically (enclitic splitting is lexically ambiguous and not
rule-recoverable, so we go coarser instead). Reversible: merged form carries
SpaceAfter, text is unchanged.

Run: .venv/bin/python scripts/coarsen_id.py in.conllu out.conllu
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from retokenize import read_conllu, repair_tree

def coarsen(comments, toks, mwts):
    idx_of = {int(t.id): k for k, t in enumerate(toks)}
    groups, i = [], 0
    while i < len(toks):
        tid = int(toks[i].id)
        if tid in mwts:
            end, form, misc = mwts[tid]
            cov = []
            while i < len(toks) and int(toks[i].id) <= end:
                cov.append(i); i += 1
            groups.append((cov, form, misc))
        else:
            groups.append(([i], toks[i].form, toks[i].misc)); i += 1
    g_of = [None] * len(toks)
    for gi, (cov, _, _) in enumerate(groups):
        for k in cov:
            g_of[k] = gi

    heads, meta = [], []
    for gi, (cov, form, misc) in enumerate(groups):
        rep = next((k for k in cov if toks[k].head == "0"
                    or g_of[idx_of[int(toks[k].head)]] != gi), cov[0])
        rt = toks[rep]
        heads.append(0 if rt.head == "0" else g_of[idx_of[int(rt.head)]] + 1)
        meta.append((form, rt.upos, rt.xpos, rt.deprel, misc))

    repair_tree(heads)
    rows = []
    for gi, (form, up, xp, dep, misc) in enumerate(meta):
        h = heads[gi]
        if h == 0 and dep != "root":
            dep = "root"
        elif h != 0 and dep == "root":
            dep = "parataxis"
        sa = "SpaceAfter=No" if "SpaceAfter=No" in misc.split("|") else "_"
        rows.append([str(gi + 1), form, "_", up, xp, "_", str(h), dep, "_", sa])
    return comments, rows

def main():
    inp, out = sys.argv[1], sys.argv[2]
    n = merged = 0
    with open(out, "w", encoding="utf-8") as f:
        for comments, toks, mwts in read_conllu(inp):
            n += 1; merged += len(mwts)
            cm, rows = coarsen(comments, toks, mwts)
            for c in cm:
                f.write(c + "\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
            f.write("\n")
    print(f"{n} sentences, {merged} MWT ranges merged -> {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
