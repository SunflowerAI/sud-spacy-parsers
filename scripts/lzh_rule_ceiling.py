#!/usr/bin/env python3
"""The ceiling on ANY deterministic UPOS-override rule set over a given signature family.

For each family and each cell (signature value), an override rule may either stay silent or
force one label L. The best possible choice is made ON THE EVALUATION SET ITSELF -- so the
number is a cheating upper bound over every subset of cells and every label assignment.
If the oracle is small, no rule over that signature can be worth finding, however it is found.

Cells with fewer than k eval occurrences are forced silent, since a rule fitted to a cell seen
once is memorising a token rather than expressing a generalisation.
"""
import collections
import os
import json
import sys

sys.path.insert(0, "scripts")
from lzh_upos_rule_mine import sigs, PRIORITY  # noqa: E402

LABELS = ["VERB", "NOUN", "PROPN", "ADV", "AUX", "ADP", "PART", "PRON", "SCONJ",
          "NUM", "CCONJ", "INTJ", "PUNCT"]


def main():
    ev_name = sys.argv[1] if len(sys.argv) > 1 else "test"
    ev = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "%s.json" % ev_name))
    n = sum(len(d) for d in ev)
    base = sum(1 for d in ev for t in d if t["gu"] == t["pu"])
    ks = (1, 3, 5, 10, 20, 50)
    print("%s: %d tokens, model UPOS %.2f%% (%d errors)"
          % (ev_name, n, 100.0 * base / n, n - base))
    print("%-20s %s" % ("family", "  ".join("k>=%-4d" % k for k in ks)))
    for fam in PRIORITY:
        cell = collections.defaultdict(list)
        for d in ev:
            for i in range(len(d)):
                for f, v in sigs(d, i, False):
                    if f == fam:
                        cell[v].append(d[i])
                        break
        row = []
        for k in ks:
            net = 0
            for v, toks in cell.items():
                if len(toks) < k:
                    continue
                best = 0
                for L in LABELS:
                    g = sum((t["gu"] == L) - (t["gu"] == t["pu"])
                            for t in toks if t["pu"] != L)
                    best = max(best, g)
                net += best
            row.append("%+5d/%+.2f" % (net, 100.0 * net / n))
        print("%-20s %s" % (fam, "  ".join(row)))


if __name__ == "__main__":
    main()
