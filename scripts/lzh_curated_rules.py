#!/usr/bin/env python3
"""A small curated set of deterministic UPOS rules for lzh, each stated as a predicate.

Every predicate reads only the form, its neighbours, and the PARSER'S OWN predicted tree --
including, crucially, the deprels of the token's OWN DEPENDENTS, which the morphologiser's
`DEP` channel does not carry (it embeds the token's own deprel and nothing else).

Support and dominance come from TRAIN GOLD; dev and test are both honest reports.
"""
import collections
import json
import os


def children_deps(doc, i, gold):
    hk, dk = ("gh", "gd") if gold else ("ph", "pd")
    return {doc[j][dk] for j in range(len(doc)) if j != i and doc[j][hk] == i}


def head_upos(doc, i, gold):
    h = doc[i]["gh" if gold else "ph"]
    return "@SELF" if h == i else doc[h]["gu" if gold else "pu"]


# name -> (predicate(doc, i, gold) -> bool, target UPOS)
RULES = [
    ("以 has a comp:obj child -> VERB",
     lambda d, i, g: d[i]["form"] == "以" and "comp:obj" in children_deps(d, i, g), "VERB"),
    ("以 has NO comp:obj child -> ADV",
     lambda d, i, g: d[i]["form"] == "以" and "comp:obj" not in children_deps(d, i, g), "ADV"),
    ("為 has a comp:pred child -> AUX",
     lambda d, i, g: d[i]["form"] == "為" and "comp:pred" in children_deps(d, i, g), "AUX"),
    ("爲 has a comp:pred child -> AUX",
     lambda d, i, g: d[i]["form"] == "爲" and "comp:pred" in children_deps(d, i, g), "AUX"),
    ("為/爲 has a comp:obj child, no comp:pred -> VERB",
     lambda d, i, g: d[i]["form"] in "為爲" and "comp:obj" in children_deps(d, i, g)
     and "comp:pred" not in children_deps(d, i, g), "VERB"),
    ("之 head is NOUN/PROPN/PRON -> PART",
     lambda d, i, g: d[i]["form"] == "之" and head_upos(d, i, g) in
     ("NOUN", "PROPN", "PRON"), "PART"),
    ("之 head is VERB/AUX -> SCONJ",
     lambda d, i, g: d[i]["form"] == "之" and head_upos(d, i, g) in ("VERB", "AUX"), "SCONJ"),
    ("之 has a comp:obj child and a NOUN head -> PART",
     lambda d, i, g: d[i]["form"] == "之" and "comp:obj" in children_deps(d, i, g)
     and head_upos(d, i, g) == "NOUN", "PART"),
    ("無 has a comp:obj child -> VERB",
     lambda d, i, g: d[i]["form"] == "無" and "comp:obj" in children_deps(d, i, g), "VERB"),
    ("與 has a comp:obj child, dep is mod -> ADP",
     lambda d, i, g: d[i]["form"] == "與" and "comp:obj" in children_deps(d, i, g)
     and d[i]["gd" if g else "pd"] == "mod", "ADP"),
]


def main():
    pre = os.environ.get("LZH_DUMP", "/tmp/lzh_")
    tr = json.load(open(pre + "train.json"))
    dev = json.load(open(pre + "dev.json"))
    te = json.load(open(pre + "test.json"))

    print("%-52s %6s %6s | %-21s | %-21s"
          % ("rule", "trN", "trDom", "dev fire/fix/brk/net", "test fire/fix/brk/net"))
    for name, pred, L in RULES:
        c = collections.Counter()
        for d in tr:
            for i in range(len(d)):
                if pred(d, i, True):
                    c[d[i]["gu"]] += 1
        n = sum(c.values())
        dom = c[L] / n if n else 0.0
        cells = []
        for ds in (dev, te):
            f = fx = bk = 0
            for d in ds:
                for i in range(len(d)):
                    if d[i]["pu"] == L or not pred(d, i, False):
                        continue
                    f += 1
                    fx += d[i]["gu"] == L
                    bk += d[i]["gu"] == d[i]["pu"]
            cells.append((f, fx, bk))
        print("%-52s %6d %6.3f | %4d %4d %4d %+5d | %4d %4d %4d %+5d"
              % (name, n, dom,
                 cells[0][0], cells[0][1], cells[0][2], cells[0][1] - cells[0][2],
                 cells[1][0], cells[1][1], cells[1][2], cells[1][1] - cells[1][2]))


if __name__ == "__main__":
    main()
