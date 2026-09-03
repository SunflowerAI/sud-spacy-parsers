#!/usr/bin/env python3
"""Rule laboratory over the dumps from lzh_rule_dump.py. Exploratory helpers only."""
import collections
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def profile(docs, top=30):
    conf = collections.Counter()
    n = tot = 0
    for d in docs:
        for t in d:
            tot += 1
            if t["gu"] != t["pu"]:
                n += 1
                conf[(t["gu"], t["pu"])] += 1
    print("UPOS errors %d / %d = %.2f%%" % (n, tot, 100.0 * n / tot))
    for (g, p), c in conf.most_common(top):
        print("  gold %-6s pred %-6s  %5d" % (g, p, c))
    return conf


def by_form(docs, gold=None, pred=None, top=40):
    c = collections.Counter()
    for d in docs:
        for t in d:
            if t["gu"] == t["pu"]:
                continue
            if gold and t["gu"] != gold:
                continue
            if pred and t["pu"] != pred:
                continue
            c[t["form"]] += 1
    for f, n in c.most_common(top):
        print("  %s  %d" % (f, n))
    return c


if __name__ == "__main__":
    docs = load(sys.argv[1])
    profile(docs)
