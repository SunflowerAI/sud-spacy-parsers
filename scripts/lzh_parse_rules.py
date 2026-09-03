#!/usr/bin/env python3
"""Deterministic PARSE rules for lzh: punctuation attachment, and label constraints.

Every rule reads only the predicted tree and the surface string, so all are runtime-legal.
Accounting is per token: FIRE = the rule changes the prediction, FIX = the new arc/label is
gold and the old was not, BREAK = the old was gold and the new is not.
"""
import collections
import os
import json
import sys

FINAL = set("。．.！？!?…；;")
MEDIAL = set("，、,:：")
OPEN = set("「『（(《〈【")
CLOSE = set("」』）)》〉】")


def pred_sents(doc):
    """predicted sentence id per index, from the predicted tree's self-headed roots."""
    root = [i for i, t in enumerate(doc) if t["ph"] == i]
    sid = [0] * len(doc)
    s = -1
    rs = set(root)
    for i in range(len(doc)):
        # a sentence starts at the first token of a maximal projection; approximate with
        # the predicted sent id already dumped
        sid[i] = doc[i]["ps"]
    return sid, rs


def unit_head(doc, lo, hi):
    """The token in [lo,hi) whose predicted head lies outside the span (leftmost such)."""
    for i in range(lo, hi):
        h = doc[i]["ph"]
        if h < lo or h >= hi or h == i:
            return i
    return hi - 1 if hi > lo else None


def rule_punct(doc, mode):
    """Return {i: new_head} for PUNCT tokens under one attachment convention."""
    out = {}
    sid = [t["ps"] for t in doc]
    root = {}
    for i, t in enumerate(doc):
        if t["ph"] == i:
            root[sid[i]] = i
    # unit boundaries: predicted-PUNCT positions
    marks = [i for i, t in enumerate(doc) if t["pu"] == "PUNCT"]
    prev_mark = -1
    for i, t in enumerate(doc):
        if t["pu"] != "PUNCT":
            continue
        lo, hi = prev_mark + 1, i
        prev_mark = i
        if mode == "sentroot":
            h = root.get(sid[i])
        elif mode == "unithead":
            h = unit_head(doc, lo, hi)
            if h is None:
                h = root.get(sid[i])
        elif mode == "mixed":
            if t["form"] and t["form"][0] in MEDIAL:
                h = unit_head(doc, lo, hi)
                if h is None:
                    h = root.get(sid[i])
            elif t["form"] and t["form"][0] in OPEN:
                continue
            else:
                h = root.get(sid[i])
        else:
            raise ValueError(mode)
        if h is not None and h != i:
            out[i] = h
    return out


def score_heads(docs, fn):
    fire = fix = brk = 0
    for d in docs:
        for i, h in fn(d).items():
            t = d[i]
            if h == t["ph"]:
                continue
            fire += 1
            was = t["ph"] == t["gh"]
            now = h == t["gh"]
            fix += now and not was
            brk += was and not now
    return fire, fix, brk


def main():
    dev = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "dev.json"))
    te = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "test.json"))
    for mode in ("sentroot", "unithead", "mixed"):
        print("punct attachment: %-9s" % mode, end="")
        for name, ds in (("dev", dev), ("test", te)):
            n = sum(len(d) for d in ds)
            f, fx, bk = score_heads(ds, lambda d, m=mode: rule_punct(d, m))
            print("  | %s fire %4d fix %4d brk %4d net %+5d (%+.3f UAS)"
                  % (name, f, fx, bk, fx - bk, 100.0 * (fx - bk) / n), end="")
        print()

    # label-only rule: PUNCT is always `punct`
    for name, ds in (("dev", dev), ("test", te)):
        n = sum(len(d) for d in ds)
        f = fx = bk = 0
        for d in ds:
            for t in d:
                if t["pu"] != "PUNCT" or t["pd"] == "punct":
                    continue
                f += 1
                fx += t["gd"] == "punct" and t["gh"] == t["ph"]
                bk += t["gd"] == t["pd"] and t["gh"] == t["ph"]
        print("label PUNCT->punct  %-4s fire %4d fix %4d brk %4d net %+4d (%+.3f LAS)"
              % (name, f, fx, bk, fx - bk, 100.0 * (fx - bk) / n))


if __name__ == "__main__":
    main()
