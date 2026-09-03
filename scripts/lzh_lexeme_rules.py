#!/usr/bin/env python3
"""Per-lexeme deterministic UPOS rules for lzh, derived from TRAIN GOLD only.

Targets the high-error lexemes the profile names. A rule is
    (form, <one context slot>) -> UPOS
harvested where train gold dominance >= --dom on >= --sup examples, with the context slot
computed at runtime from the form's neighbours or the PARSER'S OWN predicted deprel/head.
Selection never touches dev or test, so BOTH are honest reports.
"""
import argparse
import collections
import os
import json

SLOTS = ["prev", "next", "dep", "dep+headform", "prev+dep", "next+dep", "prev+next",
         "headupos", "dep+headupos", "childdeps", "dep+childdeps", "childdeps+headupos"]


def slot(doc, i, name, gold):
    d = doc[i]
    prev = doc[i - 1]["form"] if i > 0 else "^"
    nxt = doc[i + 1]["form"] if i + 1 < len(doc) else "$"
    dep = d["gd"] if gold else d["pd"]
    h = d["gh"] if gold else d["ph"]
    hf = doc[h]["form"] if h != i else "@SELF"
    hu = (doc[h]["gu"] if gold else doc[h]["pu"]) if h != i else "@SELF"
    hk, dk = ("gh", "gd") if gold else ("ph", "pd")
    kids = tuple(sorted({doc[j][dk] for j in range(len(doc))
                         if j != i and doc[j][hk] == i}))
    return {"prev": (prev,), "next": (nxt,), "dep": (dep,),
            "dep+headform": (dep, hf), "prev+dep": (prev, dep),
            "next+dep": (nxt, dep), "prev+next": (prev, nxt),
            "headupos": (hu,), "dep+headupos": (dep, hu),
            "childdeps": kids, "dep+childdeps": (dep,) + kids,
            "childdeps+headupos": (hu,) + kids}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", type=float, default=0.90)
    ap.add_argument("--sup", type=int, default=30)
    ap.add_argument("--ntarget", type=int, default=30)
    a = ap.parse_args()

    tr = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "train.json"))
    dev = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "dev.json"))
    te = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "test.json"))

    # the lexemes the model actually gets wrong, ranked on DEV (never on test)
    errs = collections.Counter()
    for d in dev:
        for t in d:
            if t["gu"] != t["pu"]:
                errs[t["form"]] += 1
    targets = {f for f, _ in errs.most_common(a.ntarget)}

    # harvest, per (form, slot, value), from TRAIN GOLD
    stats = collections.defaultdict(collections.Counter)
    for d in tr:
        for i in range(len(d)):
            f = d[i]["form"]
            if f not in targets:
                continue
            for s in SLOTS:
                stats[(f, s, slot(d, i, s, True))][d[i]["gu"]] += 1
    rules = {}
    for key, c in stats.items():
        n = sum(c.values())
        if n < a.sup:
            continue
        L, k = c.most_common(1)[0]
        if k / n < a.dom:
            continue
        rules[key] = (L, n, k / n)
    print("rules harvested from train gold (dom>=%.2f, n>=%d, top-%d dev-error lexemes): %d"
          % (a.dom, a.sup, a.ntarget, len(rules)))

    def run(docs):
        per = collections.defaultdict(lambda: [0, 0, 0])
        for d in docs:
            for i in range(len(d)):
                f = d[i]["form"]
                if f not in targets:
                    continue
                for s in SLOTS:
                    key = (f, s, slot(d, i, s, False))
                    if key not in rules:
                        continue
                    L = rules[key][0]
                    if d[i]["pu"] == L:
                        continue
                    per[key][0] += 1
                    if d[i]["gu"] == L:
                        per[key][1] += 1
                    elif d[i]["gu"] == d[i]["pu"]:
                        per[key][2] += 1
        return per

    pdev, pte = run(dev), run(te)
    rows = []
    for key in set(pdev) | set(pte):
        dv = pdev.get(key, [0, 0, 0])
        tv = pte.get(key, [0, 0, 0])
        rows.append((key, rules[key], dv, tv))
    rows.sort(key=lambda r: -(r[2][1] - r[2][2]))
    print("\n%-3s %-13s %-22s %-6s %5s %5s | %-16s | %-16s"
          % ("", "slot", "value", "->", "trN", "trDom", "dev fire/fix/brk/net",
             "test fire/fix/brk/net"))
    dnet = tnet = dfire = tfire = dfix = tfix = dbrk = tbrk = 0
    for (f, s, v), (L, n, dom), dv, tv in rows:
        if dv[0] == 0 and tv[0] == 0:
            continue
        print("%-3s %-13s %-22s %-6s %5d %5.2f | %4d %4d %4d %+5d | %4d %4d %4d %+5d"
              % (f, s, "/".join(map(str, v))[:22], L, n, dom,
                 dv[0], dv[1], dv[2], dv[1] - dv[2],
                 tv[0], tv[1], tv[2], tv[1] - tv[2]))
        dfire += dv[0]; dfix += dv[1]; dbrk += dv[2]; dnet += dv[1] - dv[2]
        tfire += tv[0]; tfix += tv[1]; tbrk += tv[2]; tnet += tv[1] - tv[2]
    print("\nSUM (rules overlap, so this over-counts fires): dev %d/%d/%d net %+d | "
          "test %d/%d/%d net %+d" % (dfire, dfix, dbrk, dnet, tfire, tfix, tbrk, tnet))

    # cascade: one override per token, most specific slot first
    prio = ["prev+dep", "next+dep", "dep+headform", "dep+childdeps", "childdeps+headupos",
            "prev+next", "childdeps", "dep+headupos", "prev", "next", "dep", "headupos"]
    for name, docs in (("dev", dev), ("test", te)):
        fire = fix = brk = 0
        for d in docs:
            for i in range(len(d)):
                f = d[i]["form"]
                if f not in targets:
                    continue
                for s in prio:
                    key = (f, s, slot(d, i, s, False))
                    if key in rules and rules[key][0] != d[i]["pu"]:
                        L = rules[key][0]
                        fire += 1
                        if d[i]["gu"] == L:
                            fix += 1
                        elif d[i]["gu"] == d[i]["pu"]:
                            brk += 1
                        break
        n = sum(len(x) for x in docs)
        print("CASCADE %-5s fire %4d fix %4d break %4d NET %+4d  (%+.3f UPOS pts)"
              % (name, fire, fix, brk, fix - brk, 100.0 * (fix - brk) / n))


if __name__ == "__main__":
    main()
