#!/usr/bin/env python3
"""Mine deterministic UPOS-correction rules for lzh, and score them fire/fix/break/net.

A rule is (signature family, signature value) -> UPOS label L. At runtime it reads only the
token's form, its neighbours' forms, and the PARSER'S OWN predicted deprel/head (the parser
runs before the morphologiser in this arm), so every signature is computable at inference.

Selection uses TRAIN GOLD for the linguistic generalisation (dominance/support over gold
structure) and DEV PREDICTIONS for whether the model is actually wrong there. TEST is only
ever scored, never selected on.
"""
import argparse
import collections
import os
import json

PUNCT_LIKE = None


def sigs(doc, i, use_gold):
    """Every candidate signature for token i, as (family, value) pairs."""
    t = doc[i]
    f = t["form"]
    dep = t["gd"] if use_gold else t["pd"]
    h = t["gh"] if use_gold else t["ph"]
    prev = doc[i - 1]["form"] if i > 0 else "^"
    nxt = doc[i + 1]["form"] if i + 1 < len(doc) else "$"
    hf = doc[h]["form"] if h != i else "@SELF"
    hu = (doc[h]["gu"] if use_gold else doc[h]["pu"]) if h != i else "@SELF"
    hk, dk = ("gh", "gd") if use_gold else ("ph", "pd")
    kids = tuple(sorted({doc[j][dk] for j in range(len(doc))
                         if j != i and doc[j][hk] == i}))
    hdep = doc[h]["gd"] if use_gold else doc[h]["pd"]
    if h == i:
        hdep = "@SELF"
    direction = "self" if h == i else ("L" if h < i else "R")
    out = [
        ("form", (f,)),
        ("form+dep", (f, dep)),
        ("form+dep+headform", (f, dep, hf)),
        ("form+prev", (f, prev)),
        ("form+next", (f, nxt)),
        ("prev+form+next", (prev, f, nxt)),
        ("form+headform", (f, hf)),
        ("form+dep+dir", (f, dep, direction)),
        ("form+dep+headdep", (f, dep, hdep)),
        ("form+prev+dep", (f, prev, dep)),
        ("form+next+dep", (f, nxt, dep)),
        # form-free (class-level) families: the traditional diagnostics live here
        ("prev", (prev,)),
        ("next", (nxt,)),
        ("prev+next", (prev, nxt)),
        ("dep", (dep,)),
        ("dep+dir", (dep, direction)),
        ("dep+headform", (dep, hf)),
        ("prev+dep", (prev, dep)),
        ("next+dep", (nxt, dep)),
        ("prev+dep+headform", (prev, dep, hf)),
        ("prev+next+dep", (prev, nxt, dep)),
        # the parser's analysis of the token's OWN DEPENDENTS -- information the
        # morphologiser's DEP channel does not carry (it reads the token's own deprel only)
        ("form+childdeps", (f,) + kids),
        ("form+dep+childdeps", (f, dep) + kids),
        ("form+childdeps+headupos", (f, hu) + kids),
        ("childdeps", kids),
        ("dep+childdeps", (dep,) + kids),
        ("upos+childdeps", (t["gu"] if use_gold else t["pu"],) + kids),
    ]
    return out


def train_stats(docs, families):
    """gold UPOS distribution per (family, sigvalue), signatures over GOLD structure."""
    st = collections.defaultdict(collections.Counter)
    for d in docs:
        for i in range(len(d)):
            for fam, val in sigs(d, i, True):
                if fam not in families:
                    continue
                st[(fam, val)][d[i]["gu"]] += 1
    return st


PRIORITY = ["prev+form+next", "form+dep+headform", "form+prev+dep", "form+next+dep",
            "form+dep+headdep", "form+headform", "form+dep+dir", "form+dep",
            "form+prev", "form+next", "form",
            "prev+dep+headform", "prev+next+dep", "prev+dep", "next+dep",
            "dep+headform", "prev+next", "dep+dir", "prev", "next", "dep",
            "form+childdeps+headupos", "form+dep+childdeps", "form+childdeps",
            "dep+childdeps", "upos+childdeps", "childdeps"]
PRIO = {f: i for i, f in enumerate(PRIORITY)}


def score_individual(docs, rules):
    """Each rule scored on its own, ignoring every other rule."""
    per = collections.defaultdict(lambda: [0, 0, 0])
    for d in docs:
        for i in range(len(d)):
            t = d[i]
            for fam, val in sigs(d, i, False):
                key = (fam, val)
                if key not in rules:
                    continue
                L = rules[key]
                if t["pu"] == L:
                    continue
                per[key][0] += 1
                if t["gu"] == L:
                    per[key][1] += 1
                elif t["gu"] == t["pu"]:
                    per[key][2] += 1
    return per


def score(docs, rules):
    """Cascade: the most specific matching rule wins. Per-rule stats are for tokens it OWNS."""
    per = collections.defaultdict(lambda: [0, 0, 0])
    agg = [0, 0, 0]
    for d in docs:
        for i in range(len(d)):
            t = d[i]
            hits = [(PRIO[fam], fam, val) for fam, val in sigs(d, i, False)
                    if (fam, val) in rules and rules[(fam, val)] != t["pu"]]
            if not hits:
                continue
            _, fam, val = min(hits)
            key = (fam, val)
            L = rules[key]
            per[key][0] += 1
            agg[0] += 1
            if t["gu"] == L:
                per[key][1] += 1
                agg[1] += 1
            elif t["gu"] == t["pu"]:
                per[key][2] += 1
                agg[2] += 1
    return per, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--min-dom", type=float, default=0.90)
    ap.add_argument("--min-dev-net", type=int, default=3)
    ap.add_argument("--families", default="form+dep,form+dep+headform,form+prev,form+next,"
                                          "prev+form+next,form+headform,form+dep+dir,"
                                          "form+dep+headdep,form+prev+dep,form+next+dep,form")
    ap.add_argument("--out", default="/tmp/lzh_upos_rules.json")
    a = ap.parse_args()
    fams = a.families.split(",")

    tr = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "train.json"))
    dev = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "dev.json"))
    te = json.load(open(os.environ.get("LZH_DUMP", "/tmp/lzh_") + "test.json"))

    st = train_stats(tr, set(fams))
    cand = {}
    for (fam, val), c in st.items():
        n = sum(c.values())
        if n < a.min_support:
            continue
        L, k = c.most_common(1)[0]
        if k / n < a.min_dom:
            continue
        cand[(fam, val)] = L
    print("candidate cells (train gold, dom>=%.2f, n>=%d): %d"
          % (a.min_dom, a.min_support, len(cand)))

    # score each candidate individually on dev, keep the net-positive ones
    perdev = score_individual(dev, cand)
    keep = {}
    for key, L in cand.items():
        fire, fix, brk = perdev.get(key, [0, 0, 0])
        if fix - brk >= a.min_dev_net:
            keep[key] = L
    print("kept after dev net >= %d: %d" % (a.min_dev_net, len(keep)))

    for name, ds in (("dev", dev), ("test", te)):
        per, agg = score(ds, keep)
        print("\n=== %s: %d rules, fire %d fix %d break %d NET %+d"
              % (name, len(keep), agg[0], agg[1], agg[2], agg[1] - agg[2]))
        rows = sorted(per.items(), key=lambda kv: -(kv[1][1] - kv[1][2]))
        for (fam, val), (fire, fix, brk) in rows[:40]:
            print("  %-18s %-28s -> %-6s fire %4d fix %4d brk %3d net %+4d"
                  % (fam, "/".join(val), keep[(fam, val)], fire, fix, brk, fix - brk))
    json.dump([[k[0], list(k[1]), v] for k, v in keep.items()],
              open(a.out, "w"), ensure_ascii=False)
    print("\nrules -> %s" % a.out)


if __name__ == "__main__":
    main()
