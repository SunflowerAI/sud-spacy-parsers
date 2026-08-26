#!/usr/bin/env python3
"""Derive a graded word-order profile per language, for the generic parser's typology channel.

NINE HEAD-DIRECTION PARAMETERS, each the FRACTION of arcs in which the dependent precedes its head:

    S-V     subj of a VERB/AUX          Adj-N   ADJ mod of a NOUN/PROPN
    O-V     comp:obj of a VERB          Det-N   det of a NOUN/PROPN
    Obl-V   comp:obl/udep of a VERB     Num-N   NUM mod/det of a NOUN/PROPN
    Neg-V   Polarity=Neg mod of a VERB  Gen-N   Case=Gen mod of a NOUN/PROPN
                                        N-Adp   comp:obj of an ADP (SUD makes the adposition the head)

⚠ GRADED, NOT BINARY, AND THAT IS THE WHOLE POINT. Binarising at 50 % makes **12 of the 13 profiles
distinct** -- only te and ko collide, and only through missing values -- so binary features are a
language identifier in disguise, and a thirteen-row language embedding (i.e. a perfect language
identifier) was already measured at **+0.34 macro LAS, which is nothing**. Worse, the threshold
destroys exactly the informative cases: Latin is genuinely mixed (O-V 49, Obl-V 53, Adj-N 49,
Num-N 52) and thresholding turns four coin flips into confident bits, two of which must be wrong.
Indonesian is the same at Det-N 45 / Num-N 51. The graded value says "free order" where a bit
cannot.

⚠ EVERY FEATURE SHIPS WITH A `known` FLAG, and this is the unset-vs-empty distinction again
(CLAUDE.md; it cost sa 6.8 LAS). A parameter measured on fewer than `--min-arcs` arcs is genuinely
UNMEASURED -- lzh has no ADJ-modifier arcs to speak of, several languages no Case=Gen -- and an
unmeasured 0.5 must not look like a measured 0.5. The value is set to 0.5 and the flag to 0, so the
model can tell the two apart.

⚠ THE HELD-OUT LANGUAGE'S PROFILE IS AN ORACLE IN THE ZERO-SHOT ARMS. Derived here from the full
gold train, including for a language a leave-one-language-out arm holds out -- so a LOLO arm reading
this table is NOT strictly zero-shot; it is zero-shot parsing plus a perfect typological profile.
That is deliberate and it is an UPPER BOUND: if the oracle profile does not help, no realistically
obtainable one (WALS, or a 50-sentence sample) will, and the cheaper experiment is the one worth
running first. If it does help, the realistic variants become worth measuring.

    .venv/bin/python scripts/build_typology.py
    .venv/bin/python scripts/build_typology.py --out assets_vec/typology.json --min-arcs 30
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_generic import LANGS, SA, read_conllu, src_path                       # noqa: E402

#: (name, predicate) -- predicate(dep, row, head_row, feats) says whether this arc counts.
PARAMS = [
    ("S-V",   lambda d, r, h, f: d == "subj" and h[3] in ("VERB", "AUX")),
    ("O-V",   lambda d, r, h, f: d == "comp:obj" and h[3] == "VERB"),
    ("Obl-V", lambda d, r, h, f: d in ("comp:obl", "udep") and h[3] == "VERB"),
    ("Neg-V", lambda d, r, h, f: d == "mod" and h[3] == "VERB" and f.get("Polarity") == "Neg"),
    ("Adj-N", lambda d, r, h, f: d == "mod" and r[3] == "ADJ" and h[3] in ("NOUN", "PROPN")),
    ("Det-N", lambda d, r, h, f: d == "det" and h[3] in ("NOUN", "PROPN")),
    ("Num-N", lambda d, r, h, f: d in ("mod", "det") and r[3] == "NUM"
              and h[3] in ("NOUN", "PROPN")),
    ("Gen-N", lambda d, r, h, f: d == "mod" and f.get("Case") == "Gen"
              and h[3] in ("NOUN", "PROPN")),
    ("N-Adp", lambda d, r, h, f: d == "comp:obj" and h[3] == "ADP"),
]
KEYS = [k for k, _ in PARAMS]


def profile(lang, min_arcs):
    path = SA["train"] if lang == "sa" else src_path(lang, "train")
    counts = collections.defaultdict(lambda: [0, 0])
    for sent in read_conllu(path):
        by = {r[0]: r for r in sent.rows}
        pos = {r[0]: i for i, r in enumerate(sent.rows)}
        for r in sent.rows:
            h = r[6]
            if h == "0" or h not in by:
                continue
            head = by[h]
            dep = r[7].split("@")[0]
            feats = (dict(kv.split("=", 1) for kv in r[5].split("|") if "=" in kv)
                     if r[5] != "_" else {})
            before = pos[r[0]] < pos[h]
            for name, pred in PARAMS:
                if pred(dep, r, head, feats):
                    counts[name][0] += before
                    counts[name][1] += 1
    out = {}
    for name in KEYS:
        n_before, n = counts[name]
        if n >= min_arcs:
            out[name] = {"value": round(n_before / n, 4), "known": 1, "arcs": n}
        else:
            # UNMEASURED, not "in the middle". The flag is what keeps the two apart.
            out[name] = {"value": 0.5, "known": 0, "arcs": n}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets_vec/typology.json")
    ap.add_argument("--min-arcs", type=int, default=30,
                    help="below this an arc type is UNMEASURED, not 0.5")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    table = {l: profile(l, a.min_arcs) for l in LANGS}

    print("fraction of arcs where the DEPENDENT PRECEDES the head "
          "(· = unmeasured, under %d arcs)\n" % a.min_arcs)
    print(f"{'lang':5} " + " ".join(f"{k:>7}" for k in KEYS))
    for l in LANGS:
        cells = []
        for k in KEYS:
            e = table[l][k]
            cells.append(f"{e['value']*100:7.0f}" if e["known"] else f"{'·':>7}")
        print(f"{l:5} " + " ".join(cells))

    # The check that decides whether this channel can be more than a language id.
    bits = {l: "".join(str(int(table[l][k]["value"] >= 0.5)) if table[l][k]["known"] else "?"
                       for k in KEYS) for l in LANGS}
    groups = collections.defaultdict(list)
    for l, b in bits.items():
        groups[b].append(l)
    print(f"\nbinarised: {len(set(bits.values()))} distinct profiles of {len(LANGS)} languages "
          f"-- so BINARY features would be a language id in disguise, which is why this table is "
          f"graded.")
    for b, ls in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(ls) > 1:
            print(f"  collide: {' '.join(ls)}  ({b})")

    meta = {"params": KEYS, "min_arcs": a.min_arcs, "dims_per_param": 2,
            "note": "value in [0,1] plus a `known` flag; ORACLE for held-out languages "
                    "(see the module docstring)"}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "languages": table}, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}  ({len(LANGS)} languages x {len(KEYS)} params x 2 dims)")


if __name__ == "__main__":
    main()
