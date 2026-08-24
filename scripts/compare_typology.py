#!/usr/bin/env python3
"""Agreement between the treebank-derived and database-derived typological profiles.

The two profiles are built for different pools -- treebank for training languages, Grambank/WALS for
test languages -- so they are never compared inside the model. This script compares them on the
languages where BOTH exist, which is the only way to put an error bar on the test-side profiles: if
the databases and the trees disagree on a fifth of the languages we can check, then a fifth of the
test profiles are probably wrong too, and every zero-shot number has to be read against that.

⚠ **DO NOT TUNE THE THRESHOLDS TO MAXIMISE AGREEMENT.** The treebank predicates answer "what does
this treebank annotate", the databases answer "what do descriptive grammars say". Those are
different questions and they are allowed to differ; forcing them together would fit the measuring
instrument to the thing being measured. English is the standing example -- UD annotates `Person` on
87 % of finite verbs because it records agreement potential, not overt affixes, so the treebank
predicate calls English head-marking where WALS 23A does not. That disagreement is a finding about
UD's annotation policy, not a bug in either source.
"""
import argparse
import collections
import json
import pathlib

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]
#: The four 2-bit fields, as (name, index-pair).
PAIRS = [("O/V", 0, 1), ("S/V", 2, 3), ("mark", 4, 5), ("gender", 6, 7)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treebank", default="assets_typ/typology_treebank.json")
    ap.add_argument("--external", default="assets_typ/typology_external.json")
    ap.add_argument("--json", default="assets_typ/typology_agreement.json")
    a = ap.parse_args()

    tb = json.loads(pathlib.Path(a.treebank).read_text(encoding="utf-8"))["languages"]
    ex = json.loads(pathlib.Path(a.external).read_text(encoding="utf-8"))["languages"]
    shared = sorted(set(tb) & set(ex))
    print(f"{len(shared)} languages have both profiles\n")

    per_field = {f: collections.Counter() for f in FIELDS}
    per_pair = {n: collections.Counter() for n, _, _ in PAIRS}
    disagree = collections.defaultdict(list)

    for k in shared:
        t, e = tb[k]["bits"], ex[k]["bits"]
        esrc = ex[k]["sources"]
        for i, f in enumerate(FIELDS):
            if esrc[f] == "none":
                per_field[f]["no-external"] += 1
                continue
            per_field[f]["agree" if t[i] == e[i] else "differ"] += 1
        for name, i, j in PAIRS:
            if esrc[FIELDS[i]] == "none":
                per_pair[name]["no-external"] += 1
                continue
            tv, ev = f"{t[i]}{t[j]}", f"{e[i]}{e[j]}"
            per_pair[name]["same" if tv == ev else "differ"] += 1
            if tv != ev:
                disagree[name].append((k, tv, ev))

    print("per BIT")
    for f in FIELDS:
        c = per_field[f]
        n = c["agree"] + c["differ"]
        rate = c["agree"] / n if n else 0.0
        print(f"  {f:6s} agree {c['agree']:3d}/{n:3d} = {rate:.2f}   (no external: {c['no-external']})")

    print("\nper 2-BIT FIELD (the unit the model actually reads)")
    for name, _, _ in PAIRS:
        c = per_pair[name]
        n = c["same"] + c["differ"]
        rate = c["same"] / n if n else 0.0
        print(f"  {name:7s} identical {c['same']:3d}/{n:3d} = {rate:.2f}   (no external: {c['no-external']})")

    print("\nwhere they differ (treebank -> external)")
    for name, _, _ in PAIRS:
        rows = disagree[name]
        if not rows:
            continue
        print(f"  {name}: {len(rows)}")
        shape = collections.Counter((tv, ev) for _, tv, ev in rows)
        for (tv, ev), n in shape.most_common(6):
            who = [k for k, a, b in rows if a == tv and b == ev][:10]
            print(f"    {tv} -> {ev}  x{n:3d}  {' '.join(who)}")

    out = {"n_shared": len(shared),
           "per_field": {f: dict(per_field[f]) for f in FIELDS},
           "per_pair": {n: dict(per_pair[n]) for n, _, _ in PAIRS},
           "disagreements": {n: disagree[n] for n, _, _ in PAIRS}}
    pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
