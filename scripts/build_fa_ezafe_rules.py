#!/usr/bin/env python3
"""Derive `fa_vocalise`'s ezāfe rules from PerDT's own orthography.

The ezāfe -e links a head to its modifier and is the most frequent unwritten vowel in Persian. It
is also the one part of Persian vocalisation that is a SYNTACTIC fact rather than a lexical one --
so unlike everything else in this component, the parser is the right predictor and a pronunciation
dictionary is no help at all.

⚠ NO TREEBANK HERE ANNOTATES IT. Not SUD_Persian-PerDT, not upstream UD_Persian-PerDT, not
UD_Persian-Seraji -- checked all three. So the gold is DERIVED from orthography, using the one
place Persian is forced to write the ezāfe: after a vowel-final stem it must appear as ی or ٔ
(`ابتدا` -> `ابتدای`, `خانه` -> `خانهٔ`). Restricting to vowel-final hosts gives a subset where
presence and ABSENCE of the mark are both real evidence, and a rule can be scored on it.

⚠ The observability test is on the FORM's own ending, not on `form == lemma + ی`. The latter looks
right and is badly wrong: it only catches uninflected stems, so `اعضای` (lemma عضو) and `نیروهای`
(lemma نیرو) -- which plainly DO carry the ezāfe -- were scored as counter-examples. Fixing that
moved the headline `mod` cell from 54.1 % to 85.4 %.

Rules are cells of (host UPOS, dependent's relation, dependent UPOS) where the dependent is the
IMMEDIATELY FOLLOWING token, kept when they dominate past `--thresh` on `--min-count` examples --
the `apply_udep_rules.py` recipe. Derived, never hardcoded. What comes out is linguistically
sensible, which is the check that matters:

    NOUN  mod       ADJ    5377   92.6 %      kept    کتابِ خوب
    NOUN  mod       PROPN  1039   89.9 %      borderline
    NOUN  mod       NOUN   5778   85.0 %      the possessive ezāfe; real but under the bar
    NOUN  mod       PRON   2187   72.9 %      enclitic possessives, mixed
    NOUN  mod       VERB     52    1.9 %      a relative clause takes no ezāfe -- correctly excluded
    NOUN  udep      ADP    1046   25.7 %      the following adposition is not a modifier of the noun
    base rate, no following dependent        12.5 %

    python scripts/build_fa_ezafe_rules.py --thresh 0.85 --min-count 20
"""
import argparse
import collections
import json
from pathlib import Path

DEFAULT_TRAIN = "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-train.conllu"
DEFAULT_OUT = Path(__file__).resolve().parent / "fa_ezafe_rules.json"
HOST = {"NOUN", "ADJ", "PROPN"}
VOWEL = "اوه"
ZWNJ = "‌"


def read(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                sents.append(cur)
            cur = []
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            continue
        cur.append(dict(id=int(f[0]), form=f[1], upos=f[3], head=int(f[6]), rel=f[7]))
    if cur:
        sents.append(cur)
    return sents


def observable(form):
    """(is the ezāfe orthographically obligatory here, is it written)."""
    f = form.rstrip(ZWNJ)
    if f.endswith(("ی", "ٔ")) and len(f) > 1 and f[-2] in VOWEL:
        return True, True
    if f and f[-1] in VOWEL:
        return True, False
    return False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--thresh", type=float, default=0.85)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    cell = collections.defaultdict(lambda: [0, 0])
    base = [0, 0]
    for s in read(a.train):
        kids = collections.defaultdict(list)
        for t in s:
            kids[t["head"]].append(t)
        for t in s:
            if t["upos"] not in HOST:
                continue
            obs, wr = observable(t["form"])
            if not obs:
                continue
            d = [k for k in kids[t["id"]] if k["id"] == t["id"] + 1]
            if not d:
                base[0] += 1
                base[1] += wr
                continue
            c = cell[(t["upos"], d[0]["rel"], d[0]["upos"])]
            c[0] += 1
            c[1] += wr

    rows = sorted(((k, n, w) for k, (n, w) in cell.items() if n >= a.min_count),
                  key=lambda r: -r[1])
    print(f"  {'host':6} {'rel':12} {'dep':6} {'n':>6} {'written':>8} {'prec':>7}  keep")
    keep = {}
    for (h, r, dp), n, w in rows:
        p = w / n
        k = p >= a.thresh
        if k:
            keep["|".join((h, r, dp))] = round(p, 4)
        print(f"  {h:6} {r:12} {dp:6} {n:6d} {w:8d} {p:6.1%}  {'YES' if k else ''}")
    print(f"\n  base rate with no following dependent: {base[1] / base[0]:.1%} "
          f"({base[0]} cases) -- the number every cell above must beat")
    print(f"  kept {len(keep)} cells at thresh {a.thresh} / min-count {a.min_count}")
    if a.stats:
        return
    Path(a.out).write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
