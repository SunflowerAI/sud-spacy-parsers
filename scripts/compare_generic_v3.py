#!/usr/bin/env python3
"""Compare v3 arms and fill regimes, SPLIT BY WHETHER THE LANGUAGE HAS ALIGNED ROWS.

⚠ THE SPLIT IS THE WHOLE ANALYSIS, and a macro over all twenty hides the result completely. Six of
the twenty held-out languages (el, hu, lt, lv, th, vi) are in fastText's aligned-44, so the `lemma`
fill finds REAL vectors for them -- that run is the channel's upper bound, not its floor. For the
other fourteen the same run is genuinely all-OOV. Averaged together the two groups cancel and the
channel looks inert; separated, one group gains +4.5 LAS and the other loses 1.7.
"""
from __future__ import annotations
import argparse, json, statistics, sys

# Emitted with a frequency head in release_vectors_v3; see assets_vec/sources_v3.json meta.test.
IN_TABLE = {"el", "hu", "lt", "lv", "th", "vi"}


def load(p):
    return {r["lang"]: r for r in json.load(open(p, encoding="utf-8"))["languages"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lemma", required=True)
    ap.add_argument("--gloss", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    base, lem, gl = load(a.base), load(a.lemma), load(a.gloss)

    out = {}
    for name, langs in [("has_aligned_rows", sorted(IN_TABLE & set(base))),
                        ("no_rows", sorted(set(base) - IN_TABLE))]:
        print(f"\n=== {name} ({len(langs)}) ===")
        print(f"{'lang':5s} {'base':>7s} {'lemma':>8s} {'d':>7s} {'gloss':>8s} {'d':>7s} {'fill':>6s}")
        dl, dg = [], []
        for l in langs:
            b, m = base[l]["las"] * 100, lem[l]["las"] * 100
            dl.append(m - b)
            row = f"{l:5s} {b:7.2f} {m:8.2f} {m-b:+7.2f}"
            if gl.get(l, {}).get("gloss_source"):
                g = gl[l]["las"] * 100
                dg.append(g - b)
                row += f" {g:8.2f} {g-b:+7.2f} {gl[l]['gloss_fill']:6.1%}"
            print(row)
        out[name] = dict(n=len(langs), mean_lemma_delta=round(statistics.mean(dl), 3),
                         mean_gloss_delta=round(statistics.mean(dg), 3) if dg else None)
        print(f"  mean vs base: lemma {statistics.mean(dl):+.2f}"
              + (f"   gloss {statistics.mean(dg):+.2f}" if dg else ""))

    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
