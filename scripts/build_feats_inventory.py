#!/usr/bin/env python3
"""Derive the per-feature channel list for `sud.MultiHashEmbedFeats.v1` from a treebank.

Which morphological categories are worth their own embedding table is a property of the TREEBANK,
not something to hardcode: PADT's XPOS restates the whole Arabic analysis, Kyoto's says nothing
about number, and SUD_Korean-GSD populates FEATS on 4.7 % of tokens at all. So this measures it,
in the same spirit as the derived rule tables elsewhere in the project.

Per FEATS key it reports:

    cover    % of tokens carrying the key
    vals     distinct values (including multi-valued bundles like `Case=Acc,Nom`)
    H(X|f)   conditional entropy of XPOS given this feature ALONE, in bits
    IG       information gain, H(XPOS) - H(XPOS|f) -- how much this one category tells you
    IG|form  the SAME gain measured WITHIN each form, i.e. after the token string is already
             known. This is the column that matters: the tagger reads the form anyway, so a
             feature whose information is already carried by the spelling is not worth a channel.

`--emit` prints the config lines. Rows are sized at the next power of two >= 4x the distinct
values (min 8), which keeps hash collisions negligible -- a feature's value set is tiny, so the
tables are nearly free and under-provisioning is the only way to mask a real gain.

    build_feats_inventory.py assets_ar/SUD_Arabic-PADT/ar_padt-sud-train.relabeled_ext.conllu --emit
"""
import argparse
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sud_misc import HOIST_PREFIX, SUD_KEYS               # noqa: E402


def read(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or "-" in f[0] or "." in f[0]:
                continue
            feats = {}
            if f[5] != "_":
                for kv in f[5].split("|"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        feats[k] = v
            rows.append((f[1].lower(), f[4], feats))       # form, XPOS, feats
    return rows


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def cond_entropy(groups):
    """H(XPOS | grouping), weighted by group size."""
    n = sum(sum(c.values()) for c in groups.values())
    return sum(sum(c.values()) / n * entropy(c) for c in groups.values()) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu")
    ap.add_argument("--emit", action="store_true", help="print the config lines")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="drop features whose IG|form is below this (bits)")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="drop features carried by fewer than this %% of tokens")
    ap.add_argument("--keep-sud", action="store_true",
                    help="do not exclude SUD's own FEATS-column keys")
    a = ap.parse_args()

    rows = read(a.conllu)
    n = len(rows)
    keys = collections.Counter(k for _, _, f in rows for k in f)
    H = entropy(collections.Counter(x for _, x, _ in rows))

    # baseline for IG|form: XPOS entropy once the form is known
    by_form = collections.defaultdict(collections.Counter)
    for form, x, _ in rows:
        by_form[form][x] += 1
    H_form = cond_entropy(by_form)

    print(f"{a.conllu}\n{n} tokens, H(XPOS) = {H:.3f} bits, H(XPOS|form) = {H_form:.3f} bits\n")
    print(f"{'feature':12s} {'cover%':>7s} {'vals':>5s} {'H(X|f)':>7s} {'IG':>6s} {'IG|form':>8s}")
    chosen = []
    for key, cnt in keys.most_common():
        vals = {f.get(key, "") for _, _, f in rows}
        by_f = collections.defaultdict(collections.Counter)
        by_ff = collections.defaultdict(collections.Counter)
        for form, x, f in rows:
            v = f.get(key, "")
            by_f[v][x] += 1
            by_ff[(form, v)][x] += 1
        Hf = cond_entropy(by_f)
        ig_form = H_form - cond_entropy(by_ff)
        cover = 100.0 * cnt / n
        print(f"{key:12s} {cover:7.2f} {len(vals):5d} {Hf:7.3f} {H-Hf:6.3f} {ig_form:8.3f}")
        # SUD's own annotation keys ride in the FEATS column (`Shared`, and the `Sud`-prefixed
        # gold hoisted by hoist_sud_gold.py). They are syntactic annotation, not morphology --
        # targets of the MISC layer rather than evidence for XPOS -- so they are excluded by
        # default. Sourced from sud_misc so the two cannot drift.
        is_sud = key in SUD_KEYS or key.startswith(HOIST_PREFIX)
        if is_sud and not a.keep_sud:
            continue
        if cover >= a.min_cover and ig_form >= a.min_gain:
            chosen.append((key, len(vals)))

    if not a.emit:
        return
    def rows_for(v):
        r = 8
        while r < 4 * v:
            r *= 2
        return r
    print(f"\nchosen ({len(chosen)}): IG|form >= {a.min_gain} and cover >= {a.min_cover} %")
    print("feats     = [" + ", ".join(f'"{k}"' for k, _ in chosen) + "]")
    print("feat_rows = [" + ", ".join(str(rows_for(v)) for _, v in chosen) + "]")


if __name__ == "__main__":
    main()
