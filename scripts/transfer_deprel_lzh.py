#!/usr/bin/env python3
"""Carry a relabel DELTA onto lzh's punctuation-restored, rule-merged CoNLL-U.

WHY NOT REBUILD THE CHAIN. `.relabeled_ext.conllu` -> `.udep_ruled` -> `.punct` -> `.rulemerged`
ends in two steps that do far more than relabel: `align_kanripo_punct.py` INSERTS 100 193 marks
aligned against the Kanseki editions, and `cross_unit_rules.py` MERGES 句讀 units and gives the
merged unit's root a new relation. Re-running both to change a DEPREL would put two large, unrelated
rebuilds inside a relabel change.

WHAT IS TRANSFERRED. Only the DELTA: tokens whose label the new relabel changed against the old one.
Everything the downstream steps decided is left alone, and any token where the target does NOT still
hold the old label is a CONFLICT — the downstream chain has since overwritten it, so the delta must
not be applied blindly there. Conflicts are counted and, unless `--force`, refused.

⚠ ALIGNMENT IS VERIFIED, NOT ASSUMED. The target carries PUNCT tokens the source has not, and its
sentences are merged; so the check is that the FORM sequence of the target's NON-PUNCT tokens equals
the source's FORM sequence, over the whole file. That is the same refusal `transfer_relabel_sa.py`
makes, for the same reason: a silent off-by-one would move every label one token along.

    transfer_deprel_lzh.py OLD.conllu NEW.conllu TARGET.conllu [--out F] [--force]
"""
import argparse
import collections


def rows(p):
    out = []
    for line in open(p, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        # ⚠ PUNCT is excluded from the SOURCE too, not only the target. Kyoto is described as having
        # no punctuation, but it has exactly 5 PUNCT tokens in 374 560 — and excluding them on one
        # side only made the two streams differ by exactly that 5, which is how this was caught.
        if len(f) > 7 and "-" not in f[0] and "." not in f[0] and f[3] != "PUNCT":
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("target")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--old-is-target", action="store_true",
                    help="take the target's own labels as the OLD state. Needed when the old "
                         "relabel output is gone: assets are gitignored, and the on-disk "
                         "`.relabeled_ext.conllu` accumulates later in-place passes, so re-running "
                         "the previous code reproduces relabel_ext but NOT that accumulated state.")
    a = ap.parse_args()

    new = rows(a.new)
    old = rows(a.target) if a.old_is_target else rows(a.old)
    if len(old) != len(new):
        raise SystemExit(f"old/new differ in length: {len(old)} vs {len(new)}")
    if any(o[1] != n[1] for o, n in zip(old, new)):
        raise SystemExit("old/new FORM sequences differ")

    lines = open(a.target, encoding="utf-8").read().split("\n")
    idx = []                                   # line numbers of the target's non-PUNCT tokens
    for i, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) > 7 and "-" not in f[0] and "." not in f[0] and f[3] != "PUNCT":
            idx.append(i)
    if len(idx) != len(old):
        raise SystemExit(f"target has {len(idx)} non-PUNCT tokens, source has {len(old)}")
    mism = [k for k in range(len(idx)) if lines[idx[k]].split("\t")[1] != old[k][1]]
    if mism:
        raise SystemExit(f"FORM mismatch at {len(mism)} positions, first at {mism[0]}")

    # ⚠ A token the RULEMERGE re-headed still reads `root` in the pre-punct source, because the
    # merge happens downstream of it. Never write that back: it would undo 15 394 cross-unit
    # relations (root -> comp:obj 8 759, mod 2 957, parataxis 2 415, conj:coord 1 263 on train).
    delta = [k for k in range(len(old))
             if old[k][7] != new[k][7] and new[k][7] != "root"]
    conflict = [k for k in delta if lines[idx[k]].split("\t")[7] != old[k][7]]
    applied = collections.Counter()
    if conflict and not a.force:
        c = collections.Counter((old[k][7], lines[idx[k]].split("\t")[7]) for k in conflict)
        print(f"REFUSED: {len(conflict)} of {len(delta)} delta tokens no longer hold the old label "
              f"in the target — the downstream chain overwrote them:")
        for (was, now), n in c.most_common(10):
            print(f"    relabel had {was} -> target now holds {now}   {n}")
        raise SystemExit(1)
    for k in delta:
        if k in set(conflict):
            continue
        f = lines[idx[k]].split("\t")
        applied[(f[7], new[k][7])] += 1
        f[7] = new[k][7]
        lines[idx[k]] = "\t".join(f)

    out = a.out or a.target
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"{out}: {sum(applied.values())} of {len(delta)} delta labels applied"
          + (f", {len(conflict)} skipped as conflicts" if conflict else ""))
    for (was, now), n in applied.most_common(8):
        print(f"    {was:16} -> {now:16} {n}")


if __name__ == "__main__":
    main()
