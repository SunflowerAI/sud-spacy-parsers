#!/usr/bin/env python3
"""Build `multifield_tagger`'s decoding contract: the attested XPOS inventory and the UPOS mask.

Two tables, both read straight off the training treebank:

  `attested`  every XPOS code the data contains. Per-field heads range over 4 x 12 x 46 x 84 =
              185 472 combinations and only **121** occur (0.07 %), so without this the joint
              prediction lands off the tagset routinely.
  `allowed`   UPOS -> the XPOS codes it was seen with. A hand-edited UPOS then CONSTRAINS the tag
              rather than merely nudging it: a mean of 11.1 of the 121 codes, and exactly one for
              SCONJ, CCONJ and INTJ. (The released tagger reads UPOS as a feature and it is live —
              forcing a UPOS moves the logits by up to 3.66 — but it never flips the argmax,
              because a counterfactual UPOS is out of distribution for a model trained where
              context predicts both.)

⚠ `allowed` is built from CO-OCCURRENCE, not from a grammar. A UPOS/XPOS pair the treebank happens
never to show is excluded, so the mask can be wrong on genuinely rare-but-valid combinations. Its
purpose is an editing workflow, where the user's UPOS is authoritative and a narrower candidate set
is what they asked for; `upos_mask = false` turns it off.

Usage:
    build_lzh_xpos_tables.py --out models/lzh_xpos_tables.json
"""
import argparse
import collections
import json
import pathlib

TRAIN = ("assets_lzh/SUD_Classical_Chinese-Kyoto/"
         "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--n-fields", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tags = collections.Counter()
    allowed = collections.defaultdict(collections.Counter)
    for line in pathlib.Path(a.train).open(encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        upos, xpos = f[3], f[4]
        if not xpos or xpos == "_":
            continue
        if len(xpos.split(a.sep)) != a.n_fields:
            raise SystemExit(f"{a.train}: {xpos!r} has {len(xpos.split(a.sep))} fields, not "
                             f"{a.n_fields} — this tagset cannot be split on {a.sep!r}")
        tags[xpos] += 1
        allowed[upos][xpos] += 1

    fields = [set() for _ in range(a.n_fields)]
    for t in tags:
        for i, p in enumerate(t.split(a.sep)):
            fields[i].add(p)
    grid = 1
    for f in fields:
        grid *= len(f)
    print(f"attested XPOS codes: {len(tags)}  of a {grid} grid "
          f"({' x '.join(str(len(f)) for f in fields)}) = {len(tags)/grid:.3%}")
    print(f"UPOS values: {len(allowed)}   mean codes admitted: "
          f"{sum(len(v) for v in allowed.values())/len(allowed):.1f}")
    for u, c in sorted(allowed.items(), key=lambda kv: len(kv[1])):
        if len(c) <= 5:
            print(f"   {u:<7}{len(c)} code(s): " + ' '.join(list(c)[:5]))

    payload = {"attested": sorted(tags), "allowed": {u: sorted(c) for u, c in allowed.items()},
               "__meta__": {"train": a.train, "sep": a.sep, "n_fields": a.n_fields}}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
