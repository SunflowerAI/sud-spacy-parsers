#!/usr/bin/env python3
"""Re-emit a CSLiser pairs JSONL under a different input-spacing regime.

The pairs files hold `samhita` in CONTINUOUS form (no spaces at all). `respace` derives the
IAST-spaced and Devanagari-spaced views of the same sentence from the same gold, so this can convert
an existing file without redoing the expensive generation and match-filtering — and all regimes
round-trip to identical CSL, so nothing is lost.

Why it matters: a model trained only on continuous input scores 98.07 split-location F on continuous
DCS test but 88.44 on the IAST-spaced version of the SAME sentences and 91.13 on the
Devanagari-spaced one, despite those having a quarter as many breaks left to find. See
`make_samhita_pairs.respace`.

    respace_pairs.py IN.jsonl OUT.jsonl [--spacing mixed] [--seed 0]
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_samhita_pairs import respace, expand                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--spacing", default="mixed",
                    choices=("continuous", "iast", "devanagari", "mixed"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    stat = collections.Counter()
    bad = 0
    with open(a.inp, encoding="utf-8") as fh, open(a.out, "w", encoding="utf-8") as out:
        for line in fh:
            r = json.loads(line)
            mode = (rng.choice(("continuous", "iast", "devanagari"))
                    if a.spacing == "mixed" else a.spacing)
            s, l = respace(r["samhita"], r["labels"], mode)
            if expand(s, l) != r["csl"]:        # the regimes MUST agree on the CSL they produce
                bad += 1
                continue
            stat[mode] += 1
            out.write(json.dumps({"sent_id": r["sent_id"], "samhita": s, "csl": r["csl"],
                                  "labels": l}, ensure_ascii=False) + "\n")
    print(f"{a.inp} -> {a.out}: {sum(stat.values())} written, regimes {dict(stat)}"
          + (f", {bad} DROPPED on a round-trip mismatch" if bad else ""))


if __name__ == "__main__":
    main()
