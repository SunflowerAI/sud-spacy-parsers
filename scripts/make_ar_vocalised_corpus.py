#!/usr/bin/env python3
"""Write the Arabic treebank with FORM = Vform, so the augmenter can derive every lighter spelling.

The Latin arm trains on the macronised copy for the same reason: marks can be removed exactly but
not invented, so the pointed text is a strict superset and the bare spelling never has to be
stored. Held out on PADT, `fold(strip(Vform))` reproduces the treebank's own FORM on **97.50 %** of
223 881 train tokens, the two folds being hamza (the vocalised column restores the hamza running
text omits) and Arabic-Indic vs ASCII digits -- both of which are sampled axes, not errors.

FORM only; every other column is copied through untouched, so the trees and the gold are the
treebank's own. Tokens with no `Vform` (there are none in PADT, but the check is free) keep theirs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def convert(src, key="Vform"):
    out, n, tot = [], 0, 0
    for line in open(src, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            out.append(line)
            continue
        tot += 1
        misc = dict(kv.split("=", 1) for kv in f[9].split("|") if "=" in kv)
        v = misc.get(key)
        if v and v != f[1]:
            f[1] = v
            n += 1
        out.append("\t".join(f) + "\n")
    return "".join(out), n, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key", default="Vform")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.srcs:
        text, n, tot = convert(s, a.key)
        p = out / Path(s).name.replace(".conllu", ".vocalised.conllu")
        p.write_text(text, encoding="utf-8")
        print(f"  {Path(s).name}: {n}/{tot} FORMs replaced -> {p}")


if __name__ == "__main__":
    main()
