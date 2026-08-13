#!/usr/bin/env python3
"""Render an Arabic (or Persian) treebank at a chosen level of vocalisation.

The Arabic analogue of `make_la_variant_conllu.py`, and it exists for the same reason: to hold the
TREES constant and move only the FORM column, so a row-by-row comparison asks exactly one question
-- how much does this arm lose when the text is written a different way.

Arabic is unusually well set up for this. PADT carries the fully vocalised form on every token
(`Vform=` in MISC), so the whole ladder from bare consonantal skeleton to full tashkīl is derivable
from the treebank itself, with no tool and no guesswork. Real Arabic sits all along that ladder:
newswire is bare, children's books and scripture are full, and a great deal of edited prose is
PARTIAL -- a shadda here, a case ending there, a diacritic placed only where a word would otherwise
be ambiguous.

Levels:
    bare      the treebank as distributed (the consonantal skeleton)
    full      every token replaced by its Vform
    p<N>      N % of tokens fully vocalised, the rest bare -- the realistic mixed case
    shadda    only the shadda kept: the single most commonly written mark
    final     only the diacritics after the last consonant, i.e. the case ending (iʿrāb) alone
    internal  everything EXCEPT the case ending -- what a text writes when it marks the stem but
              leaves iʿrāb to the reader

⚠ FORM only. Nothing here touches the head, deprel, UPOS, FEATS or MISC columns, so every variant
scores against identical gold. Verified by `--check`, which asserts the non-FORM columns are
byte-identical to the source.

    python scripts/make_ar_variant_conllu.py assets_ar/.../ar_padt-sud-test.conllu out_dir/
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_vocalise import DIAC   # noqa: E402

SHADDA = "ّ"
LEVELS = ["bare", "shadda", "final", "internal", "p25", "p50", "p75", "full"]


def strip(s):
    return "".join(c for c in s if c not in DIAC)


def variant(v, level, rng):
    """`v` is the fully vocalised form; return it written at `level`."""
    if level == "bare":
        return strip(v)
    if level == "full":
        return v
    if level == "shadda":
        return "".join(c for c in v if c not in DIAC or c == SHADDA)
    # split at the last consonant: everything after it is the case ending
    idx = max((i for i, c in enumerate(v) if c not in DIAC), default=-1)
    stem, ending = v[:idx + 1], v[idx + 1:]
    if level == "final":
        return strip(stem) + ending
    if level == "internal":
        return stem
    if level.startswith("p"):
        return v if rng.random() < int(level[1:]) / 100 else strip(v)
    raise ValueError(level)


def render(src, level, seed=0, key="Vform"):
    rng = random.Random(seed)
    out = []
    for line in open(src, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            out.append(line)
            continue
        misc = dict(kv.split("=", 1) for kv in f[9].split("|") if "=" in kv)
        v = misc.get(key)
        if v:
            f[1] = variant(v, level, rng)
        out.append("\t".join(f) + "\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--key", default="Vform", help="MISC key holding the vocalised form")
    ap.add_argument("--prefix", default="ar")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = [l.rstrip("\n").split("\t") for l in open(a.src, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    for level in a.levels:
        text = render(a.src, level, a.seed, a.key)
        p = out / f"{a.prefix}_{level}.conllu"
        p.write_text(text, encoding="utf-8")
        # the invariant this whole comparison rests on: only column 2 may differ
        got = [l.rstrip("\n").split("\t") for l in text.splitlines()
               if l.strip() and not l.startswith("#")]
        assert len(got) == len(base), (level, len(got), len(base))
        for g, b in zip(got, base):
            assert g[0] == b[0] and g[2:] == b[2:], (level, g, b)
        n = sum(1 for g, b in zip(got, base) if g[1] != b[1])
        print(f"  {level:9} {p}  ({n} FORMs differ from source)")


if __name__ == "__main__":
    main()
