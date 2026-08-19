#!/usr/bin/env python3
"""Does re-parsing our OWN macronised output help or hurt the parse?

`la_parse_macronised.py --mode reparse` rewrites the tokens with predicted macrons and parses that
string, so `token.text` genuinely carries macrons. The released Latin model is trained on the union
of plain and macronised data, so it accepts either -- but our macrons are ~94 % accurate, and a
WRONG macron is not neutral: it is a false Case cue on exactly the feature the parser leans on.

This scores three conditions on gold tokens (so tokenisation is held fixed and only the surface
changes), against the gold heads/deprels:

    plain     parse the original forms                      (what --mode attach reports)
    ours      parse forms macronised by our own component   (what --mode reparse reports)
    gold      parse forms macronised by the Alatius run     (the ceiling for reparse)

    eval_la_reparse.py build_la_macron/model \
        assets_la/la_ittbproiel-sud-test.conllu \
        assets_la/la_ittbproiel-sud-test.macron.conllu
"""
import argparse
import importlib.util

import spacy
from spacy.tokens import Doc


def load_code(path):
    spec = importlib.util.spec_from_file_location(path.split("/")[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blocks(path):
    cur = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                yield cur
            cur = []
        elif "\t" in line and line.split("\t", 1)[0].isdigit():
            cur.append(line.split("\t"))
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("plain")
    ap.add_argument("macron")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_code("scripts/la_macronise.py")
    nlp = spacy.load(args.model)

    sents = list(zip(blocks(args.plain), blocks(args.macron)))
    if args.limit:
        sents = sents[:args.limit]

    stats = {k: [0, 0, 0] for k in ("plain", "ours", "gold")}  # [uas_hit, las_hit, n]
    for pb, mb in sents:
        forms = [r[1] for r in pb]
        g_head = [int(r[6]) for r in pb]
        g_dep = [r[7] for r in pb]
        # one spacing for all three variants: they differ only in ORTHOGRAPHY (macrons), never in
        # segmentation, so the plain block's SpaceAfter is the right one throughout.
        sp = ["SpaceAfter=No" not in (r[9] if len(r) > 9 else "") for r in pb]
        variants = {"plain": forms, "gold": [r[1] for r in mb]}
        variants["ours"] = [t._.macron for t in nlp(Doc(nlp.vocab, words=forms, spaces=sp))]

        for name, words in variants.items():
            doc = nlp(Doc(nlp.vocab, words=words, spaces=sp))
            if len(doc) != len(forms):
                continue                      # tokenisation moved; skip (should not happen)
            for i, tok in enumerate(doc):
                head = 0 if tok.head.i == tok.i else tok.head.i + 1
                dep = "root" if tok.head.i == tok.i else tok.dep_
                s = stats[name]
                s[2] += 1
                if head == g_head[i]:
                    s[0] += 1
                    if dep == g_dep[i]:
                        s[1] += 1

    print(f"{len(sents)} sentences")
    base = None
    for name in ("plain", "ours", "gold"):
        u, l, n = stats[name]
        uas, las = 100 * u / n, 100 * l / n
        if base is None:
            base = las
        print(f"  {name:6s} UAS {uas:6.2f}  LAS {las:6.2f}  ({las - base:+5.2f} vs plain)")


if __name__ == "__main__":
    main()
