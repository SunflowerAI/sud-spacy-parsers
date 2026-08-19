#!/usr/bin/env python3
"""Expand a transliteration lexicon by substituting Qieyun-homophonous characters.

WHY THIS IS A GOOD USE OF QIEYUN WHEN TWO OTHERS FAILED. Qieyun did nothing as a CLASSIFICATION
feature -- neither predicting character-pair merges nor identifying transliteration characters
(`NEGATIVE-RESULTS.md`, 2026-08-17). Here it is used as an EQUIVALENCE RELATION, which is what a
rime dictionary actually encodes. The motivating evidence is in the data: Xuanzang writes 揭帝 where
Kumarajiva writes 竭帝, and 揭/竭 share 音韻地位 exactly. Likewise 訶/呵 and 帝/諦.

SCOPE, and why the expansion stays small. Substitutes are restricted to characters that are
THEMSELVES in the induced transliteration inventory, so a homophone that is ordinary classical
vocabulary cannot enter. Only one character is substituted at a time by default: transliteration
variants differ in a syllable or two, not wholesale, and unrestricted substitution explodes
combinatorially into strings that are real classical words.

⚠ WHAT IT WILL NOT CATCH. Exact homophony misses variants where the transcription convention blurs
a distinction Middle Chinese kept -- 般/波, 迦/伽 (voicing), 陀/馱. Relaxing to same-initial+rime
catches 蜜/密 but also loosens precision. Measured: exact homophony reproduces 3 of 6 known variant
pairs and correctly rejects a non-variant control.
"""
import argparse, collections, csv, pathlib

def qieyun(path="assets_qieyun/guangyun.csv"):
    t=collections.defaultdict(set)
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ch,code=row.get("字頭"),row.get("音韻地位")
            if ch and code: t[ch].add(code)
    return t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="assets_wiktionary/zh_sanskrit_terms.txt")
    ap.add_argument("--scores", default="assets_cbeta/translit_scores_lf4.tsv")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--relaxed", action="store_true",
                    help="same initial+rime instead of identical 音韻地位")
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    qy=qieyun()
    inv={l.split("\t")[0] for l in pathlib.Path(a.scores).read_text(encoding="utf-8").split("\n")
         if l.strip() and float(l.split("\t")[1])>=a.threshold}
    key=(lambda c:(c[0],c[-2])) if a.relaxed else (lambda c:c)
    byread=collections.defaultdict(set)
    for ch in inv:
        for code in qy.get(ch,()): byread[key(code)].add(ch)
    terms={t for t in pathlib.Path(a.lexicon).read_text(encoding="utf-8").split("\n") if len(t)>=2}
    out=set(terms)
    for t in terms:
        for i,ch in enumerate(t):
            if ch not in inv: continue
            subs=set()
            for code in qy.get(ch,()): subs |= byread[key(code)]
            for s in subs - {ch}:
                out.add(t[:i]+s+t[i+1:])
    pathlib.Path(a.out).write_text("\n".join(sorted(out)), encoding="utf-8")
    print(f"lexicon {len(terms):,} -> {len(out):,} ({len(out)-len(terms):,} generated variants)")
    print(f"  inventory {len(inv)} chars; equivalence = "
          f"{'initial+rime' if a.relaxed else 'identical 音韻地位'}")

if __name__=="__main__":
    main()
