#!/usr/bin/env python3
"""Merge the two Sanskrit analysers into one per-form CANDIDATE-SET table for the parser.

WHY A SET AND NOT A PREDICTION. `scripts/eval_sa_oracle_noise.py` measured what the parser loses
when its FEATS channel is filled by the morphologiser rather than by gold: **6.1 LAS**, because a
confidently wrong `Case` inverts the attachment the parser is deciding. On the same tokens the
morphologiser is wrong on `Case` 14.6 % of the time; the intersected analysers are *confidently*
wrong 1.5 % of the time and merely uninformative the rest. Same information channel, an order of
magnitude less of the one error that costs.

WHY BOTH ANALYSERS. They fail in opposite directions and the intersection beats either
(Vedic test, tokens carrying gold `Case`):

    analyser     gold in set   pinned & right   pinned & WRONG   silent
    vidyut          0.949          0.316            0.010         0.036
    Heritage        0.795          0.424            0.011         0.191
    intersect       0.954          0.495            0.015         0.025

vidyut generates exhaustively from Pāṇinian rules, so it recognises far more (93.9 % of tokens
against 79.0 %) but offers more analyses each (mean 2.40 `Case` values against 1.72). Heritage is a
curated lexicon: tight, and blind to a fifth of the corpus. Coverage is nearly nested — Heritage
alone accounts for 0.8 % of tokens — so intersecting costs almost no coverage and roughly halves the
ambiguity. Where the intersection would come out EMPTY the union is kept: an empty set is not a
constraint, it is a silent deletion of the channel.

NO JACKKNIFING, AND THAT IS THE POINT. `sud_lex_embed.py` has to fold its table because it is built
FROM the treebank, so a form seen once would be answered at training time and `<OOV>` at inference.
These tables come from outside the treebank entirely, so the training-time and inference-time answer
for a given form are the SAME answer, and there is no leakage to fold away. (Contrast the corpus
lexicon measured earlier: 83.4 % coverage and, worse, sets that look unambiguous only because the
corpus happened to attest one analysis.)

⚠ THE SHIPPED TABLE IS KEYED IN THE TOKENISER'S OWN SCRIPT (IAST), NOT THE ANALYSERS' (SLP1).
The analysers are queried in SLP1 and `--keymap` carries the answers back to the `token.norm_`
strings the model will actually look up. Getting this wrong does not fail: IAST and SLP1 agree on
every diacritic-free form, so the table matched 27 % of tokens instead of 94 % and the layer would
have trained and scored as a slightly weak capacity control.

⚠ THE KEY IS THE PREDICTED PADAPĀṬHA, NOT THE SURFACE AND NOT THE GOLD. Both stores hold pre-pausal
forms, so the lookup key is what `sud_unsandhi` produces — `token.norm_` under
`scripts/make_norm_corpus.py`. Keying on gold `Unsandhied` would build a table indexed by strings
the shipped tokeniser never produces.

    build_analyser_lexicon.py --vidyut lex_vidyut.tsv --heritage lex_heritage.tsv \
        --out scripts/sa_analyser_lut.json.gz
"""
import argparse
import collections
import csv
import gzip
import json
import pathlib

# Closed value sets, in the order their bits are laid out. Taken from the treebank's own inventory
# (corpus_sa_csl_mwt/train): Case 8 values / 58.6 % of tokens, Number 3 / 72.6, Gender 3 / 55.8,
# Person 3 / 14.1. Tense, Mood and VerbForm are NOT here yet — vidyut's `lakara` and Heritage's
# sys-md-*/sys-tp-* codes both carry them, but the mapping is a second table and these four are
# where the syntax is.
VALUES = {
    "Case": ["Nom", "Acc", "Ins", "Dat", "Abl", "Gen", "Loc", "Voc"],
    "Number": ["Sing", "Dual", "Plur"],
    "Gender": ["Masc", "Fem", "Neut"],
    "Person": ["1", "2", "3"],
}
FEATS = list(VALUES)


def read(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for key, found, lemmas, *vals in r:
            out[key] = (int(found), dict(zip(FEATS, [set(filter(None, v.split(","))) for v in vals])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidyut", required=True)
    ap.add_argument("--heritage", required=True)
    ap.add_argument("--keymap", required=True,
                    help="TSV of <SLP1 probe key>\t<the token.norm_ the tokeniser actually emits>. "
                         "The analysers need SLP1; the MODEL looks the form up by norm_, which is "
                         "IAST. Keying the shipped table in SLP1 silently matched only the "
                         "diacritic-free forms — 27 %% of tokens instead of 94 %%.")
    ap.add_argument("--out", default="scripts/sa_analyser_lut.json.gz")
    a = ap.parse_args()

    V, H = read(a.vidyut), read(a.heritage)
    keymap = collections.defaultdict(list)
    with open(a.keymap) as f:
        for slp1, norm in csv.reader(f, delimiter="\t"):
            keymap[slp1].append(norm)
    table, stats = {}, collections.Counter()
    for key in sorted(set(V) | set(H)):
        fv, av = V.get(key, (0, {}))
        fh, ah = H.get(key, (0, {}))
        if not fv and not fh:
            stats["unknown"] += 1
            continue
        bits = {}
        for f in FEATS:
            s1, s2 = av.get(f) or set(), ah.get(f) or set()
            s = (s1 & s2) or (s1 | s2) if (s1 and s2) else (s1 or s2)
            if s2 and s1 and not (s1 & s2):
                stats[f"{f}:disjoint"] += 1
            if s:
                idx = sorted(VALUES[f].index(v) for v in s if v in VALUES[f])
                if idx:
                    bits[f] = idx
        if bits:
            for norm in keymap.get(key, []):
                table[norm] = bits
            stats["known"] += 1
            if not keymap.get(key):
                stats["no norm_ maps to this key"] += 1
        else:
            stats["recognised but no mappable feature"] += 1

    payload = {"values": VALUES, "table": table}
    out = pathlib.Path(a.out)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"{len(table)} forms -> {out}  ({out.stat().st_size/1e6:.2f} MB gz)")
    for k, v in stats.most_common():
        print(f"  {k:<34}{v}")
    for f in FEATS:
        n = sum(1 for b in table.values() if f in b)
        amb = sum(len(b[f]) for b in table.values() if f in b) / max(n, 1)
        pin = sum(1 for b in table.values() if len(b.get(f, [])) == 1)
        print(f"  {f:<10} offered on {n:>6} forms  mean {amb:.2f} values  pinned {pin/max(n,1):.1%}")


if __name__ == "__main__":
    main()
