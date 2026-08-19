#!/usr/bin/env python3
"""Render Tamil MWTT onto TTB's positional XPOS tagset — the `la` treatment, not the `yue` one.

WHY THIS IS NOT OPTIONAL, and why doing nothing looks fine. MWTT's XPOS column is `_` on every
token, TTB's carries a 9-character positional code, and `spacy convert --converter conllu` does

    tag = pos if tag == "_" else tag

(`spacy/training/converters/conllu_to_docs.py`) — it silently falls XPOS back to UPOS. So a
combined corpus does not get a hole in the tagger's target that anything would complain about; it
gets TTB's 234 composite codes sitting beside 14 bare UPOS strings, a MIXED tagset of exactly the
kind `docs/xpos.md` records for PROIEL-beside-ITTB. And the base config weights `tag_acc` at 0.5
against `dep_las` 0.25, so that mixture would be selecting the parser's checkpoints.

THE SCHEME, mined rather than assumed (`--report` reprints this table). Every TTB code is 9
characters, and the tail is a restatement of FEATS:

    pos 0  lexical    coarse POS          N V Z T J A U P R D C Q
    pos 1  lexical    POS subtype         N common, E proper, ...
    pos 2  Case                           N nom, A acc, D dat, G gen, L loc, I inst, S soc
    pos 3  VerbForm x Tense
    pos 4  Person                         1 2 3
    pos 5  Number                         S P
    pos 6  Gender x Polite                N neut, M masc, A ?, H honorific
    pos 7  Voice                          A act, P pass
    pos 8  Polarity                       A affirmative, N negative

Positions 2 and 4-8 are recovered from FEATS at 0.95-1.00 by a SINGLE feature each, so they are
not modelled, they are read off. Only positions 0-1 are lexical, and they are the same shape as
ITTB's letter/digit head: a property of the WORD, not of its inflection.

HOW EACH POSITION IS KEYED. Not by guessing the scheme — by searching it. For every position the
builder scores each single feature and each PAIR of features by how well a majority map from that
key reproduces the character, and keeps the best. That is how pos 3 came out as VerbForm x Tense
and pos 6 as Gender x Polite; keying either on its best SINGLE feature loses the distinction that
the second one carries, and nothing in the tagset documentation says so.

⚠ THE LEXICAL POSITIONS NEED A LADDER, and ITS ORDER IS MEASURED, NOT ASSUMED. `docs/xpos.md`
records the same shape for ITTB's declension letter: a majority over UPOS alone is a bad model of
a lexical property. Four rungs are available — (UPOS, closed-class feats), (UPOS, LEMMA),
(UPOS, form suffix), UPOS — and they do NOT rank the same way at the two positions, which is the
whole reason the order is computed. Ordering them by assumption instead cost pos 1 twenty-five
points: the closed-class rung reproduces pos 1 at 0.79 where the suffix rung reaches 0.98, so
putting `closed` first let a worse rung answer ahead of a better one, and a wrong answer that
pre-empts the right one is worse than no answer (`docs/latin.md` records the same lesson about
`la_macronise`'s key ladder). The builder now scores every rung and sorts.

Idempotent and XPOS-column-only: FORM, LEMMA, UPOS, FEATS, HEAD and DEPREL are untouched, so the
relabel pipeline and every other transform compose with this in any order.

    normalise_ta_xpos.py --learn assets_ta/ta_ttb-sud-train.conllu --report
    normalise_ta_xpos.py --learn assets_ta/ta_ttb-sud-train.conllu \
        --apply assets_ta/ta_mwtt-sud-train.conllu --out assets_ta/ta_mwtt-sud-train.xpos.conllu
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json

#: Candidate keys for the inflectional positions. Every category TTB ever writes.
FEATS = ["Case", "Number", "Gender", "Person", "Polite", "Tense", "VerbForm", "Mood", "Voice",
         "Polarity", "PronType", "Animacy", "NumType", "AdpType", "PunctType", "Reflex", "NumForm"]

#: The lexical positions, which get the form-suffix ladder rather than a FEATS key.
LEXICAL = (0, 1)

#: Closed-class features that genuinely name a SUBTYPE (pos 1), as opposed to inflecting a word.
LEX_FEATS = ("PronType", "NumType", "AdpType", "PunctType", "Reflex", "NumForm")

WIDTH = 9
SUFFIX_LEN = 3


def read(path):
    """Yield (kind, payload). `kind` is 'tok' for a word row and 'raw' for anything else."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            cols = line.split("\t")
            if len(cols) == 10 and cols[0].isdigit():
                yield "tok", cols
            else:
                yield "raw", line


def feats_of(col):
    if col == "_" or not col:
        return {}
    out = {}
    for item in col.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def _majority(pairs):
    """pairs: iterable of (key, char) -> (map, accuracy). Ties broken on the character, for
    reproducibility: a dict iteration order must not decide a tagset."""
    tab = collections.defaultdict(collections.Counter)
    for key, char in pairs:
        tab[key][char] += 1
    table, agree, total = {}, 0, 0
    for key, counter in tab.items():
        char, n = max(counter.items(), key=lambda kv: (kv[1], kv[0]))
        table[key] = char
        agree += n
        total += sum(counter.values())
    return table, (agree / total if total else 0.0)


#: The rungs available to a lexical position, as (name, key function). Ordered by MEASURED
#: accuracy at build time, never by this listing.
def _rungs(row):
    upos, form, feats, lemma = row[0], row[1], row[2], row[4]
    return {
        "closed": "\t".join([upos, *(feats.get(f, "") for f in LEX_FEATS)]),
        "lemma": "\t".join([upos, lemma]),
        "suffix": "\t".join([upos, form[-SUFFIX_LEN:]]),
        "upos": upos,
    }


def learn(rows, report=False):
    """rows: list of (upos, form, feats_dict, xpos, lemma)."""
    model = {"inflect": {}, "lex": {}}
    lines = []

    for pos in range(WIDTH):
        chars = [(r, r[3][pos]) for r in rows if len(r[3]) == WIDTH]
        if pos in LEXICAL:
            continue
        best = None
        singles = [(f,) for f in FEATS]
        pairs = [c for c in itertools.combinations(FEATS, 2)]
        for key in singles + pairs:
            table, acc = _majority(
                (tuple(r[2].get(f, "") for f in key), ch) for r, ch in chars)
            # Prefer the SHORTER key at equal accuracy: a pair that buys nothing is a pair that
            # will miss on a bundle the other treebank writes slightly differently.
            score = (round(acc, 4), -len(key))
            if best is None or score > best[0]:
                best = (score, key, table, acc)
        assert best is not None      # `singles` is non-empty, so the search always has a winner
        _, key, table, acc = best
        model["inflect"][str(pos)] = {"key": list(key), "table": {"\t".join(k): v
                                                                 for k, v in table.items()}}
        lines.append(f"  pos{pos}  {' x '.join(key):28s} {acc:.4f}  ({len(table)} keys)")

    # The lexical ladder, ORDERED BY WHAT EACH RUNG MEASURES rather than by which is the most
    # specific. See the module docstring: assuming the order cost pos 1 twenty-five points.
    for pos in LEXICAL:
        chars = [(r, r[3][pos]) for r in rows if len(r[3]) == WIDTH]
        scored = []
        for name in ("closed", "lemma", "suffix", "upos"):
            table, acc = _majority((_rungs(r)[name], ch) for r, ch in chars)
            scored.append((acc, name, table))
        scored.sort(key=lambda t: (-t[0], t[1]))
        model["lex"][str(pos)] = {"order": [name for _, name, _ in scored],
                                  "tables": {name: table for _, name, table in scored}}
        lines.append(f"  pos{pos}  " + " > ".join(f"{n}={a:.4f}" for a, n, _ in scored)
                     + "  (lexical, ladder in this order)")

    if report:
        print("learned XPOS scheme (accuracy reproducing TTB's own column):")
        print("\n".join(lines))
    return model


def apply_row(model, upos, form, feats, lemma="_"):
    out = []
    keys = _rungs((upos, form, feats, "", lemma))
    for pos in range(WIDTH):
        if pos in LEXICAL:
            spec = model["lex"][str(pos)]
            ch = "-"
            for name in spec["order"]:
                hit = spec["tables"][name].get(keys[name])
                if hit is not None:
                    ch = hit
                    break
        else:
            spec = model["inflect"][str(pos)]
            k = "\t".join(feats.get(f, "") for f in spec["key"])
            ch = spec["table"].get(k, "-")
        out.append(ch)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", required=True, help="CoNLL-U carrying the reference XPOS column")
    ap.add_argument("--apply", help="CoNLL-U to rewrite (default: report only)")
    ap.add_argument("--out")
    ap.add_argument("--model", help="write/read the learned scheme as JSON")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--holdout", help="score the learned scheme against this file's gold XPOS")
    args = ap.parse_args()

    rows = [(c[3], c[1], feats_of(c[5]), c[4], c[2])
            for kind, c in read(args.learn) if kind == "tok" and c[4] != "_"]
    model = learn(rows, report=args.report)
    if args.model:
        with open(args.model, "w", encoding="utf-8") as fh:
            json.dump(model, fh, ensure_ascii=False)
        print(f"wrote {args.model}")

    if args.holdout:
        gold = [(c[3], c[1], feats_of(c[5]), c[4], c[2])
                for kind, c in read(args.holdout) if kind == "tok" and c[4] != "_"]
        exact = sum(apply_row(model, u, f, m, le) == x for u, f, m, x, le in gold)
        per = [0] * WIDTH
        for u, f, m, x, le in gold:
            got = apply_row(model, u, f, m, le)
            for p in range(WIDTH):
                per[p] += (len(x) == WIDTH and got[p] == x[p])
        print(f"held out on {args.holdout}: whole tag {exact}/{len(gold)} = {exact/len(gold):.2%}")
        print("  per position: " + "  ".join(f"{p}:{per[p]/len(gold):.3f}" for p in range(WIDTH)))

    if args.apply:
        out = args.out or args.apply.replace(".conllu", ".xpos.conllu")
        n = 0
        with open(out, "w", encoding="utf-8") as fh:
            for kind, payload in read(args.apply):
                if kind == "raw":
                    fh.write(str(payload) + "\n")
                    continue
                cols = list(payload)
                cols[4] = apply_row(model, cols[3], cols[1], feats_of(cols[5]), cols[2])
                n += 1
                fh.write("\t".join(cols) + "\n")
        print(f"wrote {out}  ({n} tokens re-rendered)")


if __name__ == "__main__":
    main()
