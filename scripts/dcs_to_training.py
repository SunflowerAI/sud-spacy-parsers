#!/usr/bin/env python3
"""Convert DCS CoNLL-U into extra training data for the sa morphologiser / lemmatiser / unsandhi.

DCS's FORM column is ALREADY in the representation this project adopted: a token inside a multiword
token is written unsandhied, a token that is its own orthographic word keeps its sandhied surface
(measured: 0/219 MWT-internal and 0/182 MWT-final carry sandhi, against 36.2 % of standalone). That
is not a coincidence — the representation was derived from DCS — so the data drops straight in.

It supplies LEMMA, UPOS and FEATS on 100 % of tokens, and `Unsandhied=` for the reversal
transducer. It supplies NO dependency annotation and no XPOS, so it can only feed components whose
parser is frozen: the morphologiser, the lemmatiser and `sud_unsandhi`. Never the parser.

Three tagset repairs, all verified against the Vedic treebank (Gender / Number / Person / VerbForm /
Voice already have IDENTICAL value sets, so nothing else is needed):

  * `Case=Cpd` -> `Compound=Yes`. DCS marks a non-final compound member with a pseudo-case; this
    project carries the same information as `Compound`, which is a tokeniser-supplied INPUT feature
    worth +1.30 LAS. Converting rather than dropping keeps that input present on DCS data too.
  * `CONJ` -> `CCONJ`. DCS predates UD v2.
  * `Formation` dropped (is / peri / root / them). DCS-only; keeping it would teach the
    morphologiser a feature the Vedic gold never carries, so it would score as noise at evaluation.

Dependencies are filled with a flat dummy tree (token 1 root, everything else `dep` on it) purely so
`spacy convert` produces a well-formed doc. Those trees are meaningless — keep DCS in TRAIN only and
leave dev/test Vedic, so nothing selects a model on them.

    dcs_to_training.py OUT.conllu TEXTDIR...
"""
import argparse
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_compound_rule  # noqa: E402

DROP_FEATS = ("Formation",)
UPOS_FIX = {"CONJ": "CCONJ"}


def fix_feats(feats):
    if feats == "_":
        return "_"
    out = []
    for kv in feats.split("|"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k in DROP_FEATS:
            continue
        if k == "Case" and v == "Cpd":
            out.append("Compound=Yes")          # the project's own name for this information
            continue
        out.append(f"{k}={v}")
    return "|".join(sorted(set(out))) if out else "_"


def convert(files, out_path):
    stat = collections.Counter()
    suspects = []
    with open(out_path, "w", encoding="utf-8") as out:
        for path in files:
            comments, rows = [], []
            for line in open(path, encoding="utf-8"):
                line = line.rstrip("\n")
                if not line.strip():
                    if rows:
                        emit(out, comments, rows, stat, suspects)
                    comments, rows = [], []
                elif line.startswith("#"):
                    comments.append(line)
                else:
                    rows.append(line.split("\t"))
            if rows:
                emit(out, comments, rows, stat, suspects)
    print(f"-> {out_path}")
    for k, v in stat.most_common():
        print(f"  {k:16s} {v}")
    for sid, tid, form, upos, why in suspects:
        print(f"  \u26a0 {sid} token {tid}: {form} ({upos}) stamped Compound=Yes, but {why}")


def is_range(tid):
    return "-" in tid and tid.split("-")[0].isdigit()


def stamp_implicit_compounds(rows, stat, suspects, sid):
    """Add `Compound=Yes` to every non-final MWT member that is a bare nominal. Mutates `rows`.

    DCS marks a compound member with the pseudo-case `Case=Cpd` on 780 176 of them, which
    `fix_feats` has already renamed by the time this runs — so what is left here are the members
    it did not mark at all. The rule, and why "no morphological features" is not the same test as
    "is a bare stem", are in `sa_compound_rule.py`. What that buys here is one nāmāvalī passage
    whose FEATS were never filled in: three real members (`svasti-daḥ`, `bhāga-karaḥ`,
    `sarva-dehinām`) recovered, and the four `X-aḥ + ca` sandhi joins beside them left alone,
    because `ca` cannot end a compound.
    """
    by_id = {int(c[0]): c for c in rows if not is_range(c[0]) and "." not in c[0]}
    for c in rows:
        if not is_range(c[0]):
            continue
        a, b = (int(x) for x in c[0].split("-"))
        last = by_id.get(b)
        if last is None or not sa_compound_rule.can_end_compound(last[3], last[5]):
            continue
        for i in range(a, b):
            t = by_id.get(i)
            if t is None or not sa_compound_rule.is_implicit_member(t[3], t[5]):
                continue
            t[5] = sa_compound_rule.add_compound(t[5])
            stat["compound_added"] += 1
            if sa_compound_rule.looks_inflected(t[1]):     # tripwire; expected never to fire
                stat["compound_suspect"] += 1
                suspects.append((sid, t[0], t[1], t[3], "the FORM still shows a case ending"))


def emit(out, comments, rows, stat, suspects):
    toks = [c for c in rows if not is_range(c[0])]
    if not toks or any(c[1] in ("_", "") for c in toks):
        stat["skipped"] += 1
        return
    stat["sentences"] += 1
    fixed = []
    for c in rows:
        c = list(c)
        if is_range(c[0]):
            fixed.append(c)                     # MWT range line: passes through
            continue
        c[3] = UPOS_FIX.get(c[3], c[3])
        c[5] = fix_feats(c[5])
        # flat dummy tree — DCS has no syntax; TRAIN-only data, parser frozen
        c[6] = "0" if c[0] == "1" else "1"
        c[7] = "root" if c[0] == "1" else "dep"
        c[8] = "_"
        stat["tokens"] += 1
        fixed.append(c)
    # after `fix_feats`, so the test sees the project's own `Compound=Yes` spelling on the members
    # DCS did mark, and only the unmarked ones reach the rule
    sid = next((c.split("=", 1)[1].strip() for c in comments if c.startswith("# sent_id")), "?")
    stamp_implicit_compounds(fixed, stat, suspects, sid)
    for c in comments:
        out.write(c + "\n")
    for c in fixed:
        out.write("\t".join(c) + "\n")
    out.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    files = []
    for d in a.dirs:
        files += sorted(glob.glob(os.path.join(d, "*.conllu")))
    convert(files, a.out)


if __name__ == "__main__":
    main()
