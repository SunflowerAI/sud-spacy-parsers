#!/usr/bin/env python3
"""Stage SUD_Telugu-MTG into `assets_te/`, with the one repair its empty columns force.

WHAT MTG DOES AND DOES NOT CARRY, measured over all three splits (6 465 words, 1 328 sentences):

    LEMMA    empty on EVERY token
    FEATS    115 tokens carry anything at all -- 92 `NumType=Card` and 56 SUD `Shared` -- so
             there is effectively no morphological annotation
    XPOS     a VERBATIM copy of UPOS: zero mismatches in 6 465 tokens

So the Latin/Sanskrit recipe's two parser input channels (LEMMA and decomposed FEATS) have nothing
to read here, and this is why Telugu gets the base arm and the word-order augmenter but no `lemvec`
arm. `docs/dravidian.md` records the library survey that established there is no off-the-shelf
Telugu analyser to fill the columns from.

⚠ **THE LEMMA COLUMN IS A LIVE TRAP, NOT MERELY AN ABSENCE.** `spacy convert --converter conllu`
does `lemmas.append(lemma)` with no special case, and spaCy keeps CoNLL-U `_` as a LITERAL STRING
rather than as missing. Verified on the unrepaired corpus: **all 5 082 training tokens come out
with `token.lemma_ == "_"`.** A lemmatiser trained on that learns `FORM -> "_"` for the whole
language, and `spacy evaluate` reports a LEMMA score against it that looks like an ordinary number.
CLAUDE.md records this costing Sanskrit's sandhi transducer 5 043 tokens, and prescribes the
remedy used here: **fall back to IDENTITY**, not to `_`.

Identity is the right fallback rather than a placeholder because it is what a lemmatiser for an
un-lemmatised language should predict — Telugu's own `token.lemma_` then means "we do not know that
this differs from the form", which is true, instead of meaning "underscore", which is false.

    prep_te.py     # writes assets_te/te_mtg-sud-{train,dev,test}.conllu
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "assets_te" / "SUD_Telugu-MTG" / "te_mtg-sud-%s.conllu"
MWT = ROOT / "assets_te" / "te_mtg-sud-%s.mwt.conllu"
OUT = ROOT / "assets_te" / "te_mtg-sud-%s.conllu"
SPLITS = ("train", "dev", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mwt", action="store_true",
                    help="use MTG's shipped tokenisation, with no multiword tokens at all — the "
                         "unsplit control the split arm is measured against")
    args = ap.parse_args()

    # The MWT split runs FIRST, because the lemma repair below has to cover the words it creates.
    if not args.no_mwt:
        subprocess.run([sys.executable, str(HERE / "split_te_mwt.py"), "--apply"],
                       check=True, stdout=subprocess.DEVNULL)
    src = RAW if args.no_mwt else MWT
    print(f"source: {'MTG as shipped (no MWTs)' if args.no_mwt else 'MWT-split (split_te_mwt.py)'}")

    for split in SPLITS:
        lines, fixed, total = [], 0, 0
        for line in (pathlib.Path(str(src) % split)).read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            # MWT range rows keep their `_`: they are not tokens and carry no annotation.
            if len(cols) == 10 and cols[0].isdigit():
                total += 1
                if cols[2] == "_":
                    cols[2] = cols[1]
                    fixed += 1
                line = "\t".join(cols)
            lines.append(line)
        out = pathlib.Path(str(OUT) % split)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"te {split:5s} {total:5d} words, {fixed} empty lemmas -> identity  ->  {out.name}")


if __name__ == "__main__":
    main()
