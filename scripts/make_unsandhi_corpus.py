#!/usr/bin/env python3
"""Build the training corpus for the sandhi-reversal component: FORM -> Unsandhied.

WHY A SEPARATE CORPUS. `Unsandhied=` lives in MISC, and `spacy convert` keeps only
FORM/LEMMA/UPOS/XPOS/FEATS/HEAD/DEPREL — arbitrary MISC keys are dropped, so the gold would never
reach the training docs. Rather than write a custom converter and reader, this copies the CoNLL-U
with the **LEMMA column replaced by the `Unsandhied` value**, so the component can be trained as a
stock `trainable_lemmatizer` (spaCy's edit-tree transducer, which is exactly the right model: it
learns FORM->TARGET string edits and is script-agnostic). At packaging the trained component is
renamed and wrapped so it writes `Token._.unsandhied` instead of `token.lemma_` — see
`scripts/sud_unsandhi.py`.

Tokens with no `Unsandhied` (all of UFAL) get LEMMA `_`, which spaCy treats as "no gold" and skips
in the loss, rather than being taught a spurious identity mapping.

Under the DCS representation most of the work is on STANDALONE tokens: those keep their sandhied
surface, so FORM != target and there is a real edit to learn. Tokens inside an MWT are already
written unsandhied, so their mapping is identity — useful as negative evidence (do not touch these)
but not where the value is.

    make_unsandhi_corpus.py IN.csl_mwt.conllu OUT.conllu
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conllu_misc import misc_get  # noqa: E402


def process(in_path, out_path):
    stat = collections.Counter()
    with open(in_path, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                out.write(line + "\n")
                continue
            c = line.split("\t")
            if "-" in c[0] and c[0].split("-")[0].isdigit():
                out.write(line + "\n")                       # MWT range line: untouched
                continue
            gold = misc_get(c[9], "Unsandhied")
            if not gold or gold == "_" or c[1] == "_":
                # No usable supervision. `_` must NOT be written through: spaCy's CoNLL-U converter
                # keeps it as a LITERAL lemma rather than treating it as missing, so the transducer
                # would be taught `FORM -> "_"` and duly predicts it (measured: 5 043 tokens
                # poisoned this way, and a literal `_` turning up in tokeniser output). Fall back to
                # identity, which is correct for the elided `_` tokens and harmless elsewhere —
                # and keep genuinely unsupervised treebanks out of the corpus entirely.
                # An EMPTY `gold` is caught by the same branch, and must be: writing it would leave
                # column 3 empty, which is a malformed row rather than an absent lemma. Before
                # `conllu_misc` that was the daṇḍa's fate on every line — `Unsandhied=|` split on
                # the separator and came back as `""`.
                c[2] = c[1]
                stat["no_gold"] += 1  # target = FORM (identity), not a literal `_`
            else:
                c[2] = gold
                stat["identity" if gold == c[1] else "edit"] += 1
            out.write("\t".join(c) + "\n")
    n = stat["identity"] + stat["edit"]
    print(f"{in_path} -> {out_path}")
    print(f"  {n} supervised tokens: {stat['edit']} need an edit ({100 * stat['edit'] / max(1, n):.1f} %), "
          f"{stat['identity']} identity; {stat["no_gold"]} without gold (target = identity)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    process(a.inp, a.out)


if __name__ == "__main__":
    main()
