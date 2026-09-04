#!/usr/bin/env python
"""Recode lzh's stative-predicate VERBs as ADJ.

THE PROBLEM (user diagnosis, confirmed against the resplit train corpus). Kyoto's UPOS inventory
has no ADJ at all -- classical Chinese adjectives are annotated as a subtype of VERB, distinguished
only by `Degree=Pos` in FEATS (the "positive/base" grade of a gradable predicate):

    VERB+Degree=Pos   17 816 tokens   大 太 重 明 善 寡 皇 小 同 多 強 可 然 遠 正 和 貴 平 長 異 賢 ...
    (out of 109 706 VERB tokens total -- 16.2%)

Sampled by hand: every high-frequency lemma is a canonical stative predicate/adjective (大 "big",
賢 "worthy", 高 "tall", 遠 "far", ...), including two that looked like they might be exceptions on
the surface -- 可 "may/possible" and 然 "so/thus" -- both of which, in these specific Degree=Pos
instances, are used PREDICATIVELY ("可也" "it is permissible", "然" "it is so"), exactly the same
construction as the unambiguous adjectives, not their other (modal-auxiliary / demonstrative)
senses. The treebank's own FEATS annotation has already done the disambiguation; recoding is safe
to do purely from `UPOS=VERB & Degree=Pos`, with no lexical exceptions needed.

⚠ Degree=Equ (1 412 tokens: 如, 若, 猶, 奈, 柰, ...) is DELIBERATELY NOT touched. These are
equative-comparison markers ("A 如 B" = "A resembles/is-like B") -- a different construction, and
these lemmas read as comparison verbs ("to resemble"), not as adjectives, even under the same
"stative predicate" logic that justifies recoding Degree=Pos.

⚠ ONLY THE UPOS COLUMN (field 4) IS TOUCHED. FEATS keeps `VerbForm=Part` where it already had it
(a cross-linguistically ordinary combination -- a participial ADJ retains VerbForm in many UD/SUD
treebanks) and XPOS keeps its own tagset value unchanged. Recoding the label a downstream
component reads is one variable; touching the rest of its input at the same time would not be.

Usage:
    recode_lzh_adj.py --in a.conllu --out a.adjfix.conllu
"""
import argparse
import sys

TARGET_DEGREE = "Degree=Pos"


def recode_line(line):
    if not line.strip() or line.startswith("#") or "\t" not in line:
        return line, False
    f = line.rstrip("\n").split("\t")
    if len(f) < 6:
        return line, False
    if f[3] == "VERB" and TARGET_DEGREE in f[5].split("|"):
        f[3] = "ADJ"
        return "\t".join(f) + "\n", True
    return line, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_changed = n_lines = 0
    with open(a.src, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            out, changed = recode_line(line)
            fout.write(out)
            n_lines += 1
            n_changed += changed
    print(f"  {a.src} -> {a.out}: {n_changed} VERB+Degree=Pos tokens recoded to ADJ "
          f"(of {n_lines} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
