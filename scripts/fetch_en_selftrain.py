#!/usr/bin/env python
"""Fetch and clean public-domain English prose for the left-corner parser's self-training pass.

WHY SELF-TRAINING AT ALL. SUD English EWT+GUM is 345 k tokens. That is enough to learn the left-
corner transition distribution but nowhere near enough for the WORD model, and surprisal is only as
good as the word model: dev OOV already sits at 7 %, and reading-time stimuli are out of domain on
top of that. Silver trees add no new syntax the gold does not have, but they add vocabulary and they
add it to the SAME tied embedding table the parser reads -- which is the point.

WHY GUTENBERG PROSE. The eye-tracking targets are narrative: the UCL corpus (Frank et al. 2013) is
sentences from amateur novels, and Provo's paragraphs are news, fiction and popular science. Web
text -- which is what EWT is -- is the wrong register for both. Verse and drama are excluded by
hand below rather than by a filter, because their line structure survives every cleaner and turns
into spurious sentence breaks.

⚠ THE SILVER TREES ARE ONLY AS GOOD AS THE ARM THAT MADE THEM, and errors here are not random: a
parser's systematic attachment biases get amplified by training on its own output. This is why the
self-trained model must be compared against the gold-only one on the SAME held-out gold test set
before it is used for anything (`train_en_lc.sh` step 5), not assumed better for having seen more.
"""
import argparse
import pathlib
import re
import sys
import time
import urllib.request

# Public-domain prose, deliberately weighted to narrative. Verse (Iliad, Odyssey), drama and
# dialogue-only texts are left out: their line breaks survive cleaning and become false sentences.
BOOKS = [
    1342, 11, 84, 1661, 2701, 1400, 98, 1260, 174, 345, 76, 5200, 43, 46, 219,
    158, 161, 105, 141, 768, 730, 766, 1023, 1184, 120, 35, 36, 5230, 164, 103,
    1257, 244, 2852, 74, 16, 55, 236, 271, 113, 514, 205, 2814, 1399, 2554,
    28054, 996, 209, 289, 375, 863, 1695, 2005, 2489, 3268, 2166, 1998, 25344,
]

START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def clean(raw):
    m = START.search(raw)
    if m:
        raw = raw[m.end():]
    m = END.search(raw)
    if m:
        raw = raw[:m.start()]
    # Paragraphs are blank-line separated; rejoin their hard-wrapped lines so the sentenciser sees
    # running prose rather than one "sentence" per typeset line.
    paras = re.split(r"\n\s*\n", raw)
    out = []
    for p in paras:
        p = " ".join(line.strip() for line in p.split("\n")).strip()
        p = re.sub(r"\s+", " ", p)
        # Gutenberg marks italics with underscores. They survive every other cleaning step and
        # become literal `_` TOKENS -- which CoNLL-U reads as its own empty marker, and which spaCy
        # keeps as a literal string rather than as missing (CLAUDE.md's Telugu lemma trap, in the
        # FORM column this time).
        p = p.replace("_", "")
        if len(p) < 80 or p.isupper():          # headings, chapter numbers, running heads
            continue
        if sum(c.isalpha() for c in p) < 0.6 * len(p):
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=6_000_000)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as fh:
        for book in BOOKS:
            if total >= args.max_words:
                break
            url = "https://www.gutenberg.org/cache/epub/%d/pg%d.txt" % (book, book)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    raw = r.read().decode("utf-8", "replace")
            except Exception as exc:
                print("  skip %d: %s" % (book, exc), file=sys.stderr)
                continue
            paras = clean(raw)
            words = sum(p.count(" ") + 1 for p in paras)
            total += words
            for p in paras:
                fh.write(p + "\n")
            print("  %-6d %6d paragraphs %9d words  (total %d)" % (book, len(paras), words, total),
                  file=sys.stderr, flush=True)
            time.sleep(args.sleep)
    print("wrote %s: %d words" % (path, total), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
