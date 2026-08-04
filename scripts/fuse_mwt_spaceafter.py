#!/usr/bin/env python3
"""Give every multiword-token range the `SpaceAfter=No` its sub-tokens need, in place.

CoNLL-U puts the orthographic word on the range line (`12-13 Animosque`) and leaves its
sub-tokens (`12 Animos`, `13 que`) unspaced by CONVENTION -- there is no space inside a
multiword token, so nothing has to say so. Every la range line in ITTB and Perseus relies
on that convention: 0 of 198 marks the host.

`spacy convert --converter conllu` drops range lines and reads MISC for only `SpaceAfter=No`
and the NER pattern, so it reconstructs `Animos que` -- SPACED. The fused spelling then
never appears in the corpus, and because training reads through `sud.GoldTokCorpus.v1` with
`gold_preproc` the tokeniser never runs against it either. The result is a corpus whose raw
text is not the text the treebank transcribes, and a tokeniser failure that no metric can
see.

This makes the convention explicit: `SpaceAfter=No` on every sub-token of a range but the
last, and the range line's own MISC (which spaCy discards) inherited by that last one.
Idempotent, and DEPREL/FORM/FEATS are never touched -- only MISC, and only within a range.

It changes nothing about training: under `gold_preproc` the predicted doc is built from
gold WORDS, so the token sequence is identical either way. What it changes is the raw text,
which is what raw end-to-end evaluation scores and what any future non-gold-preproc arm
would learn from.

    .venv/bin/python scripts/fuse_mwt_spaceafter.py assets_la/la_ittbproiel-sud-*.conllu
    .venv/bin/python scripts/fuse_mwt_spaceafter.py --dry-run <files>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPACE_AFTER = "SpaceAfter=No"


def add_misc(misc: str, feature: str) -> str:
    """MISC is a `|`-separated set; `_` is the empty one, not a value."""
    parts = [] if misc in ("", "_") else misc.split("|")
    if feature in parts:
        return misc
    parts.append(feature)
    return "|".join(sorted(parts))


def merge_misc(target: str, source: str) -> str:
    """Fold the range line's MISC into the last sub-token, keeping what is already there."""
    if source in ("", "_"):
        return target
    out = target
    for feature in source.split("|"):
        out = add_misc(out, feature)
    return out


def fuse(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    changed = 0
    index = 0
    while index < len(lines):
        span = re.match(r"^(\d+)-(\d+)\t", lines[index])
        if not span:
            index += 1
            continue
        first, last = int(span.group(1)), int(span.group(2))
        range_misc = lines[index].split("\t")[9] if lines[index].count("\t") >= 9 else "_"
        # the sub-tokens follow immediately, one line each
        for offset in range(last - first + 1):
            row = index + 1 + offset
            if row >= len(lines):
                break
            cols = lines[row].split("\t")
            if len(cols) < 10 or not re.match(r"^\d+$", cols[0]):
                break
            before = cols[9]
            if offset < last - first:
                cols[9] = add_misc(cols[9], SPACE_AFTER)
            else:
                cols[9] = merge_misc(cols[9], range_misc)
            if cols[9] != before:
                lines[row] = "\t".join(cols)
                changed += 1
        index += last - first + 2
    return "\n".join(lines), changed


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for name in args.files:
        path = Path(name)
        original = path.read_text(encoding="utf8")
        fused, changed = fuse(original)
        total += changed
        flag = "" if changed else "   (already fused)"
        print(f"  {changed:5d} sub-tokens marked  {path}{flag}")
        if changed and not args.dry_run:
            path.write_text(fused, encoding="utf8")
    print(f"{'would mark' if args.dry_run else 'marked'} {total} sub-tokens "
          f"across {len(args.files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
