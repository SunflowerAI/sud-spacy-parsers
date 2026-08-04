"""Harvest the closed set of Latin words that END in -que WITHOUT being host + enclitic.

Latin `-que` "and" is a productive enclitic: the treebanks write `Animosque` as one
orthographic word and analyse it as two syntactic tokens (`Animos` + `que`, CCONJ, `cc`).
Splitting it is therefore the tokeniser's job, and the rule is exception-based — split
`-que` UNLESS the word is one of the lexicalised forms (`neque`, `atque`, `usque`,
`quisque`, `quicumque`, `denique`, `plerumque`, …) or merely happens to end in those three
letters (`relinque`, `oblique`, `aeque`).  That exception set is CLOSED, which is what
makes a rule beat a trained segmenter here; the productive side needs no lexicon at all,
since any host may take the enclitic.

Three sources, and they give evidence of different kinds:

  ITTB, Perseus  mark the fusion with a CoNLL-U multiword-token range (`12-13 Animosque`),
                 so a range line ending in -que is SPLIT evidence and a plain token ending
                 in -que is WHOLE evidence.
  PROIEL         carries no range lines and respaces its own `# text` (`ne que mittatis`),
                 so a productive enclitic is already two space-separated tokens.  Every
                 single token it leaves ending in -que is therefore, by its own analysis,
                 lexicalised — the cleanest WHOLE evidence in the corpus, and the only
                 source for accidental endings like `relinque`.

A form is kept whole when the whole evidence outnumbers the split evidence.  Only four
forms are attested both ways at all (`neque`, `namque`, `nonne`, `itemque`, plus `cumque`
which genuinely differs between authors), so the vote is near-unanimous rather than a
threshold doing real work.

`-ne` and `-ve` are deliberately NOT harvested: they split 3 times in 1013 and 0 times in 4
respectively, while thousands of ordinary ablatives end in `-ne` (`ratione`, `ordine`,
`nomine`).  A `-ne` rule would be all cost and no benefit.

Output is a Python module holding the list as a literal, so the tokeniser carries no data
file.  Forms are stored macron-free and lowercased; `la_tokenizer` strips macrons and
case-folds before looking a word up, so a macronised `nēque` matches without the list
having to enumerate every macronisation (and without embedding Morpheus-derived vowel
lengths in a wheel that must not carry them).

    .venv/bin/python scripts/build_la_enclitic_lut.py            # train+dev, held-out test report
    .venv/bin/python scripts/build_la_enclitic_lut.py --all      # include test in the list
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path template, has_mwt_ranges)
SOURCES = [
    ("assets_la/SUD_Latin-ITTB/la_ittb-sud-%s.conllu", True),
    ("assets_la/SUD_Latin-Perseus/la_perseus-sud-%s.conllu", True),
    ("assets_la2/SUD_Latin-PROIEL/la_proiel-sud-%s.conllu", False),
]

SUFFIX = "que"
MIN_LEN = len(SUFFIX) + 2          # `neque` is the shortest form worth a decision

_COMBINING = "̄̆"        # macron, breve


def defang(form: str) -> str:
    """Lowercase and strip vowel-length marks — the key the tokeniser looks up."""
    nfd = unicodedata.normalize("NFD", form.lower())
    return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _COMBINING))


def rows(path: Path):
    """Yield (form, is_multiword) per ORTHOGRAPHIC word, skipping tokens inside a range."""
    covered = 0
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        span = re.match(r"^(\d+)-(\d+)$", cols[0])
        if span:
            covered = int(span.group(2))
            yield cols[1], True
            continue
        if not re.match(r"^\d+$", cols[0]):
            continue
        if int(cols[0]) <= covered:
            continue
        yield cols[1], False


def harvest(splits):
    """(whole, split) counters over -que words, plus the PROIEL adjacency evidence."""
    whole, split = collections.Counter(), collections.Counter()
    for template, has_mwt in SOURCES:
        for name in splits:
            path = ROOT / (template % name)
            if not path.exists():                 # Perseus ships no dev
                continue
            if has_mwt:
                for form, is_mwt in rows(path):
                    key = defang(form)
                    if key.endswith(SUFFIX) and len(key) > MIN_LEN:
                        (split if is_mwt else whole)[key] += 1
            else:
                prev = None
                for line in path.read_text(encoding="utf8").splitlines():
                    if not line.strip() or line.startswith("#"):
                        prev = None
                        continue
                    cols = line.split("\t")
                    if not re.match(r"^\d+$", cols[0]):
                        prev = None
                        continue
                    key = defang(cols[1])
                    if cols[1] == "que" and prev is not None:
                        # respaced host + enclitic: the fused spelling would have split
                        split[defang(prev) + SUFFIX] += 1
                    elif key.endswith(SUFFIX) and len(key) > MIN_LEN:
                        whole[key] += 1
                    prev = cols[1]
    return whole, split


def build(splits):
    whole, split = harvest(splits)
    keep = {k for k in whole if whole[k] > split.get(k, 0)}
    return keep, whole, split


def evaluate(keep, splits):
    """Held-out check of 'split -que unless the form is in KEEP', per orthographic word."""
    seen = unseen_split = unseen_whole = 0
    errors = collections.Counter()
    for template, has_mwt in SOURCES:
        if not has_mwt:              # PROIEL never fuses, so it cannot test the split rule
            continue
        for name in splits:
            path = ROOT / (template % name)
            if not path.exists():
                continue
            for form, is_mwt in rows(path):
                key = defang(form)
                if not (key.endswith(SUFFIX) and len(key) > MIN_LEN):
                    continue
                if key in keep:
                    seen += 1
                elif is_mwt:
                    unseen_split += 1
                else:
                    unseen_whole += 1
                if (key not in keep) != is_mwt:
                    errors[(key, is_mwt)] += 1
    total = seen + unseen_split + unseen_whole
    return total, seen, unseen_split, unseen_whole, errors


HEADER = '''"""Latin words ending in -que that are NOT host + enclitic. Generated; do not edit.

Rebuild with `.venv/bin/python scripts/build_la_enclitic_lut.py`, which documents the
evidence each form rests on. Forms are lowercased and macron-free — `la_tokenizer` applies
the same normalisation before looking a word up.
"""

# harvested from %s of SUD_Latin-{ITTB,Perseus,PROIEL}
KEEP_WHOLE = frozenset([
%s])
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--all", action="store_true",
                    help="harvest from test as well (ships a fuller list, no held-out report)")
    ap.add_argument("-o", "--out", default=str(ROOT / "scripts" / "la_enclitics.py"))
    args = ap.parse_args()

    fit = ["train", "dev", "test"] if args.all else ["train", "dev"]
    keep, whole, split = build(fit)
    print(f"harvested from {'+'.join(fit)}: {len(keep)} keep-whole forms "
          f"({sum(whole.values())} whole / {sum(split.values())} split tokens)")
    both = sorted(k for k in keep if split.get(k))
    if both:
        print("  attested both ways (kept by majority):",
              ", ".join(f"{k} {whole[k]}:{split[k]}" for k in both))

    if not args.all:
        total, seen, u_split, u_whole, errors = evaluate(keep, ["test"])
        print(f"\nheld-out test: {total} -que words — in list {seen}, "
              f"unseen and split {u_split}, unseen and whole {u_whole}")
        print(f"  decision errors: {sum(errors.values())} "
              f"({100 * sum(errors.values()) / total:.2f}%)")
        for (form, is_mwt), n in errors.most_common():
            print(f"    {n:4d}  {form:16s} gold={'split' if is_mwt else 'whole'}")

    body = "".join(f"    {form!r},\n" for form in sorted(keep))
    Path(args.out).write_text(HEADER % ("+".join(fit), body), encoding="utf8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
