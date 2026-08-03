#!/usr/bin/env python3
"""Extract a Sanskrit STEM lexicon from Apte's dictionaries (CDSL `csl-orig`).

Motivation, and the measurement that drives it. The CSLiser's weakest frequent class is the compound
break (`=-`, F 60.2 / recall 52.9), and compound members split sharply by position:

    non-final members  84.6 % seen in training   -- bare STEMS, highly recurrent
    final members      46.2 % seen               -- carry the inflection

Every boundary a lexicon could help place is *internal*, so it is flanked by stems on both sides.
That is exactly what a dictionary is a list of — which makes the stem inventory, not the compound
subentries, the part of Apte worth having. (Attested whole compounds cover only 22.8 % of Vedic test
compounds, and sub-compound matching adds almost nothing there because 93 % are binary; on
classical/epic DCS, 20.5 % have 3+ members, so sub-compounds matter far more in the target domain.)

**The inflection problem cuts both ways.** Apte cites headwords in the NOMINATIVE SINGULAR, not as
stems: `anuyAtraM`, `anuyAtrikaH`, `anuyAtrA`. A compound writes the bare stem (`anuyAtra-`), so a
raw headword list matches almost nothing. In SLP1 the citation endings are the visarga `H` and the
anusvara `M`, both of which are inflection, not stem. Stripping them recovers the a-stem; consonant
stems (`anuyuj`, `anuyAyin`) and ā-stems (`senA`, which compounds as `senA-`) are already correct
and are left alone. Both the stripped and the original form are kept, since either can surface.

Source: https://github.com/sanskrit-lexicon/csl-orig (`v02/ap90`, `v02/ap`).
LICENSING: Apte 1890 is public domain by age, but the CDSL digitisation carries its own terms —
settle them before bundling this in a wheel. Precedent: the Latin macroniser's Morpheus table is
CC BY-SA and could not be bundled with a CC BY-NC-SA wheel, so it ships separately.

    build_apte_lexicon.py --out models/apte_stems.txt
"""
import argparse
import pathlib
import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

HEADWORD = re.compile(r"<k1>([^<]*)")


def stems_from(form):
    """SLP1 citation form -> the shape(s) it takes as a compound member."""
    out = {form}
    # visarga / anusvara are the nom-sg endings Apte cites with; the stem is what compounds use
    if len(form) > 2 and form[-1] in "HM":
        out.add(form[:-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["assets_apte/ap90.txt", "assets_apte/ap.txt"])
    ap.add_argument("--out", default="models/apte_stems.txt")
    ap.add_argument("--min-len", type=int, default=2)
    a = ap.parse_args()

    slp = set()
    for src in a.sources:
        p = pathlib.Path(src)
        if not p.exists():
            print(f"  {src}: missing"); continue
        n = 0
        for line in p.open(encoding="utf-8", errors="replace"):
            if not line.startswith("<L>"):
                continue
            m = HEADWORD.search(line)
            if not m:
                continue
            w = m.group(1).strip()
            if not w or not w.isascii():
                continue
            n += 1
            slp |= stems_from(w)
        print(f"  {src}: {n} headwords")

    iast = set()
    bad = 0
    for w in slp:
        try:
            t = transliterate(w, sanscript.SLP1, sanscript.IAST)
        except Exception:
            bad += 1
            continue
        if len(t) >= a.min_len and t.isalpha():
            iast.add(t)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(iast)), encoding="utf-8")
    print(f"  SLP1 forms {len(slp)} -> IAST stems {len(iast)}"
          + (f" ({bad} untransliterable)" if bad else ""))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
