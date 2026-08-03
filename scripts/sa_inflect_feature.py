#!/usr/bin/env python3
"""Sandhi/inflection-aware word-boundary evidence for the CSLiser.

Apte lists STEMS, but running text contains INFLECTED forms modified by external sandhi, so plain
membership answers the wrong question: direct lookup recovers only 35.7 % of Vedic test tokens.
Stripping a small inventory of endings first takes that to 84.8 % (500 alternations), and it never
materialises stem x ending -- 128 872 x 500 would be 64 M forms.

The economy is that both inventories are closed. Endings are SHARED across thousands of stems (200
alternations cover 77 % of tokens, 500 cover 85 %; past that you are absorbing stem-specific
irregulars), and Sanskrit word-finals are a small set, so reverse sandhi at a junction enumerates a
handful of candidates rather than a search space.

**The stem must be reconstructed with the ending's OWN strip.** `deva` + `-e` surfaces as `dev`+`e`,
so the test is `text[j:k] + 'a' in stems`. A first version instead truncated every stem by 1-3
characters up front; that produced a 292 869-entry set matching almost any substring, and the
feature fired at 81 % of NON-boundary positions (1.2x enrichment -- useless). Pairing each add with
its own strips gives 2.92x enrichment at word breaks and 0.14x at code 0, i.e. "nothing plausible
here" nearly rules a break out.

Emits the same 2-bit code as the other lexicon features: 1 = a plausible word ends here, 2 = one
starts next, 3 = both.
"""
import collections
import json
import pathlib


def build(stems_path, endings_path, min_stem=3):
    stems = {w for w in pathlib.Path(stems_path).read_text(encoding="utf-8").split("\n")
             if len(w) >= min_stem}
    ends = [tuple(x) for x in json.loads(pathlib.Path(endings_path).read_text(encoding="utf-8"))]
    by_add = collections.defaultdict(set)          # add -> the strips that may accompany it
    for rm, add in ends:
        by_add[add].add(rm)
    rms = sorted({rm for rm, _ in ends})
    return stems, by_add, rms


def inflect_codes(text, stems, by_add, rms, maxstem=16, min_stem=3):
    n = len(text)
    # stem_end[rm][k]: some stem ENDS at k once `rm` is restored
    stem_end = {rm: [False] * (n + 1) for rm in rms}
    for rm in rms:
        se = stem_end[rm]
        for k in range(n):
            for L in range(min_stem, min(maxstem, k + 2)):
                if text[k - L + 1:k + 1] + rm in stems:
                    se[k] = True
                    break
    word_end = [False] * n
    for i in range(n):
        for add, strips in by_add.items():
            la = len(add)
            if la and (i - la + 1 < 0 or text[i - la + 1:i + 1] != add):
                continue
            k = i - la
            if k < 0:
                continue
            if any(stem_end[rm][k] for rm in strips if rm in stem_end):
                word_end[i] = True
                break
    starts = [False] * n
    for i in range(n):
        for L in range(min_stem, min(maxstem, n - i) + 1):
            if text[i:i + L] in stems:
                starts[i] = True
                break
    return [(1 if word_end[i] else 0) | (2 if (i + 1 < n and starts[i + 1]) else 0)
            for i in range(n)]
