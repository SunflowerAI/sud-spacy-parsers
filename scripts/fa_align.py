#!/usr/bin/env python3
"""Place a Persian pronunciation back onto its spelling as diacritics.

The whole Persian problem in one function. Arabic could be done with a lookup table because the
treebank ships the vocalised form; Persian has no such column anywhere, so the vocalisation has to
be RECONSTRUCTED from a pronunciation lexicon -- and a pronunciation is a flat phoneme string with
no indication of which letter each phoneme belongs to. `کتاب` is `k e t A b`, and knowing that the
`e` belongs under the `ک` (giving کِتاب) is an alignment problem, not a lookup.

It is an alignment problem with a specific shape: Persian writes its LONG vowels as letters
(ا و ی) and leaves its SHORT ones (a e o) unwritten, which is exactly the information a diacritic
supplies. So every phoneme either

    * is realised by a grapheme      -- consonants, and the long vowels
    * or is INSERTED                 -- a short vowel, which becomes a diacritic on the grapheme
                                        BEFORE it

and the aligner's job is to decide which. A DP over (grapheme index, phoneme index) finds a path;
`GRAPHEME_PHONEMES` supplies each letter's possible readings, ordered by preference, and the
first complete path wins.

Ambiguous letters are why this is a search rather than a walk: `و` is `v`, `u`, `o` or silent
(`خواهر` = `x A h a r`, where the `و` spells nothing at all); `ه` is `h` or a final `e`; `ی` is
`y`, `i` or a final `A`. A greedy left-to-right pass commits to the wrong reading of the first one
and then cannot finish, so the search has to be able to back out.

Phoneme scheme is Tihu's ASCII one (`A` = ɒː, `S` = ʃ, `C` = tʃ, `Z` = ʒ, `?` = glottal stop),
which is what `build_fa_vocalise_lut.py` normalises WikiPron's IPA into.
"""
from functools import lru_cache

FATHA, KASRA, DAMMA, SHADDA, SUKUN = "َ", "ِ", "ُ", "ّ", "ْ"
SHORT = {"a": FATHA, "e": KASRA, "o": DAMMA}
CONSONANTS = "bptsjChxdzrZSq?kglmnvyf"
# Markers Tihu uses for a compound seam and a syllable break; they correspond to no grapheme.
SKIP = {"^", "_"}

# Each grapheme's possible phoneme readings, PREFERRED FIRST. "" means the letter spells nothing
# (the alef of a hamza seat, the و of خوا). Doubled consonants are generated below rather than
# listed: a geminate is one letter and two phonemes, and comes out as a shadda.
GRAPHEME_PHONEMES = {
    "ا": ["A", "", "a", "e", "o", "?"], "آ": ["A", "?A"], "أ": ["?", "a", ""],
    "إ": ["?", "e", ""], "ؤ": ["?", "v", ""], "ئ": ["?", "y", ""], "ء": ["?", ""],
    "ب": ["b"], "پ": ["p"], "ت": ["t"], "ث": ["s"], "ج": ["j"], "چ": ["C"],
    "ح": ["h"], "خ": ["x"], "د": ["d"], "ذ": ["z"], "ر": ["r"], "ز": ["z"],
    "ژ": ["Z"], "س": ["s"], "ش": ["S"], "ص": ["s"], "ض": ["z"], "ط": ["t"],
    "ظ": ["z"], "ع": ["?", ""], "غ": ["q"], "ف": ["f"], "ق": ["q"], "ک": ["k"],
    "ك": ["k"], "گ": ["g"], "ل": ["l"], "م": ["m"], "ن": ["n"],
    "و": ["v", "u", "o", "", "A", "ow", "uv"], "ه": ["h", "e", ""], "ة": ["e", "t"],
    # "iy"/"uv": one letter doing double duty as its own long vowel AND the following glide,
    # which is how بیاختم = `b i y A x t a m` is spelled with a single ی. Listed after the plain
    # readings so it is only reached when nothing simpler completes.
    "ی": ["y", "i", "", "A", "iy", "ey"], "ي": ["y", "i", "", "iy"], "ى": ["A", "i", "y"],
    "‌": [""], "‍": [""],
}
for _c in "بپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی":
    for _p in GRAPHEME_PHONEMES[_c]:
        if _p and _p in CONSONANTS and _p * 2 not in GRAPHEME_PHONEMES[_c]:
            GRAPHEME_PHONEMES[_c].append(_p * 2)


def align(word, phonemes):
    """Return the diacritised spelling, or None if no alignment exists.

    `phonemes` is a list of Tihu-scheme symbols. None is a real answer -- it means this
    pronunciation cannot be laid onto this spelling, which happens for irregular spellings and for
    a lexicon entry that is simply about a different word. Returning None rather than a guess is
    what keeps the built table trustworthy.
    """
    g = [c for c in word]
    p = [x for x in phonemes if x not in SKIP]
    n, m = len(g), len(p)

    @lru_cache(maxsize=None)
    def solve(i, j):
        """Path from grapheme i, phoneme j. Returns a tuple of (grapheme_index, kind, value)."""
        if i == n and j == m:
            return ()
        if i < n:
            for opt in GRAPHEME_PHONEMES.get(g[i], [g[i]]):
                k = len(opt)
                if k == 0:
                    r = solve(i + 1, j)
                    if r is not None:
                        return (("g", i, ""),) + r
                elif j + k <= m and "".join(p[j:j + k]) == opt:
                    r = solve(i + 1, j + k)
                    if r is not None:
                        return (("g", i, opt),) + r
        # an unwritten short vowel: it belongs to the grapheme just consumed, so it needs one
        if j < m and p[j] in SHORT and i > 0:
            r = solve(i, j + 1)
            if r is not None:
                return (("v", i - 1, p[j]),) + r
        return None

    path = solve(0, 0)
    solve.cache_clear()
    if path is None:
        return None
    marks = {}
    for kind, idx, val in path:
        if kind == "v":
            marks.setdefault(idx, "")
            marks[idx] += SHORT[val]
        elif len(val) == 2 and val[0] == val[1]:
            marks[idx] = SHADDA + marks.get(idx, "")
    out = []
    for i, ch in enumerate(g):
        out.append(ch)
        mk = marks.get(i, "")
        # shadda precedes the vowel it carries: بّـَ not بـَّ
        if SHADDA in mk:
            out.append(SHADDA)
            mk = mk.replace(SHADDA, "")
        out.append(mk)
    return "".join(out)
