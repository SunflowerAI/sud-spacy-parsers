#!/usr/bin/env python
"""Forward external-sandhi generator in Clay-Sanskrit-Library (CSL) notation.

The Vedic treebank stores every word in *pausa* (pre-sandhi) form, space-separated, with
compounds marked `Compound=Yes` but unjoined. This module applies external sandhi between
adjacent words *within a sentence*, rendering the result in the same CSL convention used
for the UFAL treebank:

  * vowel coalescence -> the left word loses its final vowel and takes ' (short) / " (long);
    the right word's initial vowel becomes the marked result (â ê î ô û / âi âu / macron
    ā ē ī ō ū);  reusable, reversible.
  * yaṇ (i/u/ṛ + dissimilar vowel) -> semivowel on the left, surface form.
  * ayādi (e/o + a -> avagraha ' ; e/o + other vowel -> a + V hiatus; ai/au -> ā/āv).
  * visarga sandhi -> surface (namaḥ ca -> namaś ca, agniḥ iva -> agnir iva, namaḥ astu
    -> namo 'stu, …).
  * final m -> ṃ before consonant; final t / n / stops -> standard surface assimilation.
  * pragṛhya duals (Number=Dual ending in ī/ū/e) keep hiatus (no sandhi).

Joiner: a hyphen inside a `Compound=Yes` run (internal boundary), a space between separate
words (external). Sentence-initial and sentence-final words are left in pausa (pause = no
sandhi).

NB there is no gold sandhied form in the treebank, so this is rule-based *generation*,
not alignment — validate the output linguistically.
"""

VOW = "aāiīuūṛṝḷeo"
SHORT = set("aiuṛḷ")
LONG = set("āīūṝ")
VOICED_C = set("gjḍdbŋñṇnmyrlvh")          # voiced consonants (trigger visarga -> o/r etc.)
APOS, DAPOS = "'", '"'


def _last_vowel(s):
    if s.endswith(("ai", "au")):
        return s[-2:]
    return s[-1] if s and s[-1] in VOW else None


def _first_vowel(s):
    if s[:2] in ("ai", "au"):
        return s[:2]
    return s[0] if s and s[0] in VOW else None


def _coalesce(v1, v2):
    """savarṇa-dīrgha / guṇa / vṛddhi -> (surface result vowel, CSL right-initial mark)."""
    if v1 in ("a", "ā"):
        return {"a": ("ā", "â"), "ā": ("ā", "ā"), "i": ("e", "ê"), "ī": ("e", "ē"),
                "u": ("o", "ô"), "ū": ("o", "ō"), "e": ("ai", "âi"), "ai": ("ai", "ai"),
                "o": ("au", "âu"), "au": ("au", "āu")}.get(v2)
    if v1 in ("i", "ī") and v2 in ("i", "ī"):
        return ("ī", "î" if v2 == "i" else "ī")
    if v1 in ("u", "ū") and v2 in ("u", "ū"):
        return ("ū", "û" if v2 == "u" else "ū")
    return None


_YAN = {"i": "y", "ī": "y", "u": "v", "ū": "v", "ṛ": "r", "ṝ": "r"}
# visarga before a voiceless consonant -> sibilant (else keep ḥ, incl. before k/kh/p/ph/sibilants)
_VIS_BEFORE = {"c": "ś", "ch": "ś", "ṭ": "ṣ", "ṭh": "ṣ", "t": "s", "th": "s"}
# final dental stop t assimilations before the next word's initial
_T_ASSIM = {"c": "c", "ch": "c", "j": "j", "jh": "j", "ṭ": "ṭ", "ḍ": "ḍ",
            "l": "l", "ś": "c", "n": "n", "m": "n", "h": "d"}  # (t+h -> d, h->dh handled on R)


def is_pragrhya(form, feats):
    """Dual ī/ū/e endings (and a few particles) take no sandhi before a vowel."""
    if feats and "Number=Dual" in feats and form and form[-1] in ("ī", "ū", "e"):
        return True
    return form in ("amī", "o")            # common pragṛhya particles


def join_pair(L, R, feats_L="", internal=False):
    """Apply sandhi at the L|R junction. Returns (L_out, R_out): the surface forms with CSL
    marks. Either may be modified; the divider (hyphen/space) is added by the caller.
    `internal=True` (compound / preverb / a-/an- prefix junctions) suppresses external-only
    rules — namely the -n -> -nn gemination, which does not apply to a bound prefix."""
    if not L or not R or L == "_" or R == "_":
        return L, R                                        # elided word: no surface, blocks sandhi
    v1, v2 = _last_vowel(L), _first_vowel(R)

    # -------- vowel + vowel --------
    if v1 and v2:
        if is_pragrhya(L, feats_L):
            return L, R                                    # hiatus preserved
        c = _coalesce(v1, v2)
        if c:                                              # savarṇa / guṇa / vṛddhi
            _res, mark = c
            short = v1 in ("a", "i", "u")
            return L[:-len(v1)] + (APOS if short else DAPOS), mark + R[len(v2):]
        if v1 in ("a", "ā") and v2 in ("ṛ", "ṝ", "ḷ"):     # guṇa of ṛ/ḷ (a + ṛ -> ar, a + ḷ -> al)
            # word1's vowel is RETAINED (it is the 'a' of 'ar' — nothing merged into a vowel,
            # so no elision mark); only word2's ṛ/ḷ devocalises to r/l, kept on word2 and
            # recoverable by CSL's "initial r-before-consonant ← ṛ" rule (ca ṛṣiḥ -> ca rṣiḥ).
            return L, ("l" if v2 == "ḷ" else "r") + R[len(v2):]
        if v1 in _YAN:                                     # yaṇ: i/u/ṛ + dissimilar
            return L[:-len(v1)] + _YAN[v1], R
        if v1 == "e":                                      # ayādi
            return (L, APOS + R[1:]) if v2 == "a" else (L[:-1] + "a", R)
        if v1 == "o":
            return (L, APOS + R[1:]) if v2 == "a" else (L[:-1] + "a", R)
        if v1 == "ai":
            return L[:-2] + "ā", R
        if v1 == "au":
            return L[:-2] + "āv", R
        return L, R

    # -------- visarga (and word-final -s, which is visarga in pausa: tatas = tataḥ) --------
    if L[-1] in ("ḥ", "s") and len(L) >= 2 and L[-2] in VOW:
        x = L[-2]
        if v2:                                             # before a vowel
            if x == "a" and v2 == "a":
                return L[:-2] + "o", APOS + R[1:]          # aḥ + a -> o '
            if x == "a":
                return L[:-2] + "a", R                     # aḥ + V -> a V (hiatus)
            if x == "ā":
                return L[:-1], R                           # āḥ + V -> ā V (hiatus)
            return L[:-1] + "r", R                         # iḥ/uḥ/… + V -> r
        c2 = R[:2] if R[:2] in ("ch", "th", "ṭh") else (R[:1] if R else "")
        if x == "a" and (not R or R[0] in VOICED_C):
            return L[:-2] + "o", R                         # aḥ + voiced -> o
        if x == "ā" and (not R or R[0] in VOICED_C):
            return L[:-1], R                               # āḥ + voiced -> ā
        if x not in ("a", "ā") and R and R[0] in VOICED_C:
            return L[:-1] + "r", R                         # iḥ/uḥ/… + voiced -> r
        if c2 in _VIS_BEFORE:
            return L[:-1] + _VIS_BEFORE[c2], R             # ḥ + c/ṭ/t… -> ś/ṣ/s
        return L, R                                        # ḥ + k/p/sibilant/pause -> keep ḥ

    # -------- final m --------
    if L[-1] == "m":
        if v2:
            return L, R                                    # m + vowel -> m
        return L[:-1] + "ṃ", R                             # m + consonant -> anusvara

    # -------- final n --------
    if L[-1] == "n":
        if not R:
            return L, R
        r0 = R[0]
        if r0 in ("c", "ch"):
            return L[:-1] + "ṃ", "ś" + R if r0 == "c" else "ś" + R  # n + c/ch -> ṃś...
        if r0 in ("ṭ", "ṭh"):
            return L[:-1] + "ṃ", "ṣ" + R
        if r0 in ("t", "th"):
            return L[:-1] + "ṃ", "s" + R
        if r0 in ("j", "jh", "ś"):
            return L[:-1] + "ñ", R
        if v2 and not internal and len(L) >= 2 and L[-2] in SHORT:
            return L + "n", R                              # short V + n + vowel -> nn (external only)
        return L, R                                        # n + long-V/voiced -> n (kept)

    # -------- final dental t --------
    if L[-1] == "t":
        if v2:
            return L[:-1] + "d", R                         # t + vowel -> d
        if R[0] == "ś":
            return L[:-1] + "c", "ch" + R[1:]              # t + ś -> c ch
        r2 = R[:2] if R[:2] in ("ch",) else (R[:1] if R else "")
        if r2 == "h":
            return L[:-1] + "d", "dh" + R[1:]              # t + h -> d dh
        if r2 in _T_ASSIM:
            return L[:-1] + _T_ASSIM[r2], R
        if R and R[0] in VOICED_C:
            return L[:-1] + "d", R                         # t + voiced -> d
        return L, R                                        # t + voiceless stop -> t

    # -------- other final stops: voice before a voiced sound --------
    if L[-1] in "kṭp":
        if v2 or (R and R[0] in VOICED_C):
            return L[:-1] + {"k": "g", "ṭ": "ḍ", "p": "b"}[L[-1]], R
        return L, R

    return L, R
