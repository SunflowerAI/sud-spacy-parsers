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
  * ayādi (e/o + a -> avagraha ' ; e/o + other vowel -> ay/av + V; ai/au -> āy/āv + V). The glide
    is KEPT (not elided to bare hiatus), so the junction stays unambiguously reversible — bare -a/-ā
    hiatus is then only ever a dropped visarga, and -ay/-av/-āy/-āv only ever ayādi.
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


def _build_coalesce_surface():
    """CSL right-initial mark -> the vowel a genuinely sandhied text would print there.

    Derived by iterating `_coalesce` rather than written out, so it can never drift from the engine.
    CSL splits a coalescence across the junction (the left word keeps `'`/`"`, the right word starts
    with the mark) precisely so it stays reversible; continuous saṃhitā prints the single fused
    vowel instead. This table is the difference between the two representations — and it is the ONLY
    difference, because every other rule in `join_pair` (visarga, -m -> -ṃ, -t assimilation,
    t + ś -> c ch, yaṇ, ayādi with its preserved glide, the avagraha) already emits the true surface,
    which CSL keeps verbatim. That is what makes a saṃhitā <-> CSL character alignment exact.
    """
    import unicodedata
    out = {}
    for v1 in ("a", "ā", "i", "ī", "u", "ū"):
        for v2 in ("a", "ā", "i", "ī", "u", "ū", "ṛ", "ṝ", "ḷ", "e", "o", "ai", "au"):
            got = _coalesce(v1, v2)
            if got:
                surface, mark = got
                out[unicodedata.normalize("NFC", mark)] = surface
    return out


COALESCE_SURFACE = _build_coalesce_surface()
# longest first, so âi/âu/āu beat â/ā when matching a right word's initial mark
COALESCE_MARKS = sorted(COALESCE_SURFACE, key=len, reverse=True)


def mark_surface(mark):
    """The surface vowel for a CSL coalescence mark, or None if `mark` is not one."""
    import unicodedata
    return COALESCE_SURFACE.get(unicodedata.normalize("NFC", mark))


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


# Word-final stop neutralisation. Treebanks write these stems voiced or palatal (tad, kenacid,
# samyag, yuj, vāc); Sanskrit neutralises them word-finally, after which the existing -t / -k/-ṭ/-p
# rules cover every assimilation. The palatal -> velar mapping is the common case (yuj -> yuk,
# vāc -> vāk); the lexical exceptions that go retroflex instead (rāj -> rāṭ, viś -> viṭ) live in
# `sa_tokenizer._LAW_OF_FINALS`.
_FINAL_NEUTRALISE = {"d": "t", "g": "k", "b": "p", "j": "k", "c": "k"}


def pausa_form(w):
    """The form a word takes at a PAUSE (avasāna) — sentence edge, or before a daṇḍa.

    `generate` leaves the first and last word of a sentence unjoined, but "unjoined" is not the same
    as "unchanged": at a pause the word still surfaces in its pausa form. Leaving a bare `-as`,
    `-ar` or `-d` there is the single largest remaining disagreement with DCS's editorial text
    (83 of 156 residual divergences before this was added).
    """
    if not w or w == "_":
        return w
    if len(w) >= 2 and w[-1] in ("s", "r") and w[-2] in VOW:
        return w[:-1] + "ḥ"                                # tatas / punar -> tataḥ / punaḥ
    if w[-1] in _FINAL_NEUTRALISE:
        return w[:-1] + _FINAL_NEUTRALISE[w[-1]]           # tad -> tat, samyag -> samyak
    return w


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
        if v1 == "e":                                      # ayādi: e + a -> e' (a-lopa); e + V -> ay V
            return (L, APOS + R[1:]) if v2 == "a" else (L[:-1] + "ay", R)
        if v1 == "o":                                      # o + a -> o' (a-lopa); o + V -> av V
            return (L, APOS + R[1:]) if v2 == "a" else (L[:-1] + "av", R)
        if v1 == "ai":                                     # ai + V -> āy V
            return L[:-2] + "āy", R
        if v1 == "au":                                     # au + V -> āv V
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
        return L[:-1] + "ḥ", R                             # ḥ + k/p/sibilant/pause -> visarga
        #  ^ a treebank writes this stem-final as -s (tatas, dakṣiṇatas); where nothing assimilates
        #    it SURFACES as visarga, so returning L unchanged left a bare -s that no edition prints.
        #    Caught by validating against DCS's own sandhied text (`dcs_to_samhita.py --validate`).

    # -------- final r: visarga's twin — kept before a vowel or a voiced sound, else visarga -----
    # `punar`, `ahar`, `prātar` are r-stems whose pausa form is -ḥ (punar -> punaḥ). Handled apart
    # from the visarga branch above because -ar must NOT lose its r before a vowel the way -aḥ does.
    if L[-1] == "r" and len(L) >= 2 and L[-2] in VOW:
        if v2 or (R and R[0] in VOICED_C):
            return L, R                                    # punar iva / punar gacchati
        c2 = R[:2] if R[:2] in ("ch", "th", "ṭh") else (R[:1] if R else "")
        if c2 in _VIS_BEFORE:
            return L[:-1] + _VIS_BEFORE[c2], R             # punar ca -> punaś ca
        return L[:-1] + "ḥ", R                             # punar kṛtvā -> punaḥ kṛtvā

    # -------- final voiced stop: neutralise, then fall through to the -t / -k/-ṭ/-p rules -------
    # Treebanks write these stems voiced (tad, kenacid, samyag); word-finally Sanskrit neutralises
    # them, and every assimilation is then already covered below (t + vowel -> d, t + m -> n,
    # t + ś -> c ch, …). Without this the -d forms fell through untouched.
    if L[-1] in _FINAL_NEUTRALISE:
        L = L[:-1] + _FINAL_NEUTRALISE[L[-1]]

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
