#!/usr/bin/env python3
"""Input front-end for the Sanskrit model (`sa_sud_sandhi_csl`).

The model is trained on **CSL-reverted** wordforms: sandhied Clay-Sanskrit-Library text with the
*notation-marked* sandhi undone (vowel coalescence and avagraha) but the unmarked consonant/visarga
sandhi (visarga → -o/-r, m → ṃ, t/n assimilation) left on the surface. This tokeniser maps raw CSL
input to that representation, so the parser sees clean, normalised wordforms.

It normalises input to the treebank's unaccented IAST before whitespace tokenisation:
  * Devanagari (देवनागरी) -> IAST via indic-transliteration (ळ ḻ→ḷ, anusvara → ṃ);
  * accented Vedic IAST -> the udātta/svarita accent marks are stripped (phonemic macrons and
    under/over-dots are kept);
  * plain unaccented IAST passes through unchanged;
then **reverses the CSL-marked sandhi** with `desandhi_csl` (see below).

Input must be **word-segmented** (space-separated padas) — like the treebank; continuous
saṃhitā/Devanagari needs sandhi splitting, which is out of scope. Runtime dependency:
`pip install indic-transliteration` (only needed when Devanagari is fed; pure-Python, MIT).
"""
import re
import unicodedata

from spacy.tokens import Doc
from spacy.util import registry

# Punctuation the treebanks tokenise as separate tokens (Devanagari daṇḍa ।॥, which the
# transliterator renders as |/||, plus the Latin marks the UFAL edition uses). A maximal
# run of the SAME punctuation char is ONE token, so the double daṇḍa ॥ -> "||" stays whole
# (matching UFAL). NB: hyphen-minus '-' is deliberately absent — it is the CSL/MWT-internal
# boundary marker handled by _HYPH below (a lone '-' becomes its own token there); the
# avagraha "'" and the CSL long-elision mark '"' (U+0022) stay attached to their word.
_PUNCT = "।॥|/.?!,;:–—«»‹›”“‘’…()[]"
_PCLASS = re.escape(_PUNCT)
_SPLIT = re.compile(r"[^%s]+|([%s])\1*" % (_PCLASS, _PCLASS))
# A CSL/MWT-internal hyphen stays attached to the element on its LEFT (śrī-śāradā ->
# 'śrī-', 'śāradā'); a lone hyphen (the dash PUNCT, e.g. "ucyate -") is its own token.
_HYPH = re.compile(r"[^-]+-|[^-]+|-")
# CSL prints compound division with a thin vertical line; accept | as a compound-internal
# separator (śrī|śāradā) and normalise it to a hyphen. Only a | that is immediately followed
# by a word character is a compound join — a sentence daṇḍa (।/॥ -> |/||) is always followed
# by space, end, or other punctuation, never directly by a letter.
_PIPE = re.compile(r"\|(?=[^\s%s])" % _PCLASS)
# Straighten typographic (curly) apostrophes and double-apostrophes to the ASCII ' and "
# used for the sandhi marks (avagraha / vowel elision), so smart-quoted input matches the
# model. (CSL quotation uses guillemets « », which are distinct and pass through.)
_STRAIGHTEN = {0x2018: "'", 0x2019: "'", 0x201B: "'", 0x2032: "'",
               0x201C: '"', 0x201D: '"', 0x201F: '"', 0x2033: '"'}

# Vedic pitch-accent combining marks to drop (keep macron U+0304, dot-below U+0323,
# dot-above U+0307). NB the combining circumflex U+0302 is deliberately NOT dropped:
# in the CSL scheme circumflex-on-vowel is a meaningful sandhi-coalescence mark
# (â ê î ô û / âi âu), not an accent, so it must survive normalisation.
_ACCENTS = {chr(cp) for cp in (0x0301, 0x0300, 0x0951, 0x0952, 0x1CDA, 0x0331)}
# transliterator output -> treebank IAST conventions
_FIX = {"ḻ": "ḷ", "Ḻ": "Ḷ"}  # ḻ/Ḻ -> ḷ/Ḷ (Vedic ळ)


def _has_devanagari(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def normalise(text):
    text = text.translate(_STRAIGHTEN)            # curly apostrophes/double-quotes -> ASCII ' "
    if _has_devanagari(text):
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        text = "".join(_FIX.get(c, c) for c in text)
    # Strip Vedic pitch accents (NFD, drop the accent marks, recompose) — but KEEP the
    # combining acute that is part of ś/Ś (s/S + U+0301): it is a phonemic consonant, not
    # an accent. Vedic udātta/svarita only fall on vowels (and vocalic ṛ/ḷ), never on a true
    # consonant like s, so an acute sitting directly on s/S can only be ś/Ś.
    out, base = [], ""
    for c in unicodedata.normalize("NFD", text):
        if not unicodedata.combining(c):
            base = c
            out.append(c)
        elif c in _ACCENTS and not (c == "́" and base in ("s", "S")):  # U+0301 = acute
            continue                      # drop a genuine Vedic accent mark
        else:
            out.append(c)                 # keep macron / dot-below / the ś acute
    return unicodedata.normalize("NFC", "".join(out))


# --------------------------------------------------------------------------------------------
# Reverse the CSL-marked sandhi (the inverse of external_sandhi._coalesce + avagraha), so the
# parser receives the same CSL-reverted wordforms it was trained on. ONLY the notation-marked
# junctions are undone; the unmarked consonant/visarga sandhi (visarga → -o/-r, m → ṃ, t/n
# assimilation) is left on the surface because CSL leaves it ambiguous and it cannot be reversed
# without a lexicon. Operates on the ordered token list (coalescence is a two-token junction).
_APOS, _DAPOS = "'", '"'
_FAMILY_SHORT = {"a": "a", "i": "i", "u": "u"}
_FAMILY_LONG = {"a": "ā", "i": "ī", "u": "ū"}
# inverse of external_sandhi._coalesce: a right word's initial mark -> (left-vowel family, the
# right word's original initial vowel). The left word's final vowel is short/long per its '/" .
_MARK_INV = {
    "â": ("a", "a"), "ā": ("a", "ā"), "ê": ("a", "i"), "ē": ("a", "ī"),
    "ô": ("a", "u"), "ō": ("a", "ū"), "âi": ("a", "e"), "ai": ("a", "ai"),
    "âu": ("a", "o"), "āu": ("a", "au"),
    "î": ("i", "i"), "ī": ("i", "ī"),
    "û": ("u", "u"), "ū": ("u", "ū"),
}
_MARK_INV = {unicodedata.normalize("NFC", k): v for k, v in _MARK_INV.items()}
_MARKS_SORTED = sorted(_MARK_INV, key=len, reverse=True)         # longest first (âi/âu/āu/ai)
# circumflex-bearing marks are UNAMBIGUOUS (a circumflex vowel is never a genuine letter), so a
# word starting with one can be reverted even when its left partner is an unmarked particle.
_CIRC_MARKS = sorted([m for m in _MARK_INV if "̂" in unicodedata.normalize("NFD", m)],
                     key=len, reverse=True)


def _restore_pair(L, R):
    """Coalescence junction: L ends in '/" (maybe before a compound '-'), R starts with a mark."""
    tail, Lc = "", L
    if Lc.endswith("-"):
        tail, Lc = "-", Lc[:-1]
    if not Lc or Lc[-1] not in (_APOS, _DAPOS):
        return None
    short = Lc[-1] == _APOS
    for m in _MARKS_SORTED:
        if R.startswith(m):
            fam, v2 = _MARK_INV[m]
            v1 = (_FAMILY_SHORT if short else _FAMILY_LONG)[fam]
            return Lc[:-1] + v1 + tail, v2 + R[len(m):]
    return None


def _restore_circumflex_start(s):
    for m in _CIRC_MARKS:
        if s.startswith(m):
            return _MARK_INV[m][1] + s[len(m):]                  # restore the right word's vowel
    return s


def _restore_trailing(s):
    """An unpaired '/" — left word elided before an unmarked particle; restore the a-stem vowel."""
    tail = ""
    if s.endswith("-"):
        tail, s = "-", s[:-1]
    if s.endswith(_APOS):
        return s[:-1] + "a" + tail
    if s.endswith(_DAPOS):
        return s[:-1] + "ā" + tail
    return s + tail


def _restore_avagraha(s):
    if s.startswith(_APOS):
        return "a" + s[1:]                                       # avagraha: elided initial a
    if s.startswith(_DAPOS):
        return "ā" + s[1:]                                       # elided initial ā
    return s


def desandhi_csl(words):
    """Undo the CSL-marked (vowel-coalescence + avagraha) sandhi across a token list, preserving
    token count and the unmarked consonant/visarga surface. Returns a new list."""
    out = [unicodedata.normalize("NFC", w) for w in words]
    for i in range(len(out) - 1):
        res = _restore_pair(out[i], out[i + 1])
        if res:
            out[i], out[i + 1] = res
    for i in range(len(out)):
        out[i] = _restore_circumflex_start(out[i])
        out[i] = _restore_trailing(out[i])
        out[i] = _restore_avagraha(out[i])
    return out


@registry.tokenizers("sa.SanskritInputTokenizer.v1")
def make_sanskrit_input_tokenizer():
    def create(nlp):
        return SanskritInputTokenizer(nlp.vocab)
    return create


class SanskritInputTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        norm = _PIPE.sub("-", normalise(text))       # CSL compound | -> hyphen (not the daṇḍa |)
        words, spaces = [], []
        for chunk in norm.split():
            toks = []
            for m in _SPLIT.finditer(chunk):
                if m.group(1) is not None:                # a punctuation run (||, |, , ? …)
                    toks.append(m.group(0))
                else:                                      # a word run: split internal hyphens
                    toks.extend(_HYPH.findall(m.group(0)))
            for j, tk in enumerate(toks):
                words.append(tk)
                spaces.append(j == len(toks) - 1)          # space only after a chunk's last token
        if spaces:
            spaces[-1] = False
        if not words:
            words, spaces = [norm or text], [False]
        words = desandhi_csl(words)                    # reverse CSL-marked sandhi (vowel/avagraha)
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
