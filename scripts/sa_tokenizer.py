#!/usr/bin/env python3
"""Input front-end for the Sanskrit model (`sa_sud_vedic`).

The Vedic treebank is romanised **unaccented IAST** (Latin: ā ī ū ṛ ṃ ḥ ś ṣ …). This tokeniser
lets the model accept other common input forms by normalising them to that representation before
whitespace tokenisation:
  * Devanagari (देवनागरी) -> IAST via indic-transliteration (ळ ḻ→ḷ, anusvara → ṃ);
  * accented Vedic IAST -> the udātta/svarita accent marks are stripped (phonemic macrons and
    under/over-dots are kept);
  * plain unaccented IAST passes through unchanged.

Input must be **word-segmented** (space-separated padas) — like the treebank; continuous
saṃhitā/Devanagari needs sandhi splitting, which is out of scope. Runtime dependency:
`pip install indic-transliteration` (only needed when Devanagari is fed; pure-Python, MIT).
"""
import re
import unicodedata

from spacy.tokens import Doc
from spacy.util import registry

# Sanskrit clause-boundary punctuation: Devanagari daṇḍa ।॥ + period/?/!/pipe/slash (|| and //
# fall out as two single tokens). Split off as tokens so clause_parser can segment on them.
_PUNCT = "।॥.?!|/"
_SPLIT = re.compile(r"[^" + re.escape(_PUNCT) + r"]+|[" + re.escape(_PUNCT) + r"]")

# Vedic accent combining marks to drop (keep macron U+0304, dot-below U+0323, dot-above U+0307).
_ACCENTS = {"́", "̀", "̂", "॑", "॒", "᳚", "̱", "́", "̀"}
# transliterator output -> treebank IAST conventions
_FIX = {"ḻ": "ḷ", "Ḻ": "Ḷ"}  # ḻ/Ḻ -> ḷ/Ḷ (Vedic ळ)


def _has_devanagari(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def normalise(text):
    if _has_devanagari(text):
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        text = "".join(_FIX.get(c, c) for c in text)
    # strip Vedic accents (NFD, drop accent marks, NFC)
    text = unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", text) if c not in _ACCENTS))
    return text


@registry.tokenizers("sa.SanskritInputTokenizer.v1")
def make_sanskrit_input_tokenizer():
    def create(nlp):
        return SanskritInputTokenizer(nlp.vocab)
    return create


class SanskritInputTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        norm = normalise(text)
        words, spaces = [], []
        for chunk in norm.split():
            parts = _SPLIT.findall(chunk)          # split punctuation off each word
            for j, part in enumerate(parts):
                words.append(part)
                spaces.append(j == len(parts) - 1)  # space only after a chunk's last token
        if spaces:
            spaces[-1] = False
        if not words:
            words, spaces = [norm or text], [False]
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
