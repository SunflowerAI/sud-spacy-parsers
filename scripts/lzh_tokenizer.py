#!/usr/bin/env python3
"""Character tokeniser for Classical Chinese (lzh).

spaCy has no `lzh` language module, so the lzh pipeline runs on the multi-language
blank (`lang = "xx"`) with this tokeniser registered via `spacy train --code`.

The SUD_Classical_Chinese-Kyoto treebank is tokenised one Han character = one token,
with no whitespace between tokens. So a tokeniser that splits raw text into individual
non-space characters reproduces the treebank tokenisation *exactly* and deterministically
(like en rules / ko eojeol whitespace — a matchable, shippable raw-text tokenisation).

Reversibility: every emitted token is a single surface character; whitespace in the raw
text becomes a token boundary with the preceding token carrying SpaceAfter implicitly via
the spacing array, so concatenating token texts + spaces reproduces the input.
"""
from spacy.tokens import Doc
from spacy.util import registry


@registry.tokenizers("lzh.CharTokenizer.v1")
def make_char_tokenizer():
    def create(nlp):
        return CharTokenizer(nlp.vocab)
    return create


class CharTokenizer:
    """One token per non-whitespace character; whitespace sets the preceding token's
    trailing space (so the original text round-trips)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words, spaces = [], []
        for ch in text:
            if ch.isspace():
                # attach the space to the previous token rather than emitting a token
                if spaces:
                    spaces[-1] = True
                continue
            words.append(ch)
            spaces.append(False)
        if not words:
            return Doc(self.vocab, words=[], spaces=[])
        return Doc(self.vocab, words=words, spaces=spaces)

    # spaCy calls these when (de)serialising the pipeline; nothing to persist.
    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
