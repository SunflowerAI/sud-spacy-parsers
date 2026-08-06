#!/usr/bin/env python3
"""One script inside, either script outside: `sud.ZhTradTokenizer.v1` + the `zh_script` component.

The zh arm used to train on BOTH real treebanks for the same sentences -- SUD_Chinese-GSD
(traditional) and SUD_Chinese-GSDSimp (simplified) -- so it read either script natively. That works,
but it SPLITS the vocabulary: 22.7 % of the type inventory is a cross-script twin (15,848 types
collapse to 12,248 under `t2s`), so 個 and 个 never pool their counts, and any ranking over
types comes out mixed-script.

This instead normalises at the BOUNDARY. Input is converted to traditional, the whole pipeline runs
in one script, and FORM/LEMMA are converted back iff the input was simplified. Same shape as
`sa_deva`, which emits Devanagari iff the input was Devanagari.

WHY THIS IS SAFE IN THE DIRECTION THAT MATTERS.  Simplification is many-to-one, so `s2t` is the
ambiguous direction and `t2s` is a function. Measured on the two treebanks, which are the SAME
98,614 tokens in both scripts:

    s2t    97.016 %   agreement with the traditional gold
    s2tw   98.617 %   <- used here
    s2twp  98.294 %

and 665 of `s2tw`'s 1,364 disagreements are QUOTATION MARKS (`”` vs `」`), a punctuation convention
rather than a script fact -- hence `_PUNCT_MAP`, after which agreement is ~99.3 %. Most of the
remainder is regional preference (台/臺, 意大利/義大利, 分布/分佈) where the treebank's form is one
valid convention among several. Genuine semantic ambiguity is rare: 里/裡 at 22 tokens in 98,614.

The OUTPUT round trip is separately safe: `t2s(s2t(w)) == w` for 12,242 of 12,244 simplified types
(99.98 %), because `t2s` being a function undoes whichever traditional variant `s2t` chose. So a
wrong `s2t` can cost the PARSE, but never the surface string the caller gets back.

⚠ Assigning `nlp.tokenizer` does NOT update the config -- `to_disk` writes the config as it stands,
so a reloaded model rebuilds the stock tokenizer and silently converts nothing.
`nlp.config["nlp"]["tokenizer"]` must be set too. (Same trap as the la enclitic tokeniser.)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    from spacy.language import Language
    from spacy.tokens import Doc, Token
    from spacy.util import registry
except ImportError:                                       # pragma: no cover
    Language = None

# The treebank writes CJK corner brackets; OpenCC leaves Western quotes alone. 665 of 1,364
# disagreements were exactly this, so it is fixed here rather than left to look like a script error.
_PUNCT_MAP = {"“": "「", "”": "」",     # “ ” -> 「 」
              "‘": "『", "’": "』"}     # ‘ ’ -> 『 』
_INV_PUNCT = {v: k for k, v in _PUNCT_MAP.items()}

_SIMP_ONLY = None          # characters that exist in simplified and differ under s2t


def _converters():
    import opencc
    return opencc.OpenCC("s2tw"), opencc.OpenCC("t2s")


def _looks_simplified(text, s2t):
    """True if converting to traditional would CHANGE the text -- i.e. it contains simplified forms.

    Deliberately not a character-set test: a traditional string is already a fixed point of `s2t`,
    so this is exact for the decision being made (do we need to convert?) and costs one conversion.
    """
    return s2t.convert(text) != text


class ZhScript:
    """Restores the input's script on the way out. LAST in the pipeline."""

    def __init__(self, nlp, name="zh_script"):
        self.name = name
        self._s2t, self._t2s = _converters()
        for attr in ("zh_simplified", "zh_trad"):
            if not Token.has_extension(attr):
                Token.set_extension(attr, default=None)

    def __call__(self, doc):
        if not doc.user_data.get("zh_input_simplified"):
            return doc
        words, spaces = [], []
        for token in doc:
            back = self._t2s.convert(token.text)
            back = "".join(_INV_PUNCT.get(ch, ch) for ch in back)
            words.append(back)
            spaces.append(bool(token.whitespace_))
        new = Doc(doc.vocab, words=words, spaces=spaces)
        # Anything that rebuilds a Doc owns carrying EVERY annotation -- this repo has been bitten
        # twice by dropping lemma/morph and once by dropping Token extensions.
        for old, tok in zip(doc, new):
            tok.tag_, tok.pos_, tok.dep_ = old.tag_, old.pos_, old.dep_
            tok.lemma_ = self._t2s.convert(old.lemma_)
            tok.morph = old.morph
            tok._.zh_trad = old.text
            tok._.zh_simplified = tok.text
        new_heads = [t.head.i for t in doc]
        for tok, h in zip(new, new_heads):
            tok.head = new[h]
        new.user_data.update(doc.user_data)
        return new


def make_zh_trad_tokenizer(inner):
    """Wrap a tokenizer so it converts simplified input to traditional before segmenting."""
    s2t, _t2s = _converters()

    def tokenizer(text):
        simplified = _looks_simplified(text, s2t)
        if simplified:
            text = s2t.convert(text)
            text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
        doc = inner(text)
        doc.user_data["zh_input_simplified"] = simplified
        return doc

    tokenizer.inner = inner
    return tokenizer


if Language is not None:
    @Language.factory("zh_script")
    def _make_zh_script(nlp, name):
        return ZhScript(nlp, name)
