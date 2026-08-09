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


_CONV = None               # (s2tw, t2s), built once -- this sits in the per-text tokenizer path


def _converters():
    global _CONV
    if _CONV is None:
        import opencc
        _CONV = (opencc.OpenCC("s2tw"), opencc.OpenCC("t2s"))
    return _CONV


def _looks_simplified(text, s2t, t2s):
    """True if `text` is written in SIMPLIFIED characters.

    ⚠ The obvious test -- "would `s2t` change it?" -- is WRONG, and it shipped in zh_sud_gsd 0.2.0:
    traditional input came back simplified. A traditional string is NOT a fixed point of `s2t`,
    because simplification is many-to-one and several of the merged forms are themselves perfectly
    good traditional characters: `s2t` maps 台 -> 臺, 里 -> 裡, 面 -> 麵, 后 -> 後, 只 -> 隻. Any
    traditional sentence containing one of them was read as simplified and `t2s`-converted on the
    way out. On the traditional GSD test that is 45 sentences in 500.

    So ask for evidence of the script that has an EXCLUSIVE inventory instead. A traditional-only
    character (這, 樣, 處, 題) is one `t2s` changes, and simplified text contains none; the second
    clause then requires at least one convertible character, so a sentence with nothing to go on
    (no script-distinguishing character at all) is left alone -- correctly, since the two renderings
    are the same string and no conversion is owed either way.

    Measured against the two aligned treebanks -- SUD_Chinese-GSD and SUD_Chinese-GSDSimp are the
    same 4,997 sentences in the two scripts, so each is a label for the other:

        traditional read as simplified   3 / 4,997      (was 45 / 500 on test alone)
        simplified read as traditional   9 / 4,997
                                        -> 0.120 % over both

    The twelve residuals are the same merged-character ambiguity seen from each side, and they are
    genuine: GSD's own traditional text writes 酒吧里 and 何家干 where 裡/幹 would be expected, and the
    simplified sentences that read as traditional carry the era name 乾德, in which 乾 survives
    unsimplified. A rule cannot settle those; the character really is shared.
    """
    return t2s.convert(text) == text and s2t.convert(text) != text


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
    s2t, t2s = _converters()

    def tokenizer(text):
        simplified = _looks_simplified(text, s2t, t2s)
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

    # lzh gets the SAME component under its own factory name. The argument is identical and was
    # already measured for zh: a both-scripts inventory never pools 個 with 个, so every character
    # competes with its own variant and a ranking over types splits across scripts. lzh's qualitative
    # output showed exactly that -- 遠 and 远, 禮 and 礼, 諸 and 诸 all inside one top-10 list.
    #
    # The traditional-only treebank already exists at the released generation
    # (SUD_Classical_Chinese-Kyoto/*.relabeled_ext.udep_ruled.punct.rulemerged.conllu), so this is a
    # retrain plus this component, not a data rebuild. A separate factory name rather than a
    # parameter because spaCy configs name factories, and lzh's wheel should not carry a pipe called
    # `zh_script`.
    @Language.factory("lzh_script")
    def _make_lzh_script(nlp, name):
        return ZhScript(nlp, name)


# ---------------------------------------------------------------------------------------------
# The tokenizer this module's own docstring has always named -- and which was never implemented.
# `make_zh_trad_tokenizer` above wraps an existing tokenizer OBJECT, which cannot be named in a
# config, and a config that does not name it is the documented trap: `to_disk` writes the config as
# it stands, so a reloaded model rebuilds a plain CharSegTokenizer, converts nothing, and says
# nothing. A registered subclass can be named, so it survives the round trip.
from spacy.util import registry as _registry          # noqa: E402
from char_seg_tokenizer import CharSegTokenizer as _CST   # noqa: E402


class ZhTradTokenizer(_CST):
    """CharSegTokenizer that converts simplified input to traditional before segmenting.

    The model trains on traditional GSD alone, so 個 and 个 do not split one character's probability
    mass across two types. Simplified input is converted here and converted BACK on the way out by
    the `zh_script` component, which reads the flag this leaves in `doc.user_data`.
    """

    def __call__(self, text):
        s2t, t2s = _converters()
        simplified = _looks_simplified(text, s2t, t2s)
        if simplified:
            text = s2t.convert(text)
            text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
        doc = super().__call__(text)
        doc.user_data["zh_input_simplified"] = simplified
        return doc


@_registry.tokenizers("sud.ZhTradTokenizer.v1")
def _make_zh_trad_tokenizer():
    def tokenizer(nlp):
        return ZhTradTokenizer(nlp.vocab)
    return tokenizer
