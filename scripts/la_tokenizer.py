#!/usr/bin/env python3
"""Latin tokeniser that splits the enclitic `-que`.

spaCy's stock `la` rules split nothing ending in `-que`, so `Animosque` arrives as one
token — but ITTB and Perseus write that word fused in the text and analyse it as two
syntactic tokens (`Animos` + `que`, CCONJ, attached `cc`), which is what the parser was
trained on. Real classical orthography therefore reached the released model with a token
boundary missing, costing 1.8 strict token F on Perseus.

The split is rule-separable, which is why this is a rule and not a trained segmenter (the
Indonesian route): the productive side needs no lexicon, since any host may take the
enclitic, and the exception side — words that end in those three letters without being
host + enclitic — is CLOSED. `la_enclitics.KEEP_WHOLE` holds it, harvested by
`build_la_enclitic_lut.py` from all three treebanks. It covers both the lexicalised
compounds (`neque`, `atque`, `usque`, `quisque`, `quicumque`, `plerumque`) and the
accidental endings a naive suffix rule would maul (`relinque`, `oblique`, `aeque`).

Lookup is on the lowercased, macron-free form, so the union parser's macronised input
(`nēque`) matches an ASCII list entry. That also keeps Morpheus-derived vowel lengths out
of a wheel licensed CC BY-NC-SA, which must not carry them.

`-ne` and `-ve` are deliberately left alone: they split 3 times in 1013 and 0 times in 4,
against thousands of ordinary ablatives ending in `-ne` (`ratione`, `ordine`, `nomine`).

Registered as `sud.LatinEncliticTokenizer.v1` and bundled into the wheel with
`spacy package --code`, like the lzh and zh tokenisers.
"""

from __future__ import annotations

import unicodedata

from spacy.lang.la import LatinDefaults
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex, registry


def _sibling(name):
    """Import a sibling module across all three ways this file gets loaded.

    A packaged wheel imports it as `pkg.la_tokenizer`, so a relative import works; a
    `--code` run puts scripts/ on sys.path, so a plain import works; but `spacy package`
    loads each --code file standalone via spec_from_file_location, where NEITHER works --
    hence the file-path fallback.
    """
    import importlib
    import importlib.util
    import pathlib
    import sys as _sys

    if __package__:
        try:
            return importlib.import_module("." + name, __package__)
        except ImportError:
            pass
    if name in _sys.modules:
        return _sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    _sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KEEP_WHOLE = _sibling("la_enclitics").KEEP_WHOLE

SUFFIX = "que"
MIN_LEN = len(SUFFIX) + 2       # `neque` is the shortest word worth a decision
_COMBINING = "̄̆"     # macron, breve


def defang(form: str) -> str:
    """Lowercase and strip vowel-length marks -- the key `KEEP_WHOLE` is stored under."""
    nfd = unicodedata.normalize("NFD", form.lower())
    return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _COMBINING))


def splits_enclitic(form: str) -> bool:
    """Is this orthographic word host + `-que` rather than a word that merely ends in it?"""
    key = defang(form)
    return key.endswith(SUFFIX) and len(key) > MIN_LEN and key not in KEEP_WHOLE


class LatinEncliticTokenizer(Tokenizer):
    """The stock Latin rules, then `-que` peeled off any token that takes it.

    Splitting afterwards rather than through a suffix regex keeps the two decisions
    separate: spaCy's affix machinery has already dealt with punctuation and its own
    exceptions by this point, so the enclitic rule sees whole words and nothing else.
    The parts concatenate back to the token they came from, so character offsets and
    `doc.text` are untouched -- this only ever inserts a boundary.

    Host and enclitic are pieces of ONE orthographic word, so the doc also publishes the
    multiword-token ranges, on the same contract `ar_tokenizer` uses:

        doc.user_data["mwt_ranges"] = [(first, last, surface), …]   1-based, inclusive

    That is what lets a CoNLL-U consumer write ITTB's `12-13 Animosque` range line back out.
    An empty list states "no MWTs here"; an absent key would mean "this tokeniser doesn't
    know", so the key is always set.
    """

    def __call__(self, text):
        doc = super().__call__(text)
        if not any(splits_enclitic(token.text) for token in doc):
            doc.user_data["mwt_ranges"] = []
            return doc
        words, spaces, ranges = [], [], []
        for token in doc:
            if splits_enclitic(token.text):
                ranges.append((len(words) + 1, len(words) + 2, token.text))
                words.append(token.text[:-len(SUFFIX)])
                spaces.append(False)
                words.append(token.text[-len(SUFFIX):])
            else:
                words.append(token.text)
            spaces.append(bool(token.whitespace_))
        out = Doc(self.vocab, words=words, spaces=spaces)
        out.user_data["mwt_ranges"] = ranges
        return out


@registry.tokenizers("sud.LatinEncliticTokenizer.v1")
def make_la_enclitic_tokenizer():
    def create(nlp):
        return LatinEncliticTokenizer(
            nlp.vocab,
            rules=LatinDefaults.tokenizer_exceptions,
            prefix_search=compile_prefix_regex(LatinDefaults.prefixes).search,
            suffix_search=compile_suffix_regex(LatinDefaults.suffixes).search,
            infix_finditer=compile_infix_regex(LatinDefaults.infixes).finditer,
            token_match=LatinDefaults.token_match,
            url_match=LatinDefaults.url_match,
        )
    return create
