#!/usr/bin/env python3
"""`sa_deva` — render the output in Devanagari when the INPUT was Devanagari.

The model works entirely in IAST: the tokeniser romanises Devanagari on the way in, and every
component's embed reads `NORM`, `PREFIX`, `SUFFIX` and `SHAPE`, the last three derived from the
token's orth. Putting Devanagari text on the tokens before the parser runs would therefore hand it
a lexeme shape it has never seen — so this component runs **LAST**, after everything has been
predicted, and rebuilds the doc with:

    FORM   -> Devanagari,  with `Token._.translit`  holding the IAST form   (UD's `Translit`)
    LEMMA  -> Devanagari,  with `Token._.ltranslit` holding the IAST lemma  (UD's `LTranslit`)

which is the Universal Dependencies convention for a non-Latin script: the native script in
FORM/LEMMA, the romanisation in MISC. It is a no-op when the input was already Latin, so IAST users
see IAST and nothing is transliterated that should not be.

Everything predicted upstream is carried across the rebuild — tag, POS, morph, lemma, head, deprel,
sentence starts, `_.unsandhied`, `_.compound_flags` and the source spans. `clause_parser` had to
learn the same lesson (it once dropped lemma and morph when it rebuilt the doc); the rule is that a
component which rebuilds a `Doc` owns carrying EVERY annotation, not the ones it happens to think of.

Ordering: after `clause_parser`, which also rebuilds the doc and would otherwise undo this.
"""
from spacy.language import Language
from spacy.tokens import Doc, Token

_DEVA_PUNCT = {"|": "।", "‖": "॥"}      # `desandhi_csl` normalises the daṇḍas to | and ‖


for _name in ("translit", "ltranslit"):
    if not Token.has_extension(_name):
        Token.set_extension(_name, default="")


def _has_devanagari(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def to_devanagari(s):
    """IAST -> Devanagari, with the daṇḍas mapped back to their native characters."""
    if not s:
        return s
    if s in _DEVA_PUNCT:
        return _DEVA_PUNCT[s]
    if not any(c.isalpha() for c in s):
        return s                                   # pure punctuation: leave alone
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    return transliterate(s, sanscript.IAST, sanscript.DEVANAGARI)


@Language.factory("sa_deva")
def make_sa_deva(nlp, name):
    return SaDevanagari()


class SaDevanagari:
    def __call__(self, doc):
        src = doc._.src_text if Doc.has_extension("src_text") else None
        if not src or not _has_devanagari(src):
            return doc                              # Latin input: nothing to do
        words = [to_devanagari(t.text) for t in doc]
        lemmas = [to_devanagari(t.lemma_) for t in doc]
        out = Doc(
            doc.vocab,
            words=words,
            spaces=[bool(t.whitespace_) for t in doc],
            tags=[t.tag_ for t in doc],
            pos=[t.pos_ for t in doc],
            morphs=[str(t.morph) for t in doc],
            lemmas=lemmas,
            heads=[t.head.i for t in doc],
            deps=[t.dep_ for t in doc],
            sent_starts=[bool(t.is_sent_start) for t in doc],
        )
        for old, new in zip(doc, out):
            new._.translit = old.text               # UD Translit  — the romanised FORM
            new._.ltranslit = old.lemma_            # UD LTranslit — the romanised LEMMA
            if Token.has_extension("unsandhied"):
                new._.unsandhied = old._.unsandhied
        if Doc.has_extension("src_text"):
            out._.src_text = src
        if Doc.has_extension("src_spans"):
            out._.src_spans = doc._.src_spans       # per token, and the count is unchanged
        if Doc.has_extension("compound_flags"):
            out._.compound_flags = doc._.compound_flags
        return out
