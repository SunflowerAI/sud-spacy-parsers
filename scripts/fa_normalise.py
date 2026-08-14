#!/usr/bin/env python3
"""``sud.FaNormTokenizer.v1`` -- normalise Persian input orthography at the tokeniser boundary.

The same move zh makes (`ZhTradTokenizer` converts simplified in, `zh_script` converts back out)
rather than training on every spelling: where a variation is REVERSIBLE and carries no information,
normalising the input is exact, free, and needs no retraining.

Two axes, both measured worth having on the released arm before any augmentation existed:

    Persian input                 released raw   + this normaliser
    Arabic letterforms ي/ك        57.55 LAS         86.92 LAS
    fully vocalised               33.28 LAS         86.92 LAS

and it composes with the augmented arm, which is what ships: 86.09 -> **87.02** on letterforms,
85.64 -> **86.75** on all axes at once.

WHAT IS NORMALISED, AND WHY EXACTLY THESE TWO.

  * `ي` -> `ی` and `ك` -> `ک`. These are the ARABIC codepoints for yeh and kaf, which is what an
    Arabic keyboard produces; Persian orthography uses `ی`/`ک`. The mapping is 1:1, length
    preserving, and carries no linguistic content whatever -- a pure encoding artefact, and a large
    share of real Persian text on the web.
  * Diacritics are STRIPPED. Persian has no case system, so its short vowels carry no syntactic
    information, and the parser is measurably better without them (+0.63 LAS). ⚠ This is exactly
    the opposite of the right answer for ARABIC, where the final short vowel IS the case ending:
    stripping before the augmented Arabic arm costs 0.77 LAS and 4.42 TAG. Do not copy this module
    to ar.

WHAT IS NOT NORMALISED. The ZWNJ. Dropping one is IRREVERSIBLE -- `میرود` gives no way to know
where the joiner was without a lexicon -- so there is nothing to normalise TOWARDS, and that axis
is the one the augmentation genuinely earns (normalisation alone reaches 82.65 on the all-axes row
against the augmented arm's 85.64).

⚠ `doc.text` IS THE NORMALISED TEXT, not the caller's input. That is a deliberate contract, and the
opposite of the one `ar_vocalise` holds -- a vocaliser ADDS to what it was given and must give it
back unchanged, whereas a tokeniser's job here is to hand the model the spelling it was trained on.
The caller's original is never discarded: it is kept on ``doc.user_data["fa_source_text"]``, with
``doc.user_data["fa_normalised"]`` recording whether anything actually changed.
"""
from spacy.tokenizer import Tokenizer
from spacy.util import (compile_infix_regex, compile_prefix_regex, compile_suffix_regex,
                        registry)

#: Arabic codepoints an Arabic keyboard produces, and their Persian equivalents. 1:1, so the
#: character offsets of everything else are untouched.
LETTERFORMS = {"ي": "ی",   # ARABIC YEH        -> FARSI YEH
               "ك": "ک",   # ARABIC KAF        -> KEHEH
               "ى": "ی"}   # ALEF MAKSURA      -> FARSI YEH
#: The combining marks Persian leaves unwritten in ordinary prose.
DIACRITICS = set("ًٌٍَُِّْٰ")


def normalise(text):
    """Return the normalised text. Only ever maps 1:1 or DELETES, never inserts, so a token's
    characters are always a subsequence of the caller's."""
    out = []
    for ch in text:
        if ch in DIACRITICS:
            continue
        out.append(LETTERFORMS.get(ch, ch))
    return "".join(out)


class FaNormTokenizer(Tokenizer):
    """spaCy's rule tokeniser with the input normalised first."""

    def __call__(self, text):
        norm = normalise(text)
        doc = super().__call__(norm)
        doc.user_data["fa_source_text"] = text
        doc.user_data["fa_normalised"] = norm != text
        return doc


@registry.tokenizers("sud.FaNormTokenizer.v1")
def make_fa_norm_tokenizer():
    def tokenizer(nlp):
        # Built from the language's OWN defaults, so the rules, prefixes, suffixes and infixes are
        # exactly the released tokeniser's and the only difference is the input string. Rebuilding
        # them by hand here would be a second, silently diverging copy of spaCy's fa rules.
        d = nlp.Defaults
        return FaNormTokenizer(
            nlp.vocab,
            rules=d.tokenizer_exceptions,
            prefix_search=compile_prefix_regex(d.prefixes).search if d.prefixes else None,
            suffix_search=compile_suffix_regex(d.suffixes).search if d.suffixes else None,
            infix_finditer=compile_infix_regex(d.infixes).finditer if d.infixes else None,
            token_match=d.token_match,
            url_match=d.url_match,
        )
    return tokenizer
