#!/usr/bin/env python3
"""Raw-text Arabic tokeniser that reproduces SUD_Arabic-PADT tokenisation, so `ar_sud_padt`
runs on raw text instead of needing gold tokens.

PADT splits proclitic conjunctions/prepositions (و، ف، ل، ب، ك، س) and enclitic pronouns into
separate tokens while keeping the definite article ال attached — i.e. the Penn-Arabic-Treebank
(ATB) tokenisation scheme. CAMeL Tools' MLE disambiguator + `atbtok` morphological tokeniser
produces exactly that segmentation. We use CAMeL only to locate the **clitic boundaries** and
then split the *original surface string* at those positions, so the emitted tokens preserve the
treebank's orthography exactly (no alef/hamza normalisation) — which keeps the text aligned for
scoring and matches what the parser was trained on. Raw end-to-end LAS 44.6 → 69.4 (gold-token
ceiling 78.4); token-F1 vs PADT 0.83 → 0.91.

Runtime dependency (like `spacy-pkuseg` for Chinese, mecab for Korean): install with
    pip install camel-tools
    camel_data -i morphology-db-msa-r13 disambig-mle-calima-msa-r13
The CAMeL data is GPL v2 and is NOT bundled in the model wheel; the parser stays CC BY-SA 4.0.
"""
import re

from spacy.tokens import Doc
from spacy.util import registry

_CAMEL = None  # lazily-built (MLEDisambiguator, MorphologicalTokenizer)


def _camel():
    global _CAMEL
    if _CAMEL is None:
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            from camel_tools.tokenizers.morphological import MorphologicalTokenizer
            from camel_tools.tokenizers.word import simple_word_tokenize
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ar_sud_padt needs CAMeL Tools for raw-text tokenisation. Install with:\n"
                "  pip install camel-tools\n"
                "  camel_data -i morphology-db-msa-r13 disambig-mle-calima-msa-r13"
            ) from e
        mle = MLEDisambiguator.pretrained("calima-msa-r13")
        tok = MorphologicalTokenizer(disambiguator=mle, scheme="atbtok", split=True, diac=False)
        _CAMEL = (tok, simple_word_tokenize)
    return _CAMEL


def _peel(surface, tok):
    """Split `surface` at the clitic boundaries CAMeL finds, keeping the original characters."""
    marks = tok.tokenize([surface])
    front, rem = [], surface
    for m in marks:                      # leading proclitics carry a trailing '+'
        if m.endswith("+") and 1 < len(m) and len(m) - 1 < len(rem):
            front.append(rem[: len(m) - 1]); rem = rem[len(m) - 1:]
        else:
            break
    back = []
    for m in reversed(marks):            # trailing enclitics carry a leading '+'
        if m.startswith("+") and 1 < len(m) and len(m) - 1 < len(rem):
            back.insert(0, rem[-(len(m) - 1):]); rem = rem[: -(len(m) - 1)]
        else:
            break
    return [t for t in front + [rem] + back if t]


@registry.tokenizers("ar.CamelAtbTokenizer.v1")
def make_camel_atb_tokenizer():
    def create(nlp):
        return CamelAtbTokenizer(nlp.vocab)
    return create


class CamelAtbTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        tok, simple = _camel()
        words, spaces = [], []
        for chunk in re.findall(r"\S+", text):       # whitespace-delimited chunks
            subs = []
            for w in simple(chunk):                   # split off punctuation
                subs += _peel(w, tok)
            if not subs:
                subs = [chunk]
            for j, s in enumerate(subs):
                words.append(s)
                spaces.append(j == len(subs) - 1)     # space only after a chunk's last token
        if spaces:
            spaces[-1] = False
        if not words:
            words, spaces = [text], [False]
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
