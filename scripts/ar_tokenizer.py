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

Because those tokens are clitic *pieces* of one orthographic word, the doc also publishes the
multi-word-token ranges on `doc.user_data`, which spaCy's Doc has no column for:

    doc.user_data["mwt_ranges"]  = [(first, last, surface), …]   1-based, inclusive
    doc.user_data["source_text"] = the raw input

A consumer that wants CoNLL-U `n-m` range lines (SUD Workbench does) can then use them verbatim
instead of guessing the grouping back out of a flat token list — the only signals left to guess
from are spacing and the tagger's PUNCT labels, and the latter is a *prediction*, so a mis-tagged
mark inside a chunk silently invents or destroys a range. Empty list = "no MWTs here", which is a
statement, not a missing value; absent key = "this tokeniser doesn't know", so infer.

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
        words, spaces, mwt = [], [], []
        for chunk in re.findall(r"\S+", text):       # whitespace-delimited chunks
            subs = []
            for w in simple(chunk):                   # split off punctuation
                parts = _peel(w, tok)
                if len(parts) > 1:                    # one orthographic word → ≥2 tokens: an MWT
                    a = len(words) + len(subs) + 1    # 1-based id of this word's first component
                    mwt.append((a, a + len(parts) - 1, w))
                subs += parts
            if not subs:
                subs = [chunk]
            for j, s in enumerate(subs):
                words.append(s)
                spaces.append(j == len(subs) - 1)     # space only after a chunk's last token
        if spaces:
            spaces[-1] = False
        if not words and text:                        # all-whitespace input → keep it as one token
            words, spaces = [text], [False]           # (guard `text`: Doc rejects "" with E031)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        # Publish the grouping instead of leaving consumers to re-infer it. The peel loop above is
        # the ONLY place that knows which tokens came from one orthographic word; downstream all a
        # consumer sees is a flat token list, so its only recourse is to guess from spacing plus
        # the tagger's PUNCT labels — which mis-groups a chunk as soon as a mark inside it is
        # tagged as anything else. The surface is `w`, NOT "".join(parts): they agree today only
        # because `_peel` slices the original string, and carrying `w` is what keeps this correct
        # under a scheme that normalises (atbtok's own marks restore the alif that لل elides, so a
        # concatenation of CAMeL's output would read لالمدرسة for surface للمدرسة).
        doc.user_data["mwt_ranges"] = mwt             # [(first, last, surface), …] 1-based inclusive
        doc.user_data["source_text"] = text           # doc.text collapses whitespace runs (the
        return doc                                    # re.findall above drops them) — this doesn't

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
