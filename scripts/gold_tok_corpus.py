#!/usr/bin/env python3
"""Custom spaCy corpus reader: multi-sentence docs with GOLD tokenisation.

Standard spaCy training forces an either/or that this project kept running into:

  * ``gold_preproc = true``  — the predicted doc uses the treebank's gold tokens (so a
    tokeniser that can't reproduce the treebank, e.g. zh/yue pkuseg at ~0.88–0.95 word-F1,
    doesn't corrupt training), BUT every doc is split back into **single sentences**, so the
    parser is never shown a sentence boundary and never learns to segment running text. Models
    trained this way collapse multi-sentence input into one tree.
  * ``gold_preproc = false`` — the corpus yields whole **multi-sentence** docs, so the parser
    learns sentence boundaries, BUT the predicted doc is re-tokenised with the model tokeniser
    (``nlp.make_doc``), so a treebank whose tokens the tokeniser can't reproduce trains on
    misaligned tokens.

This reader gives both at once: it yields one Example per (multi-sentence) reference doc — the
corpus is converted at ~10 sentences/doc (``spacy convert -n 10``) with ``SENT_START`` set — and
builds the *predicted* doc from the reference's own gold words/spaces. So the parser sees the
sentence boundaries it must learn, with no tokenisation skew. (A doc longer than ``max_length`` is
split at sentence boundaries, as in the stock reader; those rare pieces lose the in-doc boundary
signal, so keep ``max_length = 0`` unless memory forces otherwise.)

Train-time only — wire it in via ``[corpora.*] @readers = "sud.GoldTokCorpus.v1"`` and pass
``--code scripts/gold_tok_corpus.py``. It ships nothing into the packaged model.
"""
from spacy.tokens import Doc
from spacy.training.corpus import Corpus
from spacy.training.example import Example
from spacy.util import registry


@registry.readers("sud.GoldTokCorpus.v1")
def create_gold_tok_reader(path, max_length: int = 0, limit: int = 0, augmenter=None):
    return GoldTokCorpus(path, max_length=max_length, limit=limit, augmenter=augmenter)


class GoldTokCorpus(Corpus):
    """Like ``spacy.Corpus`` with ``gold_preproc=False`` (whole docs, so segmentation is
    learned), but the predicted doc always carries the reference's gold tokenisation instead of
    being re-tokenised by the model tokeniser."""

    def __init__(self, path, *, limit: int = 0, max_length: int = 0,
                 augmenter=None, shuffle: bool = False):
        super().__init__(path, limit=limit, gold_preproc=False, max_length=max_length,
                         augmenter=augmenter, shuffle=shuffle)

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        # always gold tokens for the predicted doc (ignore gold_preproc / make_doc),
        # so multi-sentence docs train the parser to segment without any tokeniser skew.
        return Example(
            Doc(nlp.vocab,
                words=[w.text for w in reference],
                spaces=[bool(w.whitespace_) for w in reference]),
            reference,
        )
