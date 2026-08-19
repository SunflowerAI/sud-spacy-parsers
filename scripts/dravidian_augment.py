#!/usr/bin/env python3
"""Train-time word-order augmentation for Tamil and Telugu: one copy, re-linearised every epoch.

Registers ``sud.dravidian_order_variants.v1``. The transform itself, and the measurements that
license each of its constraints, are in ``scripts/dravidian_order.py`` — read that first. In one
line: Dravidian is rigidly head-final and the parser SHOULD learn that, so the side of the head is
read off the data and never assigned; what gets shuffled is the order of siblings in the preverbal
field, which the treebanks show is genuinely free (26 % OSV in Tamil, 23 % in Telugu).

    [corpora.train]
    @readers = "sud.GoldTokCorpus.v1"
    shuffle = true
    [corpora.train.augmenter]
    @augmenters = "sud.dravidian_order_variants.v1"
    lang = "ta"

⚠ **``max_epochs`` must be ``-1``, and that has two consequences.** With ``0`` spaCy's
``create_train_batches`` does ``examples = list(corpus(nlp))`` ONCE and reshuffles that same list
every epoch, so a corpus-level augmenter samples a single linearisation per document for the whole
run — the run looks entirely normal and trains on one fixed permutation. ``-1`` streams instead,
but then the training loop stops shuffling, so the reader needs ``shuffle = true``. Under
``sud.GoldTokCorpus.v1`` a document IS an example, so that is the same shuffle by another name.
This is `docs/latin.md`'s hazard reached by a different route, and it is silent in both.

⚠ **THE LABEL SETS MOVE, unlike Latin's orthographic case.** There the parser's labels were
properties of the TREES and only the lemmatiser's edit trees were properties of the FORMS. Word
order is different: a non-projective gold tree is pseudo-projectivised, and the lifted arc picks up
a ``||`` suffix naming what it was lifted over — so the PARSER's label set is a property of the
ORDER. Collect it with ``scripts/init_aug_labels.py`` over several passes, exactly as
``train_la_order.sh`` does, and read the coverage it prints. This matters more here than for Latin
because these treebanks are small enough that a label seen in one linearisation and not another is
an ordinary event rather than a rare one.

Nothing here ships in a wheel: an augmenter is a training-time reader hook, and the model that
comes out is an ordinary spaCy pipeline.
"""
from __future__ import annotations

import pathlib
import random
import sys
from typing import Callable, Iterator

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import registry

# `spacy train --code` loads this file by path, so scripts/ is not necessarily importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dravidian_order import OrderPolicy, POLICIES, Tok, reorder_doc  # noqa: E402


def sentence_starts(doc: Doc) -> list[int]:
    """Indices that open a sentence. A doc with no boundary annotation is one sentence."""
    starts = [i for i, tok in enumerate(doc) if tok.is_sent_start]
    return starts or [0]


def reorder_example(nlp: Language, example: Example, rng: random.Random,
                    policy: OrderPolicy) -> Example:
    """Re-linearise each sentence of an example. The TREE is untouched — heads are re-indexed
    through the permutation, so every arc still joins the same two words.

    ``SENT_START`` is deliberately NOT permuted: a permutation inside a sentence leaves every
    sentence occupying the positions it already did, so the boundary list is a property of the
    POSITIONS and is already correct.
    """
    ref = example.reference
    data = example.to_dict()
    ents = data["doc_annotation"]["entities"]
    if any(tag not in ("O", "-") for tag in ents):
        # A BILUO span is a fact about ADJACENT tokens, so permuting it silently invents entities.
        # No SUD treebank carries one, which is why this refuses rather than trying to cope.
        return example

    toks = [Tok(form=t.text, lemma=t.lemma_, upos=t.pos_, deprel=t.dep_,
                head=t.head.i if t.head.i != t.i else -1,
                feats=str(t.morph), space_after=bool(t.whitespace_))
            for t in ref]
    r = reorder_doc(toks, sentence_starts(ref), rng, policy)
    if r.order == list(range(len(toks))):
        return example

    where = {old: new for new, old in enumerate(r.order)}
    ta = data["token_annotation"]
    for key in ("ORTH", "LEMMA", "POS", "TAG", "MORPH", "DEP"):
        ta[key] = [ta[key][i] for i in r.order]
    ta["HEAD"] = [where[ta["HEAD"][i]] for i in r.order]
    ta["SPACY"] = r.spaces
    data["doc_annotation"]["entities"] = [ents[i] for i in r.order]

    predicted = Doc(nlp.vocab, words=ta["ORTH"], spaces=r.spaces)
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.dravidian_order_variants.v1")
def create_dravidian_order_augmenter(
    lang: str = "ta",
    p_sentence: float = 0.5,
    p_hyperbaton: float = -1.0,
    min_len: int = 3,
    clause_only: bool = True,
    keep_original: bool = False,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """``p_hyperbaton = -1`` means "take this language's calibrated default" — 0.08 for Tamil
    (18.0 % of its training sentences carry a crossing arc) and 0.0 for Telugu (0.1 %, so any
    displacement at all would be inventing a construction the language does not have).

    ``keep_original`` additionally yields the untouched example, which doubles the epoch and makes
    the arm a superset rather than a replacement — off by default, since the point is to stop
    paying for copies.
    """
    base = POLICIES.get(lang, OrderPolicy())
    policy = OrderPolicy(
        p_sentence=p_sentence,
        p_hyperbaton=base.p_hyperbaton if p_hyperbaton < 0 else p_hyperbaton,
        min_len=min_len,
        clause_only=clause_only,
    )
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if keep_original:
            yield example
        yield reorder_example(nlp, example, rng, policy)

    return augmenter
