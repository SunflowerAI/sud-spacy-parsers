#!/usr/bin/env python3
"""Train-time orthographic augmentation for Latin: one copy of the data, resampled every epoch.

The released Latin arm is robust to macrons because it is trained on the UNION of the plain and the
macronised treebank -- two literal copies of the same 586 604 tokens (``train_la_ext_macron.sh``).
That buys exactly the two spellings it contains, at twice the data, and says nothing about the
other axes on which printed Latin varies: breves, ``j``/``v``, ``æ``/``œ``, and whether a sentence
opens with a capital.

This replaces the copies with sampling. Every epoch, each document is assigned an EDITION STYLE
(see ``la_orth.Style``) and rewritten into it, so over a run the model sees the same sentence under
many orthographies instead of two. The transforms and what licenses each of them are documented in
``scripts/la_orth.py``.

    [corpora.train]
    @readers = "sud.GoldTokCorpus.v1"
    shuffle = true
    [corpora.train.augmenter]
    @augmenters = "sud.la_orth_variants.v1"

⚠ **``max_epochs`` must be ``-1``.** With ``0`` (the project's usual setting) spaCy's
``create_train_batches`` does ``examples = list(corpus(nlp))`` ONCE and reshuffles that same list
for every epoch, so a corpus-level augmenter samples a single style per document for the whole run
and the resampling silently never happens -- the run looks normal and trains on one fixed
perturbation. ``-1`` streams the corpus instead, re-reading and re-augmenting each epoch; it also
turns off the training loop's own shuffling, which is why the reader must be given
``shuffle = true`` to shuffle at the document level itself. For an arm trained through
``sud.GoldTokCorpus.v1`` a document IS an example, so that shuffle is exactly the one the loop was
doing before.

Nothing here ships in a wheel: the augmenter is a training-time reader hook, and the model that
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
from la_order import OrderPolicy, Tok, reorder_doc  # noqa: E402
from la_orth import OrthPolicy, Style, sample_style, set_initial_case, vary_word  # noqa: E402


def sentence_starts(doc: Doc) -> list[int]:
    """Indices that open a sentence. A doc with no boundary annotation is one sentence."""
    starts = [i for i, tok in enumerate(doc) if tok.is_sent_start]
    return starts or [0]


def vary_example(nlp: Language, example: Example, style: Style, rng: random.Random,
                 policy: OrthPolicy) -> Example:
    """Rewrite an example's word forms into ``style``, keeping every annotation on its token.

    Only ORTH changes. LEMMA is deliberately left canonical -- it is the lemmatiser's target, and
    the point of the exercise is that a form spelled four ways still reaches one lemma.
    """
    ref = example.reference
    words = [vary_word(tok.text, style, rng, tok.lemma_) for tok in ref]
    for i in sentence_starts(ref):
        if not style.capitalise and policy.protect_propn and ref[i].pos_ == "PROPN":
            continue                     # a name keeps its capital in either convention
        words[i] = set_initial_case(words[i], style.capitalise)
    if words == [tok.text for tok in ref]:
        return example
    data = example.to_dict()
    data["token_annotation"]["ORTH"] = words
    predicted = Doc(nlp.vocab, words=words, spaces=[bool(tok.whitespace_) for tok in ref])
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.la_orth_variants.v1")
def create_la_orth_augmenter(
    p_v: float = 0.5,
    p_j: float = 0.5,
    p_lig: float = 0.25,
    p_capital: float = 0.5,
    p_length: float = 0.5,
    p_uniform_length: float = 0.5,
    p_breve_doc: float = 0.3,
    max_breve_rate: float = 0.5,
    protect_propn: bool = True,
    keep_original: bool = False,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """See ``la_orth.OrthPolicy`` for what each rate means.

    ``keep_original`` additionally yields the untouched example, which doubles the epoch and makes
    the arm a superset of the old union rather than a replacement -- off by default, since the
    whole point is to stop paying for copies.
    """
    policy = OrthPolicy(p_v=p_v, p_j=p_j, p_lig=p_lig, p_capital=p_capital, p_length=p_length,
                        p_uniform_length=p_uniform_length, p_breve_doc=p_breve_doc,
                        max_breve_rate=max_breve_rate, protect_propn=protect_propn)
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if keep_original:
            yield example
        yield vary_example(nlp, example, sample_style(rng, policy), rng, policy)

    return augmenter


# ---------------------------------------------------------------- word order

def reorder_example(nlp: Language, example: Example, rng: random.Random,
                    policy: OrderPolicy) -> Example:
    """Re-linearise each sentence of an example. The TREE is untouched — heads are re-indexed
    through the permutation, so every arc still joins the same two words.

    ``SENT_START`` is deliberately NOT permuted: a permutation inside a sentence leaves every
    sentence occupying the positions it already did, so the boundary list is a property of the
    positions and is already correct.
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
    for key in ("LEMMA", "POS", "TAG", "MORPH", "DEP"):
        ta[key] = [ta[key][i] for i in r.order]
    ta["HEAD"] = [where[ta["HEAD"][i]] for i in r.order]
    ta["ORTH"] = r.forms
    ta["SPACY"] = r.spaces
    data["doc_annotation"]["entities"] = [ents[i] for i in r.order]

    predicted = Doc(nlp.vocab, words=r.forms, spaces=r.spaces)
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.la_variants.v1")
def create_la_variants_augmenter(
    p_sentence: float = 0.5,
    p_hyperbaton: float = 0.08,
    p_rise: float = 0.4,
    min_len: int = 3,
    p_v: float = 0.5,
    p_j: float = 0.5,
    p_lig: float = 0.25,
    p_capital: float = 0.5,
    p_length: float = 0.5,
    p_uniform_length: float = 0.5,
    p_breve_doc: float = 0.3,
    max_breve_rate: float = 0.5,
    protect_propn: bool = True,
    keep_original: bool = False,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """Orthography AND word order: ``sud.la_orth_variants.v1`` with a re-linearisation in front.

    ORDER FIRST, and that is not arbitrary. The orthographic pass decides whether the sentence
    opens with a capital and applies it to whatever token is FIRST; run the other way round it
    would capitalise a word that the shuffle is about to bury mid-sentence and leave the new
    opening lowercase in a style that capitalises. See ``scripts/la_order.py`` for what the
    linearisation preserves and ``scripts/calibrate_la_order.py`` for how ``p_hyperbaton`` was set.
    """
    orth = OrthPolicy(p_v=p_v, p_j=p_j, p_lig=p_lig, p_capital=p_capital, p_length=p_length,
                      p_uniform_length=p_uniform_length, p_breve_doc=p_breve_doc,
                      max_breve_rate=max_breve_rate, protect_propn=protect_propn)
    order = OrderPolicy(p_sentence=p_sentence, p_hyperbaton=p_hyperbaton, p_rise=p_rise,
                        min_len=min_len)
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if keep_original:
            yield example
        moved = reorder_example(nlp, example, rng, order)
        yield vary_example(nlp, moved, sample_style(rng, orth), rng, orth)

    return augmenter
