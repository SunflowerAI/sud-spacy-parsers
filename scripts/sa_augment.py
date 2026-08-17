#!/usr/bin/env python3
"""Train-time CASE augmentation for Sanskrit: teach the model that a capital carries no syntax.

WHY. The corpus is lower-case throughout (0 capitals in 163 802 Vedic tokens; DCS likewise), but the
wheel takes RAW IAST typed by a reader, who writes `Rāmaḥ` and capitalises a sentence opening.
`sa_tokenizer.normalise` case-folds so the CSLiser (41-character inventory, no capitals) and the
analyser table (126 809 lower-case keys) never meet one, and `__call__` then RESTORES the user's
case in ORTH so the wheel does not silently rewrite its input. That restoration has a measured cost:

    sentence-initial capital, 6 416 tokens compared -> 14.62 % get a DIFFERENT analysis
    (head, deprel or UPOS) than the same sentence lower-cased

because PREFIX, SUFFIX and SHAPE are lexeme attributes read off the ORTH. Folding protects NORM;
only augmentation protects those three. Driving that 14.62 % towards zero is this module's whole job.

NOT PROPN-TARGETED, DELIBERATELY. The obvious design capitalises proper nouns, and it cannot work
here: PROPN is 94 tokens in the Vedic+UFAL half and 67 in DCS's 1 896 183 (0.004 %), and all 56
distinct DCS PROPN lemmas are Pañcatantra characters -- one text's annotator being diligent, not a
labelled category. Nor can the gap be filled by detection: absence from vidyut's kosha, the most
promising heuristic, gives 11.7 % recall at 2.2 % noise, which at that base rate makes essentially
every flag a false positive. So instead of finding names, `p_any` capitalises ANY token at a low
rate, which teaches case-invariance generally and therefore covers a capitalised name at inference
without ever having to identify one. (Devanagari has no case distinction at all, so none of this
applies to that input path.)

⚠ NORM MUST NOT BE CAPITALISED. It is the analyser table's key and the channel the model trained on,
and `Example.to_dict()` does NOT carry it (verified: its token_annotation has ORTH/SPACY/TAG/LEMMA/
POS/MORPH/HEAD/DEP/SENT_START and no NORM), so a rebuilt Doc silently loses it -- NORM differs from
ORTH on 30 % of tokens here (`panthāna` -> `panthānaḥ`), so the loss is not cosmetic. This copies it
across explicitly, unchanged. That is exactly the inference-time regime: ORTH cased, NORM folded.

⚠ LEMMA STAYS CANONICAL, which is a feature: a capitalised form still has to reach its lower-case
lemma, so the lemmatiser is taught to strip the capital rather than to preserve it.

    [corpora.train]
    @readers = "sud.NormCorpus.v1"
    shuffle = true
    [corpora.train.augmenter]
    @augmenters = "sud.sa_case_variants.v1"

⚠ `max_epochs` MUST BE `-1` (standing hazard 9). With `0` spaCy lists the corpus ONCE and reshuffles
that same list every epoch, so one capitalisation pattern is sampled per document for the whole run
and the augmentation silently never varies -- the logs look completely normal. `-1` streams the
corpus and re-augments each epoch, and it also stops the loop shuffling, which is why the reader
needs `shuffle = true`.

Nothing here ships in a wheel: an augmenter is a training-time reader hook.
"""
from __future__ import annotations

import random
from typing import Callable, Iterator

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import registry


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:] if word else word


def case_example(nlp: Language, example: Example, rng: random.Random,
                 p_sent: float, p_any: float) -> Example:
    """Capitalise sentence-initial tokens at `p_sent` and any token at `p_any`. ORTH only."""
    ref = example.reference
    words = [t.text for t in ref]
    starts = {0} | {t.i for t in ref if t.is_sent_start}
    for i, t in enumerate(ref):
        if not t.text[:1].isalpha() or t.text[:1].isupper():
            continue
        p = p_sent if i in starts else p_any
        if rng.random() < p:
            words[i] = _cap(words[i])
    if words == [t.text for t in ref]:
        return example
    data = example.to_dict()
    data["token_annotation"]["ORTH"] = words
    predicted = Doc(nlp.vocab, words=words, spaces=[bool(t.whitespace_) for t in ref])
    # Carry NORM from the doc the READER built. Rebuilding drops it, and it is the analyser's key.
    for new, old in zip(predicted, example.predicted):
        new.norm_ = old.norm_
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.sa_case_variants.v1")
def create_sa_case_augmenter(
    p_sent: float = 0.5,
    p_any: float = 0.05,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """`p_sent` is the headline case (21 707 sentence-initial tokens, 13 % of the corpus).
    `p_any` stands in for proper nouns, which cannot be identified here -- see the module docstring.
    """
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        yield case_example(nlp, example, rng, p_sent, p_any)

    return augmenter
