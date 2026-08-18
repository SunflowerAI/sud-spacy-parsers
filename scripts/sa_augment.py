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


# ---------------------------------------------------------------- word order

def _runs(ref):
    """Sentence-by-sentence, the maximal PUNCTUATION-FREE stretches of `ref`, as index lists.

    A daṇḍa is a PAUSE: words do not cross it, and external sandhi does not apply over it (which is
    exactly how `sa_tokenizer._punct_kind` already models it). So it doubles as the boundary of what
    the permutation may touch, and the marks themselves never move.
    """
    out = []
    for sent in ref.sents:
        run = []
        for t in sent:
            if t.pos_ == "PUNCT":
                if run:
                    out.append(run)
                run = []
            else:
                run.append(t.i)
        if run:
            out.append(run)
    return out


def order_example(nlp: Language, example: Example, rng: random.Random) -> Example:
    """Re-linearise each clause under the constrained permutation, regenerating sandhi.

    The TREE is untouched: heads are re-indexed through the permutation, so every arc still joins
    the same two words. NORM is untouched too — the padapāṭha is a property of the word, not of its
    position, which is why the analyser channel and the lookup key survive permutation for free.
    Only ORTH changes, because the sandhi at each junction does.

    ⚠ SANDHI IS REGENERATED FROM NORM, NOT FROM ORTH. The corpus's ORTH is already sandhied for the
    order it is in; re-joining those surfaces would apply sandhi twice. The padapāṭha in NORM is the
    pausa form the engine expects as input.

    ⚠ THE UNIT OF PERMUTATION IS THE CLAUSE, NOT THE EXAMPLE. Under `gold_preproc` an example was
    one sentence and permuting the lot was the same thing; under `sud.GoldTokNormCorpus.v1` an
    example is ELEVEN sentences, and permuting across them would interleave sentences, strand the
    daṇḍas mid-clause and leave the SENT_START array describing boundaries that had moved. Runs stay
    inside their sentence and inside their clause, so every sentence span — and therefore the whole
    SENT_START array — is preserved by construction, and the marks stay where the editor put them.

    ⚠ SANDHI IS REGENERATED PER RUN, never across a mark, for the same reason the tokeniser treats a
    daṇḍa as opaque: it is a pause, and pausa forms are what stand on either side of it.
    """
    from sa_order import reorder
    import external_sandhi as ES
    from apply_vedic_sandhi import generate

    ref = example.reference
    n = len(ref)
    heads_all = [t.head.i for t in ref]

    # ---- permute inside each run, leaving everything else where it is ----------------------------
    order = list(range(n))
    moved = set()                            # start index of every run the permutation touched
    for run in _runs(ref):
        if len(run) < 4:
            continue
        pos = {g: k for k, g in enumerate(run)}
        toks = [{"lemma": ref[g].lemma_, "norm": ref[g].norm_, "upos": ref[g].pos_,
                 "compound": bool(ref[g].morph.get("Compound"))} for g in run]
        # heads INDUCED on the run: an arc leaving the run makes its dependent a local root, which
        # `reorder` handles — `_children` returns every self-headed token, not just the first.
        heads = [pos.get(heads_all[g], k) for k, g in enumerate(run)]
        local = reorder(toks, heads, rng)
        if local is None:
            continue
        for k, g in enumerate(run):
            order[g] = run[local[k]]
        moved.add(run[0])
    if not moved:
        return example

    # ---- regenerate the surface, run by run ------------------------------------------------------
    words = [ref[i].norm_ for i in order]
    feats = [str(ref[i].morph) for i in order]
    internal = [bool(ref[i].morph.get("Compound")) for i in order]
    forms, spaces = [None] * n, [True] * n
    k = 0
    while k < n:
        if ref[order[k]].pos_ == "PUNCT":       # a mark passes through untouched
            forms[k] = ref[order[k]].text
            k += 1
            continue
        j = k
        while j < n and ref[order[j]].pos_ != "PUNCT":
            j += 1
        if k not in moved:
            # ⚠ Regenerate ONLY what moved. An untouched run keeps the corpus's attested surface:
            # the engine is 91.8 % sentence-exact, so re-deriving a clause nothing happened to would
            # corrupt roughly one in twelve of them for nothing. (Runs never cross a sentence and the
            # permutation never leaves its run, so an untouched run's tokens are still in place.)
            for m in range(k, j):
                forms[m], spaces[m] = ref[order[m]].text, bool(ref[order[m]].whitespace_)
            k = j
            continue
        try:
            pieces = generate(words[k:j], feats[k:j], internal[k:j])
        except Exception:
            return example                      # a junction the engine cannot form: leave it alone
        if len(pieces) != j - k:
            return example
        for m, piece in enumerate(pieces):
            t = piece
            for mk in ES.COALESCE_MARKS:
                t = t.replace(mk, ES.COALESCE_SURFACE[mk])
            t = t.replace("'", "").replace("\u2019", "")
            joined = t.endswith("-")
            forms[k + m] = t[:-1] if joined else t
            # ⚠ A word can be ABSORBED ENTIRELY by coalescence, leaving an empty piece (`Doc` rejects
            # it with E031). That is a genuine orthographic fusion — the corpus represents such cases
            # as multiword tokens — and reproducing it here would break the 1:1 token mapping the
            # whole permutation rests on. Reject the permutation instead of inventing an MWT.
            spaces[k + m] = not joined and (k + m) < n - 1
        k = j
    if any(not f for f in forms):
        return example

    where = {old: new for new, old in enumerate(order)}
    data = example.to_dict()
    ta = data["token_annotation"]
    for key in ("LEMMA", "POS", "TAG", "MORPH", "DEP"):
        ta[key] = [ta[key][i] for i in order]
    ta["HEAD"] = [where[ta["HEAD"][i]] for i in order]
    ta["ORTH"] = forms
    ta["SPACY"] = spaces
    # SENT_START is NOT permuted: runs never cross a sentence, so every boundary is where it was.
    ents = data["doc_annotation"]["entities"]
    data["doc_annotation"]["entities"] = [ents[i] for i in order]

    predicted = Doc(nlp.vocab, words=forms, spaces=spaces)
    for new, old in enumerate(order):        # NORM is order-independent; carry it across
        predicted[new].norm_ = ref[old].norm_
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.sa_order_variants.v1")
def create_sa_order_augmenter(p_order: float = 0.5, p_sent: float = 0.0,
                              p_any: float = 0.0, seed: int = 0):
    """Constrained word-order permutation, optionally with the case variants on top.

    `p_order` is the chance a given example is re-linearised at all; the rest pass through in their
    attested order, so the model never stops seeing real Sanskrit.
    """
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example):
        eg = example
        if rng.random() < p_order:
            eg = order_example(nlp, eg, rng)
        if p_sent or p_any:
            eg = case_example(nlp, eg, rng, p_sent, p_any)
        yield eg

    return augmenter
