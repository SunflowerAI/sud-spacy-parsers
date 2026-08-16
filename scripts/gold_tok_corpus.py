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
def create_gold_tok_reader(path, max_length: int = 0, limit: int = 0, augmenter=None,
                           shuffle: bool = False):
    # `shuffle` matters only under `max_epochs = -1`, where spaCy streams the corpus instead of
    # materialising it once and shuffling in the loop -- which is what an augmenter that must
    # resample every epoch requires (see scripts/la_augment.py). One doc is one example here, so
    # shuffling docs is the same shuffle the training loop would otherwise be doing.
    return GoldTokCorpus(path, max_length=max_length, limit=limit, augmenter=augmenter,
                         shuffle=shuffle)


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


@registry.readers("sud.CompoundCorpus.v1")
def create_compound_reader(path, gold_preproc: bool = True, max_length: int = 0,
                           limit: int = 0, augmenter=None):
    return CompoundCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                          limit=limit, augmenter=augmenter)


class CompoundCorpus(Corpus):
    """Like ``spacy.Corpus``, but the PREDICTED doc carries the reference's ``Compound`` feature.

    For Sanskrit, ``Compound=Yes`` is not something a model has to guess: ``sa_tokenizer`` reads it
    straight off the CSL join marker (a hyphen/pipe binding a samāsa member to the next word) and
    stamps it on the doc, at precision/recall 0.9998 against the treebank. That makes it available
    as an INPUT feature — ``configs/config_sa*.cfg`` list ``MORPH`` among the embed attrs — which is
    worth having, since the join marker used to be visible in the token form itself and stripping it
    to clean wordforms cost ~0.5 LAS.

    But it is only usable if it is present at TRAINING time too, and it isn't: the stock reader
    builds the predicted doc directly from the reference's words and never runs the tokeniser, so a
    tokeniser-set feature is silently absent while training and present at inference — the model
    would learn to ignore a feature that then appears from nowhere. This reader closes that gap by
    copying the feature across from the reference.

    ONLY ``Compound`` is copied, never the rest of the gold FEATS: everything else is what the
    morphologizer is being asked to predict, and copying it would be leakage. ``Compound`` is not
    leakage precisely because the tokeniser supplies the identical value at inference.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        # With gold_preproc the predicted doc is token-for-token the reference, so the copy is
        # well defined. Without it the base class re-tokenises and the two need not align — but
        # that path also runs the real tokeniser, which sets the feature itself, so skip.
        if len(eg.predicted) == len(reference):
            for pred_tok, ref_tok in zip(eg.predicted, reference):
                if ref_tok.morph.get("Compound"):
                    pred_tok.set_morph("Compound=Yes")
        return eg

# ---------------------------------------------------------------------------------------------
# ja: the tokeniser-supplied Inflection channel
# ---------------------------------------------------------------------------------------------

def _move_infl(predicted, reference) -> int:
    """Move ``SudInfl`` from the reference onto the predicted doc as ``Inflection``.

    MOVE, not copy, and the direction matters in both halves:

    * onto the PREDICTED doc under the name the tokeniser uses at inference (``Inflection``), since
      that is the string ``sud.MultiHashEmbedFeats.v1`` hashes and the whole point is that training
      and inference hash the same thing;
    * OFF the reference, because the reference is the scoring target. ``SudInfl`` is a transport
      key, not gold morphology, and leaving it there would drag ``morph_acc`` down against a
      feature the morphologiser is not being asked to predict -- a wrong number in a log, which is
      how this repo has been misled before.

    Tokens with no value are left UNSET rather than set to an empty morph: the two are different
    inputs (CLAUDE.md), and ``MultiHashEmbedFeats`` already maps an absent feature to its own
    ``Inflection=`` row.
    """
    if len(predicted) != len(reference):
        return 0
    n = 0
    for pred_tok, ref_tok in zip(predicted, reference):
        vals = ref_tok.morph.get("SudInfl")
        if not vals:
            continue
        pred_tok.set_morph({"Inflection": ",".join(vals)})
        d = ref_tok.morph.to_dict()
        d.pop("SudInfl", None)
        ref_tok.set_morph(d or None)
        n += 1
    return n


@registry.readers("sud.InflCorpus.v1")
def create_infl_reader(path, max_length: int = 0, limit: int = 0, augmenter=None,
                       shuffle: bool = False):
    return InflCorpus(path, max_length=max_length, limit=limit, augmenter=augmenter,
                      shuffle=shuffle)


class InflCorpus(GoldTokCorpus):
    """``GoldTokCorpus`` plus the tokeniser-supplied ``Inflection`` on the predicted doc.

    TRAIN-TIME reader for the ja conditioned-XPOS arms. The corpus must have been stamped by
    ``scripts/stamp_ja_inflection.py`` first; an unstamped corpus read through here is silently a
    plain ``GoldTokCorpus`` with a constant channel, so ``check_xpos_inputs.py`` reports the
    coverage before any training run is burned.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        _move_infl(eg.predicted, reference)
        return eg


@registry.readers("sud.InflEvalCorpus.v1")
def create_infl_eval_reader(path, gold_preproc: bool = True, max_length: int = 0,
                            limit: int = 0, augmenter=None):
    return InflEvalCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                          limit=limit, augmenter=augmenter)


class InflEvalCorpus(Corpus):
    """EVAL-time counterpart, mirroring what ``spacy evaluate --gold-preproc`` builds.

    ``spacy evaluate --gold-preproc`` uses the stock reader, which never runs the tokeniser, so an
    arm conditioned on ``Inflection`` would be scored with one of its inputs deleted. Same reason
    ``eval_sa_compound.py`` exists for ``Compound``, one language over.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        _move_infl(eg.predicted, reference)
        return eg


def _move_infl_tag(predicted, reference) -> int:
    """As ``_move_infl``, plus the tokeniser's XPOS onto the predicted doc's ``tag``.

    The tag goes on the TAG ATTRIBUTE, not into MORPH, because that is where the tokeniser puts it
    at inference (``token.tag_ = dtoken.tag``) and therefore what an ``attrs=[..., "TAG"]`` embed
    column reads. Transporting it through FEATS and leaving it there would train on a channel that
    is empty at inference -- the exact skew this whole mechanism exists to close.

    ⚠ The value is the TOKENISER's tag, carried in ``SudTag``; the reference's own ``tag`` is GOLD
    XPOS and is never copied. Gold would be leakage into the tagger's own target, and it would
    misrepresent inference, where the tokeniser agrees with gold on 76.7 % of tokens.
    """
    if len(predicted) != len(reference):
        return 0
    n = 0
    for pred_tok, ref_tok in zip(predicted, reference):
        d = ref_tok.morph.to_dict()
        infl = d.pop("SudInfl", None)
        tag = d.pop("SudTag", None)
        if infl:
            pred_tok.set_morph({"Inflection": infl})
        if tag:
            pred_tok.tag_ = tag
            n += 1
        ref_tok.set_morph(d or None)
    return n


@registry.readers("sud.InflTagCorpus.v1")
def create_infl_tag_reader(path, max_length: int = 0, limit: int = 0, augmenter=None,
                           shuffle: bool = False):
    return InflTagCorpus(path, max_length=max_length, limit=limit, augmenter=augmenter,
                         shuffle=shuffle)


class InflTagCorpus(GoldTokCorpus):
    """TRAIN-time reader for the ja arms conditioned on Inflection AND the tokeniser's XPOS."""

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        _move_infl_tag(eg.predicted, reference)
        return eg


@registry.readers("sud.InflTagEvalCorpus.v1")
def create_infl_tag_eval_reader(path, gold_preproc: bool = True, max_length: int = 0,
                                limit: int = 0, augmenter=None):
    return InflTagEvalCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                             limit=limit, augmenter=augmenter)


class InflTagEvalCorpus(Corpus):
    """EVAL-time counterpart. Scoring one of these arms through the stock reader deletes BOTH
    channels; scoring it through ``InflEvalCorpus`` deletes only the tag, which is worse than
    useless because it looks like it worked."""

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        _move_infl_tag(eg.predicted, reference)
        return eg
