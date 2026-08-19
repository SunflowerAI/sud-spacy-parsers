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


@registry.readers("sud.OracleCorpus.v1")
def create_oracle_reader(path, gold_preproc: bool = True, max_length: int = 0,
                         limit: int = 0, augmenter=None):
    return OracleCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                        limit=limit, augmenter=augmenter)


class OracleCorpus(Corpus):
    """Like ``spacy.Corpus``, but the PREDICTED doc carries the reference's LEMMA and FEATS.

    ⚠ THIS IS LEAKAGE, DELIBERATELY. Unlike ``CompoundCorpus`` — which copies ``Compound`` only
    because the tokeniser supplies the identical value at inference — nothing supplies gold lemma
    or gold morphology at inference. An arm trained through this reader is an ORACLE: it bounds
    what a parser could gain from a lemma channel and per-feature morphology channels if they were
    perfect, and it is not shippable. Its purpose is to answer, in ONE training run, whether the
    pipeline surgery that a real (predicted) version needs is worth doing at all.

    The exposure is controlled by the EMBED, not here: this reader copies the whole gold bundle and
    ``sud.MultiHashEmbedFeats.v1``'s ``feats`` list decides which features the model can actually
    see. Keep syntactic annotations (``Shared``) off that list — they are part of what is being
    predicted.

    LEMMA falls back to the FORM where the treebank has none, matching ``trainable_lemmatizer``'s
    ``backoff = "orth"``, so the oracle's input regime is the one a real lemmatiser would produce
    rather than a mix of strings and unset values. (On ``corpus_sa_csl_mwt`` the fallback never
    fires: 0 of 163 802 tokens lack a lemma.)

    MORPH is set only where the reference HAS features: ``set_morph("")`` stores the *empty* morph
    (key 456) where an untouched token is *unset* (key 0), and the two are different inputs — the
    distinction once cost sa 6.8 LAS. ``MultiHashEmbedFeats`` reads each feature individually and is
    immune by construction, but the reader must not depend on that.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        # With gold_preproc the predicted doc is token-for-token the reference, so the copy is well
        # defined; without it the base class re-tokenises and the two need not align, and there is
        # no honest way to project gold annotation across a different tokenisation.
        if len(eg.predicted) != len(reference):
            raise ValueError(
                f"OracleCorpus needs gold_preproc = true: predicted {len(eg.predicted)} tokens "
                f"vs reference {len(reference)}")
        for pred_tok, ref_tok in zip(eg.predicted, reference):
            pred_tok.lemma_ = ref_tok.lemma_ or ref_tok.text
            if ref_tok.morph:
                pred_tok.set_morph(ref_tok.morph)
        return eg


@registry.readers("sud.NormCorpus.v1")
def create_norm_reader(path, gold_preproc: bool = True, max_length: int = 0,
                       limit: int = 0, augmenter=None, shuffle: bool = False):
    # `shuffle` matters only under `max_epochs = -1`, which a corpus-level augmenter REQUIRES: with
    # `0` spaCy lists the corpus once and reshuffles that same list, so one augmentation is sampled
    # per document for the whole run and never varies (standing hazard 9). `-1` streams the corpus
    # and stops the loop shuffling, so the reader has to shuffle itself.
    return NormCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                      limit=limit, augmenter=augmenter, shuffle=shuffle)


class NormCorpus(CompoundCorpus):
    """``CompoundCorpus`` plus the reference's NORM.

    For a corpus written by ``scripts/make_norm_corpus.py``, whose NORM column holds the
    sandhi-reversed (padapāṭha) form that ``sud_unsandhi`` predicts from the surface. The predicted
    doc is built from gold words, so its NORM would otherwise fall back to the lexeme default
    (lower-cased ORTH) and the channel the arm is being trained on would silently be absent —
    the same gap ``CompoundCorpus`` exists to close, one column over.

    This is NOT leakage in the way ``OracleCorpus`` is: the value copied here is the TRANSDUCER's
    prediction, baked into the corpus, and the released frontend runs the identical transducer
    inside the tokeniser (``sa_tokenizer.py`` stage 2) before any component sees the doc. The one
    honest caveat is that the transducer was trained on this training split, so it scores 0.9855
    there against 0.9685 on the held-out test — the parser trains on a NORM 1.7 points cleaner than
    it will meet. Jackknifing the transducer would close that; it has not been done.

    ⚠ An arm trained through this reader is only deployable once the TOKENISER writes the same value
    into ``token.norm_`` (it currently publishes it on ``Token._.unsandhied`` only, which no embed
    can read). Without that step the arm would be trained on padapāṭha NORM and run on surface NORM.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        if len(eg.predicted) == len(reference):
            for pred_tok, ref_tok in zip(eg.predicted, reference):
                pred_tok.norm_ = ref_tok.norm_
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


@registry.readers("sud.LemmaOracleCorpus.v1")
def create_lemma_oracle_reader(path, gold_preproc: bool = True, max_length: int = 0,
                               limit: int = 0, augmenter=None):
    return LemmaOracleCorpus(path, gold_preproc=gold_preproc, max_length=max_length,
                             limit=limit, augmenter=augmenter)


class LemmaOracleCorpus(CompoundCorpus):
    """Gold LEMMA on the predicted doc, and NOTHING else. ``CompoundCorpus``'s ``Compound`` still
    comes across, because the tokeniser supplies it at inference and every arm in the grid has it.

    WHY THIS EXISTS, WRITTEN DOWN BECAUSE IT COST A RUN. The lemma-vector arms were first built on
    ``sud.OracleCorpus.v1``, which stamps gold LEMMA *and gold FEATS*. With ``MORPH`` among the
    embed's attrs that silently handed them the gold-morphology block channel — worth +7.59 LAS on
    its own — so an arm intended to isolate "lemma identity vs lemma similarity" was actually
    measuring "lemma vectors PLUS gold morphology" against a lemma-hash arm that had neither. The
    capacity control is what exposed it: a block of CONSTANT vectors scored 60.85 dev against the
    base arm's 54.58, and a constant cannot be worth six points.

    A reader that copies more than the experiment intends does not fail; it inflates, and it inflates
    the arm you were hoping to see win.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        if len(eg.predicted) == len(reference):
            for pred_tok, ref_tok in zip(eg.predicted, reference):
                pred_tok.lemma_ = ref_tok.lemma_ or ref_tok.text
        return eg


@registry.readers("sud.GoldTokNormCorpus.v1")
def create_gold_tok_norm_reader(path, max_length: int = 0, limit: int = 0, augmenter=None,
                                shuffle: bool = False):
    return GoldTokNormCorpus(path, max_length=max_length, limit=limit,
                             augmenter=augmenter, shuffle=shuffle)


class GoldTokNormCorpus(GoldTokCorpus):
    """``GoldTokCorpus`` (whole multi-sentence docs, gold tokenisation) plus NORM and ``Compound``.

    Needed the moment an arm reads either. ``GoldTokCorpus`` builds its predicted doc from gold
    WORDS alone, so NORM falls back to the lexeme default — lower(ORTH), the SANDHIED surface — and
    ``sud.AnalyserFeatsEmbed.v1``, whose candidate-set lookup is keyed on ``token.norm_``, goes to
    its silent bit on every token. That is the same shape as the three ``clause_parser`` bugs: a
    rebuilt ``Doc`` silently missing an annotation something downstream depends on, with nothing
    raising and only the score to show for it.

    ``Compound`` is copied for the reason ``CompoundCorpus`` exists: the tokeniser supplies it at
    inference at P/R 0.9998, so training without it teaches the model to ignore a feature that then
    appears from nowhere.

    Use this rather than ``gold_preproc`` wherever the point is that the parser sees MULTI-CLAUSE
    input and learns to segment — ``gold_preproc`` splits every doc back into single sentences and
    would discard a clause merge entirely.
    """

    def _make_example(self, nlp, reference, gold_preproc) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        if len(eg.predicted) == len(reference):
            for pred_tok, ref_tok in zip(eg.predicted, reference):
                pred_tok.norm_ = ref_tok.norm_
                if ref_tok.morph.get("Compound"):
                    pred_tok.set_morph("Compound=Yes")
        return eg
