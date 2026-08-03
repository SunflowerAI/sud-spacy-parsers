#!/usr/bin/env python3
"""`sud.SamplingCorpus.v1` — rebalance a corpus by SAMPLING, not by duplicating documents.

Duplicating a small split to upweight it is the obvious move and it is the wrong one, because it
inflates the training set. Sanskrit's UFAL half is 1 323 tokens against Vedic's 161 985, so bringing
it to parity by copying costs:

    plain multitask         163 308 parser-visible tokens   1.0x    ~50 min to converge
    Vedic x1  + UFAL x122   323 391                         2.0x    ~20 min PER EVAL
    Vedic x5  + UFAL x612  1 619 601                         9.9x    stalled; killed after 3 h

and the cost lands specifically on the parser, the expensive transition-based component that has to
run its oracle over every syntactic token. The x612 arm produced one checkpoint in three hours.

Sampling separates the two things duplication conflates. The MIX is what we wanted to change; the
SIZE was collateral damage. This reader keeps the syntactic budget fixed at its original token count
and fills it by drawing from the two pools at the requested ratio — so UFAL reaches parity while an
epoch costs exactly what it did before. Documents with no syntax (DCS, which supplies morphology and
lemmas only) pass through untouched, since the parser never sees them.

    [corpora.train]
    @readers = "sud.SamplingCorpus.v1"
    path = "corpus_sa_multitask/train.spacy"
    boost_path = "corpus_sa_split/ufal_train.spacy"
    boost_ratio = 1.0        # target boost:base token ratio WITHIN the syntactic budget
    seed = 0

`boost_ratio = 1.0` means parity. The stream is reshuffled every epoch, so the model does not see
the same UFAL subset in the same order each time.
"""
import random

from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.training import Example


_DOC_CACHE = {}


def _load(path, vocab):
    """Load once per process: `generate` is called afresh for every epoch."""
    if path not in _DOC_CACHE:
        _DOC_CACHE[path] = list(DocBin().from_disk(path).get_docs(vocab))
    return _DOC_CACHE[path]


@util.registry.readers("sud.SamplingCorpus.v1")
def create_sampling_corpus(path: str, boost_path: str, boost_ratio: float = 1.0, seed: int = 0):
    state = {"epoch": 0}

    def generate(nlp):
        vocab = nlp.vocab
        base_docs = _load(path, vocab)
        boost_docs = _load(boost_path, vocab)

        # the base file already contains the boost split; separate them by content, not position
        boost_texts = {d.text for d in boost_docs}
        with_syntax, no_syntax = [], []
        for d in base_docs:
            (with_syntax if d.has_annotation("DEP") else no_syntax).append(d)
        base_syn = [d for d in with_syntax if d.text not in boost_texts]
        boost_syn = [d for d in boost_docs if d.has_annotation("DEP")]

        budget = sum(len(d) for d in with_syntax)          # keep the ORIGINAL syntactic budget
        want_boost = budget * boost_ratio / (1.0 + boost_ratio)

        rng = random.Random(seed + state["epoch"])

        def draw(pool, target_tokens):
            """Sample with replacement until the token target is met (pool may be tiny)."""
            out, got = [], 0
            if not pool:
                return out
            while got < target_tokens:
                d = pool[rng.randrange(len(pool))]
                out.append(d); got += len(d)
            return out

        # MUST be finite. spaCy's `initialize` iterates the whole training corpus to collect
        # labels, so an infinite generator hangs initialization forever at 100 % CPU with no
        # output — which is exactly what an earlier `while True` version did. One call = one
        # epoch; spaCy calls `generate` again for the next, and `state["epoch"]` reshuffles it.
        stream = draw(boost_syn, want_boost) + draw(base_syn, budget - want_boost) + no_syntax
        rng.shuffle(stream)
        state["epoch"] += 1
        for d in stream:
            yield _example(nlp, d)

    return generate


def _example(nlp, reference):
    """Predicted doc built from the reference's GOLD WORDS, never by re-tokenising.

    `nlp.make_doc` is wrong here for the same reason `sud.CompoundCorpus.v1` exists: the Sanskrit
    tokeniser REWRITES its input (CSLise, de-sandhi, compound-join strip), so a predicted doc made
    from `reference.text` need not align with the reference at all and spaCy raises E949. Building
    from gold words is what `gold_preproc` does, and it is what every sa arm trains under.

    `Compound` is copied across because the tokeniser supplies exactly that feature at inference and
    the encoder reads MORPH as an input. Nothing else from the gold FEATS is copied — that would be
    leakage into what the morphologizer is being asked to predict. Note also that an UNSET morph and
    an EMPTY one are different keys to the encoder, so the feature is set only where it has a value.
    """
    words = [t.text for t in reference]
    spaces = [bool(t.whitespace_) for t in reference]
    pred = Doc(nlp.vocab, words=words, spaces=spaces)
    for pred_tok, ref_tok in zip(pred, reference):
        if ref_tok.morph.get("Compound"):
            pred_tok.set_morph("Compound=Yes")
    return Example(pred, reference)
