#!/usr/bin/env python3
"""`sud.GenericCorpus.v1` — one training stream over thirteen per-language `.spacy` corpora.

It does three things the stock reader cannot, and each is load-bearing for the generic arm.

**1. It stamps the language on every doc.** `Doc._.tb_lang` is how `sud.GenericEmbed.v1` picks which
of the thirteen row-sets to look a token up in. It is NOT a model input -- no parameter varies with
it -- but without it the vector channel cannot be read at all, and the layer refuses rather than
guessing (a default would miss nearly every token and score like a dead channel).

**2. It copies gold UPOS, FEATS and LEMMA onto the PREDICTED doc.** This looks like leakage and is
not: they are the generic parser's declared INPUTS. The arm is defined as "given a tagged,
morphologically analysed token sequence, produce a tree", so the tagging has to be present on the
doc the parser actually reads. The stock reader builds the predicted doc from the reference's words
and nothing else, which would leave POS at 0 and MORPH unset on every token and train the model to
ignore three channels that then appear from nowhere at inference -- the same gap
`sud.CompoundCorpus.v1` exists to close for sa's `Compound` feature.

  ⚠ WHAT IS COPIED IS EXHAUSTIVE AND DELIBERATE. UPOS, FEATS and LEMMA, and nothing else. HEAD and
  DEPREL are the target. `Shared` never reaches here at all -- `prep_generic.py` strips it from
  FEATS upstream, because it is a fact about the coordination structure the parser is trying to
  recover. LEMMA is copied only because sa's aligned vectors are keyed by lemma rather than by form
  (`docs/aligned-vectors.md` trap 4); for the other twelve languages it is inert, and an arm that
  wanted to prove that could drop it and re-measure.

**3. It interleaves the languages.** The per-language files are read whole and then SHUFFLED
TOGETHER, so a batch mixes languages instead of walking one treebank at a time. Consecutive
same-language batches would make the optimiser see thirteen sequential domain shifts per epoch, and
whatever the final model learned about Telugu would have been overwritten by Latin.

The predicted doc carries the reference's GOLD TOKENISATION, inherited from `GoldTokCorpus`'s
contract: docs are ten sentences (`spacy convert -n 10`) so the parser learns to START a sentence
rather than scoring a cosmetic `SENTS_F` of 100 (CLAUDE.md hazard 4), and no tokeniser runs, so
thirteen mutually incompatible tokenisers never enter the picture.

    [corpora.train]
    @readers = "sud.GenericCorpus.v1"
    path = "corpus_generic"
    split = "train"
"""
from __future__ import annotations

import os
import pathlib
import random
import sys

from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.util import registry

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Either embed module registers `Doc._.tb_lang`; prefer v2 so the v2 pipeline has no hidden
# dependency on v1 -- which in turn imports `aligned_vectors`, a file that is not even tracked.
try:
    import sud_generic_embed_v2  # noqa: E402,F401
except ImportError:  # pragma: no cover - v1 checkouts
    import sud_generic_embed  # noqa: E402,F401

#: Loaded corpora, keyed by path. `generate` is called afresh for every epoch and these files are
#: read whole; re-reading thirteen DocBins each time would dominate a short epoch.
_DOC_CACHE: dict = {}


def _load(path, vocab):
    if path not in _DOC_CACHE:
        _DOC_CACHE[path] = list(DocBin().from_disk(path).get_docs(vocab))
    return _DOC_CACHE[path]


def languages_in(directory, split):
    """The languages this directory actually holds for `split`.

    Derived from the files rather than from a hardcoded list, which is what makes `--hold-out` work:
    a zero-shot arm's corpus simply has no `ja-train.spacy`, and nothing else needs to know.
    """
    d = pathlib.Path(directory)
    return sorted(p.name.split("-")[0] for p in d.glob(f"*-{split}.spacy"))


@registry.readers("sud.GenericCorpus.v1")
def create_generic_corpus(path: str, split: str = "train", seed: int = 0, limit: int = 0):
    state = {"epoch": 0}

    def generate(nlp):
        langs = languages_in(path, split)
        if not langs:
            raise ValueError(
                f"sud.GenericCorpus.v1: no <lang>-{split}.spacy under {path}. Build the corpus "
                f"with scripts/prep_generic.py and scripts/train_generic.sh (which runs the "
                f"`spacy convert` step).")
        pairs = []
        for lang in langs:
            for doc in _load(os.path.join(path, f"{lang}-{split}.spacy"), nlp.vocab):
                pairs.append((lang, doc))

        # MUST be finite. spaCy's `initialize` iterates the whole training corpus to collect
        # labels, so an infinite generator hangs initialisation at 100 % CPU with no output --
        # the failure `sampling_corpus.py` records hitting with a `while True`.
        rng = random.Random(seed + state["epoch"])
        rng.shuffle(pairs)
        state["epoch"] += 1
        if limit:
            pairs = pairs[:limit]
        for lang, reference in pairs:
            yield _example(nlp, lang, reference)

    return generate


def _example(nlp, lang, reference):
    reference._.tb_lang = lang
    predicted = Doc(nlp.vocab,
                    words=[t.text for t in reference],
                    spaces=[bool(t.whitespace_) for t in reference])
    predicted._.tb_lang = lang
    for pred_tok, ref_tok in zip(predicted, reference):
        # The three declared INPUTS. Note that copying `ref_tok.morph` verbatim preserves the
        # unset/empty distinction exactly as the reference has it -- which costs nothing here
        # because `MultiHashEmbedFeats` reads each feature individually and renders both as
        # `Case=`, but would matter to any layer reading the bundle hash (CLAUDE.md; sa, 6.8 LAS).
        pred_tok.pos = ref_tok.pos
        pred_tok.set_morph(ref_tok.morph)
        pred_tok.lemma = ref_tok.lemma
    return Example(predicted, reference)


def annotate(doc, lang, tagged_source=None):
    """Prepare a doc for INFERENCE the same way the reader prepares one for training.

    Exposed because the training-time and inference-time input regimes must be identical and there
    should be exactly one place that says what they are (CLAUDE.md hazard 10: ask the model rather
    than assuming its input regime). `tagged_source`, when given, is a doc of the same length whose
    UPOS/FEATS/LEMMA are copied across -- that is the shape of "run a tagger, then this parser".
    """
    doc._.tb_lang = lang
    if tagged_source is not None:
        if len(tagged_source) != len(doc):
            raise ValueError(f"annotate(): tagged_source has {len(tagged_source)} tokens and the "
                             f"doc has {len(doc)}; they must be token-for-token the same.")
        for tok, src in zip(doc, tagged_source):
            tok.pos = src.pos
            tok.set_morph(src.morph)
            tok.lemma = src.lemma
    return doc
