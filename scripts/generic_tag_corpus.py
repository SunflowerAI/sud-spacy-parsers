#!/usr/bin/env python3
"""`sud.GenericTagCorpus.v1` — the multi-language stream for the TAGGING arm.

The parser's reader (`sud.GenericCorpus.v1`) copies gold UPOS, FEATS and LEMMA onto the predicted
doc, because for the parser those are declared INPUTS. Here they are the targets, so the predicted
doc carries the words and `tb_lang` and nothing else.

⚠ Getting this wrong does not raise. A morphologiser handed its own gold UPOS would train to ~100 %
and reveal nothing, and the only symptom would be a suspiciously good number.
"""
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.util import registry

import sud_generic_embed_v2  # noqa: F401  (registers Doc._.tb_lang)

_CACHE: dict = {}


def _load(path, vocab):
    if path not in _CACHE:
        _CACHE[path] = list(DocBin().from_disk(path).get_docs(vocab))
    return _CACHE[path]


def languages_in(directory, split):
    out = []
    for p in sorted(pathlib.Path(directory).glob(f"*-{split}.spacy")):
        out.append(p.name.rsplit("-", 1)[0])
    return out


@registry.readers("sud.GenericTagCorpus.v1")
def create_generic_tag_corpus(path: str, split: str = "train", seed: int = 0,
                              limit: int = 0, give_pos: bool = False):
    state = {"epoch": 0}

    def generate(nlp):
        langs = languages_in(path, split)
        if not langs:
            raise ValueError(f"sud.GenericTagCorpus.v1: no <lang>-{split}.spacy under {path}")
        pairs = []
        for lang in langs:
            for doc in _load(os.path.join(path, f"{lang}-{split}.spacy"), nlp.vocab):
                pairs.append((lang, doc))
        # MUST be finite: spaCy's `initialize` iterates the whole training corpus to collect
        # labels, so an infinite generator hangs at 100 % CPU with no output.
        rng = random.Random(seed + state["epoch"])
        rng.shuffle(pairs)
        state["epoch"] += 1
        if limit:
            pairs = pairs[:limit]
        for lang, reference in pairs:
            reference._.tb_lang = lang
            predicted = Doc(nlp.vocab,
                            words=[t.text for t in reference],
                            spaces=[bool(t.whitespace_) for t in reference])
            predicted._.tb_lang = lang
            if give_pos:
                # The annotator supplies UPOS by hand anyway (it is the one column that cannot be
                # transferred), so an arm that READS it is the realistic configuration -- and
                # predicting FEATS given the part of speech is a far smaller problem. FEATS and
                # LEMMA stay off the predicted doc: those are still the targets.
                for pt, rt in zip(predicted, reference):
                    pt.pos = rt.pos
            yield Example(predicted, reference)

    return generate
