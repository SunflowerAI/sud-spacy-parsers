#!/usr/bin/env python3
"""Single ``--code`` entry point for the GENERIC (language-agnostic) parser.

``spacy train --code`` takes ONE file, so this imports everything `configs/config_generic*.cfg`
names: the cross-lingual embed, the multi-language reader, and the per-feature FEATS embed the
first of those builds on.

Deliberately NOT `seg_code.py`. That module loads every released arm's tokenizer and component so a
monolingual wheel can be opened; the generic arm has no tokenizer of its own and needs none of it,
and importing thirteen languages' optional dependencies to train a model that reads no strings
would only add ways for the run to fail.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sud_feats_embed   # noqa: E402,F401  (sud.MultiHashEmbedFeats.v1 — the FEATS decomposition)
import sud_generic_embed # noqa: E402,F401  (sud.GenericEmbed.v1 — UPOS + FEATS + aligned vector)
import generic_corpus    # noqa: E402,F401  (sud.GenericCorpus.v1 — the thirteen-language stream)
