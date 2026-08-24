#!/usr/bin/env python3
"""Single ``--code`` entry point for the generic parser v2.

``spacy train --code`` takes ONE file, so this imports everything `configs/config_g2_*.cfg` names:
the per-FEATS-category embed, the v2 language-agnostic layer, and the multi-language reader.

Deliberately NOT `seg_code.py`. That loads every released arm's tokenizer so a monolingual wheel can
be opened; this arm has no tokenizer of its own, and importing eighty languages' optional
dependencies to train a model that reads no strings would only add ways for the run to fail.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sud_feats_embed      # noqa: E402,F401  (sud.MultiHashEmbedFeats.v1 — FEATS decomposition)
import sud_generic_embed_v2  # noqa: E402,F401  (sud.GenericEmbed.v2 — UPOS + FEATS + typology)
import generic_corpus        # noqa: E402,F401  (sud.GenericCorpus.v1 — the multi-language stream)
