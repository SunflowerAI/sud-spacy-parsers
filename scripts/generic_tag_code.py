#!/usr/bin/env python3
"""Single `--code` entry point for the generic TAGGING arm."""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sud_feats_embed        # noqa: E402,F401
import sud_generic_embed_v2   # noqa: E402,F401  (sud.GenericTagEmbed.v1)
import generic_tag_corpus     # noqa: E402,F401  (sud.GenericTagCorpus.v1)
