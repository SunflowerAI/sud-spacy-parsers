#!/usr/bin/env python3
"""Single ``--code`` entry point for the generic parser v3.

v2's three imports plus the v3 layer. `sud_generic_embed_v3` imports v2 rather than copying it, so
both architectures are registered and `config_g3_base.cfg` -- which is the RELEASED v2 arm, verified
key-for-key against `training_v2_g2_bundle/config.cfg` -- still names `sud.GenericEmbed.v2` and
resolves. One `--code` file trains every arm in the sweep, baseline included, which is the point:
a baseline built from a different entry point is a baseline that can differ in something nobody
wrote down.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sud_feats_embed        # noqa: E402,F401  (sud.MultiHashEmbedFeats.v1)
import sud_generic_embed_v2   # noqa: E402,F401  (sud.GenericEmbed.v2 — the released arm)
import sud_generic_embed_v3   # noqa: E402,F401  (sud.GenericEmbed.v3 — + the lexical channel)
import generic_corpus         # noqa: E402,F401  (sud.GenericCorpus.v1)
