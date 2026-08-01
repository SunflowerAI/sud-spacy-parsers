#!/usr/bin/env python3
"""Refuse to silently clobber a hand-maintained derived config.

``make_morph_config.py`` / ``make_lemma_config.py`` regenerate ``configs/config_<lang>_{morph,lemma}.cfg``
from scratch every time ``train_morph.sh`` / ``train_lemma.sh`` runs, and they always emit a flat
``spacy.HashEmbedCNN.v2`` encoder. That architecture **hard-codes** ``NORM/PREFIX/SUFFIX/SHAPE`` and
cannot express ``MORPH`` — so for ``sa``, whose morphologiser and lemmatiser were hand-edited to a
``spacy.Tok2Vec.v2`` + ``MultiHashEmbed`` stack precisely so they can read the tokeniser's
``Compound=Yes`` INPUT feature, a regeneration deletes that feature and retrains a strictly worse
component with no error and nothing in the log to notice.

This guard turns that into a loud failure. It is deliberately generic: any arm whose derived config
has been customised away from the generator's output is protected, not just ``sa``.
"""
import pathlib

from thinc.api import Config


def guard_overwrite(out, component, expected_arch, force=False):
    """Raise unless it is safe to overwrite `out`.

    Safe means: the file does not exist yet, `--force` was passed, or the encoder architecture
    already on disk is the one this generator produces (so nothing hand-made is being lost).
    """
    path = pathlib.Path(out)
    if force or not path.exists():
        return
    try:
        old = Config().from_disk(path, interpolate=False)
        arch = old["components"][component]["model"]["tok2vec"]["@architectures"]
    except Exception:
        return                      # unreadable, or not the shape we generate: nothing to protect
    if arch == expected_arch:
        return
    raise SystemExit(
        f"\n{out} already exists and its {component} encoder is `{arch}`, not the\n"
        f"`{expected_arch}` this script generates — i.e. it has been hand-maintained.\n"
        f"Regenerating would discard that customisation (for sa this silently drops the MORPH\n"
        f"input feature and costs real accuracy). Refusing.\n\n"
        f"  * to train from the checked-in config, skip the generator and pass {out} to spacy train\n"
        f"  * to regenerate anyway, re-run with --force\n")
