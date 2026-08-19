#!/usr/bin/env python3
"""Point the ko morphologiser and lemmatiser configs at a given base, by the freeze recipe.

The layers above a base are not retrained from their own configs by hand: those configs name the
base they were built on, and a base change that leaves the name behind produces a stack trained on
the PREVIOUS generation while every log looks healthy. That is standing hazard 2 in its usual shape,
and the fix is the same one — generate the config from the arm you actually mean.

The recipe itself is unchanged (CLAUDE.md): source the base's components, FREEZE them, and train
only the new one, giving it its OWN small HashEmbedCNN rather than a listener. Frozen components
must come out byte-identical, which `scripts/verify_ko_release.sh` asserts with `cmp` rather than
assuming.

    .venv/bin/python scripts/make_ko_stack_configs.py training_ko_anseg_s2/model-best
"""
from __future__ import annotations

import sys

from thinc.api import Config


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = sys.argv[1]

    morph = Config().from_disk("configs/config_ko_eojeol_morph.cfg", interpolate=False)
    for comp in ("tok2vec", "tagger", "parser"):
        morph["components"][comp]["source"] = base
    morph.to_disk("configs/config_ko_anseg_morph.cfg")
    print(f"wrote configs/config_ko_anseg_morph.cfg   sourcing {base}")

    lemma = Config().from_disk("configs/config_ko_eojeol_lemma.cfg", interpolate=False)
    for comp in ("tok2vec", "tagger", "parser", "morphologizer"):
        lemma["components"][comp]["source"] = "training_ko_anseg_morph/model-best"
    lemma.to_disk("configs/config_ko_anseg_lemma.cfg")
    print("wrote configs/config_ko_anseg_lemma.cfg   sourcing training_ko_anseg_morph/model-best")


if __name__ == "__main__":
    main()
