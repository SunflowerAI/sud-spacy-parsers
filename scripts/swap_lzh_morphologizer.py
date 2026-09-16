#!/usr/bin/env python
"""Replace an lzh arm's `morphologizer` with a differently-trained one, in place.

Modelled on scripts/swap_lzh_mftagger.py (same pattern, same traps). Written for the
SikuBERT-vector morphologiser (`configs/config_lzh_morph_adjfix_siku*.cfg`,
docs/chinese-family.md "The morphologiser-side channel, re-measured on the ADJ-recoded
(v0.3.2) generation") but takes any donor with a `morphologizer` component.

⚠ MUST STAY BEFORE `tagger` (and after `tok2vec`/`parser`, which don't read it). The
combined multi-field tagger reads UPOS+FEATS as input; a morphologizer placed after it
would leave the tagger reading an empty POS column. This script preserves the donor's
position by re-inserting where the original component was.

⚠ CARRY THE VECTORS if the donor's morphologizer reads a static-vector channel
(`pretrained_vectors = true` -> a StaticVectors lookup on `doc.vocab.vectors` at forward
time). `add_pipe(source=...)` copies the component, not the donor's vocab, so without this
the swapped-in component runs on an all-zero table and is silently out of distribution on
the one input it was trained to use.

Usage:
    swap_lzh_morphologizer.py IN_MODEL OUT_MODEL --donor training_lzh_morph_adjfix_siku_s0/model-best
"""
import argparse
import importlib.util
import pathlib

import numpy as np
import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--donor", default="training_lzh_morph_adjfix_siku_s0/model-best")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    nlp = spacy.load(a.in_model)
    if "morphologizer" not in nlp.pipe_names:
        raise SystemExit(f"{a.in_model}: no `morphologizer` to replace ({nlp.pipe_names})")
    pos = nlp.pipe_names.index("morphologizer")
    if "tagger" in nlp.pipe_names and pos > nlp.pipe_names.index("tagger"):
        raise SystemExit(f"{a.in_model}: morphologizer follows tagger — the tagger reads UPOS/FEATS "
                          f"from it, this would silently reorder the dependency ({nlp.pipe_names})")
    before = nlp.pipe_names[pos + 1] if pos + 1 < len(nlp.pipe_names) else None

    donor = spacy.load(a.donor)
    if "morphologizer" not in donor.pipe_names:
        raise SystemExit(f"{a.donor}: no `morphologizer` component ({donor.pipe_names})")

    nlp.remove_pipe("morphologizer")
    kw = {"before": before} if before else {"last": True}
    nlp.add_pipe("morphologizer", source=donor, **kw)
    pipe = nlp.get_pipe("morphologizer")

    # ⚠ CARRY THE VECTORS -- see module docstring; same trap swap_lzh_mftagger.py guards against.
    if donor.vocab.vectors.shape[0]:
        nlp.vocab.vectors = donor.vocab.vectors
    have = nlp.vocab.vectors
    reads_vectors = any(n.name == "static_vectors" for n in pipe.model.walk())
    if reads_vectors and (not have.shape[0] or not int((np.abs(have.data).sum(1) > 0).sum())):
        raise SystemExit(
            "  REFUSING: the morphologizer reads a static-vector channel but the vocab has no "
            "usable vectors. Shipping this would run that channel on zeros.")
    print(f"  reads_vectors={reads_vectors}  vectors carried: {have.shape}  nonzero rows "
          f"{int((np.abs(have.data).sum(1) > 0).sum()) if have.shape[0] else 0}")

    # ⚠ STRIP ANY HOST-PATH CONFIG VALUE. Unlike the multifield tagger's `tables`, the stock
    # `spacy.Morphologizer.v1` factory takes no file-path argument, but check for one anyway --
    # a future donor config could add one, and CLAUDE.md hazard 4 is exactly this class of bug.
    cfg = nlp.get_pipe_config("morphologizer")
    path_like = {k: v for k, v in cfg.items() if isinstance(v, str) and ("/" in v or v.endswith((".json", ".cfg")))}
    if path_like:
        print(f"  ⚠ path-like config values found, NOT auto-stripped, inspect before shipping: {path_like}")

    print(f"{a.out_model}: {nlp.pipe_names}")
    nlp.to_disk(a.out_model)


if __name__ == "__main__":
    main()
