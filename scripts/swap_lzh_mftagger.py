#!/usr/bin/env python
"""Replace an lzh arm's `tagger` with the trained COMBINED multi-field tagger, in place.

⚠ IT IS INSTALLED UNDER THE NAME `tagger`, NOT `multifield_tagger`. Both of `package_sud.sh`'s
guards — the XPOS-downstream order check and the silenced-tagger `overwrite` check — look for a
component literally named `tagger`. A differently-named component passes both without being
checked, which is the failure mode those guards exist to prevent. The FACTORY is
`multifield_tagger`; only the pipe name is `tagger`.

⚠ AND IT MUST STAY BEHIND THE MORPHOLOGISER. The whole point of the component is that UPOS is an
input (and, with `upos_mask`, a constraint), so a tagger placed before the morphologiser would read
an empty POS column. The order guard enforces this; this script preserves the donor's position.

Usage:
    swap_lzh_mftagger.py IN_MODEL OUT_MODEL --donor training_lzh_mftagger/model-best
"""
import argparse
import importlib.util
import pathlib

import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--donor", default="training_lzh_mftagger/model-best")
    ap.add_argument("--upos-mask", action="store_true",
                    help="ship with masking ON (+3.11 TAG on gold UPOS, -0.33 on predicted)")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    nlp = spacy.load(a.in_model)
    if "tagger" not in nlp.pipe_names:
        raise SystemExit(f"{a.in_model}: no `tagger` to replace ({nlp.pipe_names})")
    pos = nlp.pipe_names.index("tagger")
    if "morphologizer" in nlp.pipe_names and pos < nlp.pipe_names.index("morphologizer"):
        raise SystemExit(f"{a.in_model}: tagger precedes morphologizer — this component reads UPOS")
    before = nlp.pipe_names[pos + 1] if pos + 1 < len(nlp.pipe_names) else None

    donor = spacy.load(a.donor)
    dname = next(n for n in donor.pipe_names if n.endswith("multifield_tagger")
                 or donor.get_pipe_meta(n).factory == "multifield_tagger")
    nlp.remove_pipe("tagger")
    # `source=` copies the component under the DONOR's name, so the donor's pipe is renamed to
    # `tagger` afterwards rather than at add time (spaCy's add_pipe takes no source_name).
    kw = {"before": before} if before else {"last": True}
    nlp.add_pipe(dname, source=donor, **kw)
    if dname != "tagger":
        nlp.rename_pipe(dname, "tagger")
    pipe = nlp.get_pipe("tagger")

    # ⚠ CARRY THE VECTORS. The component reads `sud.StaticVecChannel.v1`, which looks the table up
    # on `doc.vocab.vectors` at FORWARD time — and `add_pipe(source=…)` copies the component, not
    # the donor's vocab. Without this the shipped wheel runs the channel on an EMPTY table: every
    # lookup returns zeros, the model loads and tags and is quietly out of distribution on the one
    # input it was given external knowledge through. spaCy says so (W113) and says it as a warning.
    import numpy as np
    if donor.vocab.vectors.shape[0]:
        nlp.vocab.vectors = donor.vocab.vectors
    have = nlp.vocab.vectors
    reads_vectors = any(n.name == "static_vectors" for n in pipe.model.walk())
    if reads_vectors and (not have.shape[0] or not int((np.abs(have.data).sum(1) > 0).sum())):
        raise SystemExit(
            "  REFUSING: the tagger reads a static-vector channel but the vocab has no usable "
            "vectors. Shipping this would run that channel on zeros.")
    print(f"  vectors carried: {have.shape}  nonzero rows "
          f"{int((np.abs(have.data).sum(1) > 0).sum()) if have.shape[0] else 0}")
    pipe.cfg["upos_mask"] = bool(a.upos_mask)
    pipe.cfg["overwrite"] = True
    # ⚠ AND STRIP THE BUILD PATH FROM THE CONFIG. `tables` is a factory argument, so whatever was
    # passed at training time is serialised into the packaged config verbatim — a CWD-relative
    # path that means nothing on any other machine.
    # ⚠ `nlp.config` IS A PROPERTY THAT REBUILDS A FRESH Config EACH CALL, so assigning into it
    # mutates a temporary and does nothing. The live store is `get_pipe_config`, and the packaging
    # host-path guard is what caught the difference.
    nlp.get_pipe_config("tagger")["tables"] = None
    assert nlp.config["components"]["tagger"]["tables"] is None, "tables path not stripped"
    print(f"{a.out_model}: {nlp.pipe_names}")
    print(f"  factory={nlp.get_pipe_meta('tagger').factory}  fields={[len(f) for f in pipe.fields]}"
          f"  attested={len(pipe.attested)}  upos_mask={pipe.cfg['upos_mask']}"
          f"  field_weight={pipe.cfg.get('field_weight')}")
    nlp.to_disk(a.out_model)


if __name__ == "__main__":
    main()
