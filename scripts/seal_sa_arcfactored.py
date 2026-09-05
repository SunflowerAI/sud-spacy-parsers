#!/usr/bin/env python
"""Graft `models/sa_arcfactored_full` into a copy of the released sa base, replacing `parser`
in-place with `sud.ArcFactoredParser.v1`, and SEAL every weight it reads (including the lemma-
vector table, a repo-relative path at build time) into the component's own saved bytes -- CLAUDE.md
hazard 4. Verify from a COPY under /tmp before trusting it loads at all (hazard 4's own lesson).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
import sud_arcfactored_parser  # noqa: F401 -- registers the "sud_arcfactored_parser" factory
import train_arcfactored as tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/sa_arcfactored_full")
    ap.add_argument("--base", default="")
    ap.add_argument("--out", default="training_sa_arcfactored_graft")
    a = ap.parse_args()
    ckpt = pathlib.Path(a.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    a.base = a.base or meta["src"]

    print(f"  base={a.base}  checkpoint={ckpt}  lang={meta['lang']}", flush=True)
    nlp = spacy.load(a.base)
    assert "parser" in nlp.pipe_names, f"{a.base}: no 'parser' pipe to replace"
    parser_idx = nlp.pipe_names.index("parser")
    before, after = nlp.pipe_names[:parser_idx], nlp.pipe_names[parser_idx + 1:]
    print(f"  replacing 'parser' at position {parser_idx}: {before} [parser] {after}", flush=True)

    nlp.remove_pipe("parser")
    # `add_pipe` runs the registered factory (an EMPTY shell -- see ArcFactoredParser's own
    # docstring) and does ALL of spaCy's bookkeeping correctly (nlp._components, nlp.config both
    # "nlp.pipeline" and "components"); populating that exact tracked object afterwards, rather
    # than hand-splicing pipeline internals, is the only part of this that needs to be careful.
    nlp.add_pipe("sud_arcfactored_parser", name="parser", before=after[0] if after else None)
    shell = nlp.get_pipe("parser")
    shell.from_disk(ckpt)
    if meta.get("lemvec") or meta.get("lemvec_dep"):
        idx, V = tr.load_lemvec_table(meta["lemvec_table"])
        shell._lemvec_idx, shell._lemvec_V = idx, V.astype("float32")
        print(f"  sealed lemvec table: {meta['lemvec_table']} ({len(idx)} lemmas, dim {V.shape[1]})",
              flush=True)

    assert nlp.pipe_names == before + ["parser"] + after, \
        f"pipeline order changed unexpectedly: {nlp.pipe_names}"

    out = pathlib.Path(a.out)
    nlp.to_disk(out)
    print(f"  saved -> {out}  pipe_names={nlp.pipe_names}", flush=True)


if __name__ == "__main__":
    main()
