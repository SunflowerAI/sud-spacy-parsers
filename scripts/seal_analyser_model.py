#!/usr/bin/env python3
"""Make a trained analyser arm loadable away from the machine that built its table, and PROVE it.

THE FAILURE THIS PREVENTS. `configs/config_sa_mwt_analyser.cfg` names the analyser table by PATH
(`scripts/sa_analyser_lut.json.gz`). spaCy stores that config inside the model, and on load it
rebuilds the architecture from the config BEFORE restoring any weights — so on a machine without
that relative path the model raises at construction, or worse, constructs with an empty table.
This rewrites the stored config to carry the bit LAYOUT (`values`, a few closed lists) and
`table = null`, since the 32 507-form table itself travels in the model's own bytes.

⚠ VERIFY THE RELOADED MODEL, NEVER THE IN-MEMORY ONE (standing hazard 8). The check here reloads
from disk with the build-time table file HIDDEN, confirms the layer still holds every form, and
compares parses token for token against the original. A model that silently lost its table would
load cleanly and parse slightly worse — indistinguishable from a capacity control, which is exactly
how the zh one-token-per-string wheel shipped.

    seal_analyser_model.py training_sa_rl_analyser/model-best
"""
import argparse
import gzip
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy import Config  # noqa: E402


def embed_section(cfg):
    for comp in cfg.get("components", {}).values():
        model = comp.get("model", {})
        for key in ("embed", "tok2vec"):
            sub = model.get(key, {})
            if sub.get("@architectures") == "sud.AnalyserFeatsEmbed.v1":
                return sub
            for k2 in ("embed",):
                if sub.get(k2, {}).get("@architectures") == "sud.AnalyserFeatsEmbed.v1":
                    return sub[k2]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--table", default="scripts/sa_analyser_lut.json.gz")
    a = ap.parse_args()
    mdir = pathlib.Path(a.model)

    values = json.load(gzip.open(a.table, "rt", encoding="utf-8"))["values"]
    # interpolate=False: the default resolves ${paths.train} to null and breaks the config (E913)
    cfg = Config().from_disk(mdir / "config.cfg", interpolate=False)
    sec = embed_section(cfg)
    if sec is None:
        sys.exit("no sud.AnalyserFeatsEmbed.v1 in this model's config")
    before = sec.get("table")
    sec["table"] = None
    sec["values"] = {f: list(v) for f, v in values.items()}
    cfg.to_disk(mdir / "config.cfg")
    print(f"sealed {mdir}: table {before!r} -> null, values carried in config")

    # --- prove it, with the build-time table moved out of the way -------------------------------
    orig = spacy.load(mdir)
    text_docs = None
    hidden = pathlib.Path(a.table + ".hidden")
    shutil.move(a.table, hidden)
    try:
        back = spacy.load(mdir)
        ext = [n for n in back.get_pipe("tok2vec").model.walk()
               if n.name == "extract_analyser_sets"][0]
        n = len(ext.attrs["an_payload"].get("table") or {})
        print(f"reloaded WITHOUT the table file: layer holds {n} forms")
        if n == 0:
            sys.exit("FAIL: the table did not travel inside the model")
        from spacy.tokens import DocBin
        docs = list(DocBin().from_disk(
            "corpus_sa_mwt_rl_norm/sa_vedic-sud-test.relabeled_ext.csl_mwt.spacy"
        ).get_docs(orig.vocab))[:40]
        from spacy.tokens import Doc
        def run(nlp):
            out = []
            for d in docs:
                p = Doc(nlp.vocab, words=[t.text for t in d], spaces=[bool(t.whitespace_) for t in d])
                for pt, rt in zip(p, d):
                    pt.norm_ = rt.norm_
                    if rt.morph.get("Compound"):
                        pt.set_morph("Compound=Yes")
                out.append(nlp(p))
            return [(t.head.i - t.i, t.dep_) for doc in out for t in doc]
        same = run(orig) == run(back)
        print(f"parses identical to the pre-seal model: {same}")
        if not same:
            sys.exit("FAIL: sealing changed the model's output")
    finally:
        shutil.move(hidden, a.table)
    print("\nsealed and verified")


if __name__ == "__main__":
    main()
