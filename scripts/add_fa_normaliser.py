#!/usr/bin/env python3
"""Swap `sud.FaNormTokenizer.v1` into a built Persian arm.

Post hoc, not a retrain: training reads through `sud.GoldTokCorpus.v1` under `gold_preproc`, so the
parser is segmenter-agnostic and every component's weights come out byte-identical -- `--verify`
checks that rather than trusting it.

⚠ ASSIGNING `nlp.tokenizer` DOES NOT UPDATE THE CONFIG. `to_disk` writes the config as it stands,
so the reloaded model rebuilds a stock `spacy.Tokenizer.v1` and `from_disk` quietly refills it with
the base rules -- it loads, runs, normalises NOTHING, and says nothing. `nlp.config["nlp"]
["tokenizer"]` must be set too, and `--verify` re-checks the RELOADED model rather than the
in-memory one. This is the trap `add_la_enclitic_tokenizer.py` was written to record.
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_code(path, required=True):
    path = str(path if "/" in str(path) else _HERE / path)
    name = path.split("/")[-1][:-3]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        if required:
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--code", default="")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    load_code(_HERE / "fa_normalise.py")
    for extra in filter(None, (c.strip() for c in a.code.split(","))):
        load_code(extra, required=False)

    import spacy
    import fa_normalise
    nlp = spacy.load(a.in_model)
    nlp.tokenizer = fa_normalise.make_fa_norm_tokenizer()(nlp)
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.FaNormTokenizer.v1"}
    out = Path(a.out_model)
    if out.exists():
        shutil.rmtree(out)
    nlp.to_disk(out)
    print(f"wrote {out}")

    if a.verify:
        back = spacy.load(out)
        probe = "وزير خارجه گفت كه دولت اين طرح را بررسي مي‌كند."
        doc = back(probe)
        if doc.text == probe or "ي" in doc.text or "ك" in doc.text:
            raise SystemExit("RELOADED model did not normalise -- the config did not take")
        print(f"  reloaded normalises: {doc.text}")
        print(f"  source kept: {doc.user_data.get('fa_source_text') == probe}")
        moved = []
        for f in sorted(Path(a.in_model).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(a.in_model)
            g = out / rel
            if not g.exists() or g.read_bytes() != f.read_bytes():
                moved.append(str(rel))
        weights = [m for m in moved if m.endswith("model")]
        print(f"  files differing: {len(moved)}  (weight files: {len(weights)})")
        for m in moved:
            print("    ", m)
        if weights:
            raise SystemExit("a weight file moved -- this was supposed to be pure surgery")


if __name__ == "__main__":
    main()
