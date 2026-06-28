#!/usr/bin/env python3
"""Single ``--code`` entry point for the sentence-segmentation retraining.

``spacy train --code`` accepts only ONE file (unlike ``spacy package``, which splits on commas),
so this module imports everything the per-language seg configs need: the gold-token multi-sentence
reader (required — segmentation training depends on it) plus every custom tokenizer/factory the
configs reference. The custom tokenizers are loaded best-effort so a language whose optional deps
are absent doesn't block the others; the reader is imported directly so its absence fails loudly.
"""
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gold_tok_corpus  # noqa: E402,F401  (registers sud.GoldTokCorpus.v1 — required)


def _load(fname):
    path = _HERE / fname
    if not path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # a language without its extra deps still loads its own tokenizer
        print(f"seg_code: skipped {fname}: {type(e).__name__}: {e}")


for _f in ("ar_tokenizer.py", "yue_tokenizer.py", "lzh_tokenizer.py",
           "sa_tokenizer.py", "clause_parser.py"):
    _load(_f)
