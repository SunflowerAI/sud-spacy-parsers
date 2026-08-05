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
           "sa_tokenizer.py", "clause_parser.py", "sud_unsandhi.py", "sa_devanagari.py",
           # registers sud.MultiHashEmbedAffix.v1 (per-component affix windows; sa morph/lemma)
           "sud_affix_embed.py",
           # registers sud.CharSegTokenizer.v1 — the treebank-trained character segmenter used as
           # the TOKENIZER for zh (pkuseg 0.8385 -> 0.9202, the last +3 from jieba's segmentation
           # decision as an input channel) and id (enclitic split, 0.9985).
           # Must come after sa_presegment's dependencies; it imports that module lazily.
           "char_seg_tokenizer.py",
           # registers sud.SamplingCorpus.v1 — rebalance by SAMPLING rather than
           # duplicating docs, which inflates the parser's workload 10x
           "sampling_corpus.py",
           # sud_misc first: sud_tagger imports it. sud_shared_data holds the coordination
           # candidate mask, which sud_tagger looks up by name. (sud_idiom is packaging-time only
           # and needs no registration here, as with id_lemma_case_fix.)
           "sud_misc.py", "sud_shared_data.py", "sud_tagger.py"):
    _load(_f)
