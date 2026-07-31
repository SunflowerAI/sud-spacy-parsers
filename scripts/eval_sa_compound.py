#!/usr/bin/env python3
"""Evaluate a Sanskrit arm with the tokeniser-supplied Compound=Yes feature present.

`spacy evaluate --gold-preproc` builds the predicted doc from gold words with the stock
`spacy.Corpus`, which never runs the tokeniser — so for an arm that reads MORPH as an INPUT
feature, the feature is simply absent and every component runs out of distribution. That is not a
measurement of the model, it is a measurement of the model with one of its inputs deleted.

This scores the same gold-preproc setup but builds the examples with `sud.CompoundCorpus.v1`, which
stamps Compound on the predicted doc exactly as `sa_tokenizer` does at inference.

Usage: eval_sa_compound.py <model-dir> <test.spacy> [--plain] [--out metrics.json]
       --plain uses the stock reader (no Compound), i.e. reproduces `spacy evaluate`.
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
import sa_tokenizer  # noqa: E402,F401  (registers the tokenizer + widened lex attrs)
import clause_parser  # noqa: E402,F401
from gold_tok_corpus import CompoundCorpus  # noqa: E402
from spacy.training.corpus import Corpus  # noqa: E402

model, test = sys.argv[1], sys.argv[2]
plain = "--plain" in sys.argv
out = None
if "--out" in sys.argv:
    out = sys.argv[sys.argv.index("--out") + 1]

nlp = spacy.load(model)
reader = (Corpus if plain else CompoundCorpus)(test, gold_preproc=True)
examples = list(reader(nlp))
scores = nlp.evaluate(examples)

keys = ["tag_acc", "pos_acc", "morph_acc", "lemma_acc", "dep_uas", "dep_las"]
print(f"{model}  ({'stock reader, no Compound' if plain else 'Compound supplied'})")
for k in keys:
    v = scores.get(k)
    if isinstance(v, float):
        print(f"  {k:10s} {v:.4f}")
per = scores.get("morph_per_feat") or {}
for f in ("Compound", "Case", "Number", "Gender", "Tense"):
    if f in per:
        print(f"    {f:9s} F {per[f]['f']:.3f}")
if out:
    pathlib.Path(out).write_text(json.dumps(
        {k: v for k, v in scores.items() if not callable(v)}, ensure_ascii=False, default=str))
