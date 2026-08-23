#!/usr/bin/env python3
"""Evaluate a Sanskrit arm with the tokeniser-supplied Compound=Yes feature present.

`spacy evaluate --gold-preproc` builds the predicted doc from gold words with the stock
`spacy.Corpus`, which never runs the tokeniser — so for an arm that reads MORPH as an INPUT
feature, the feature is simply absent and every component runs out of distribution. That is not a
measurement of the model, it is a measurement of the model with one of its inputs deleted.

This scores the same gold-preproc setup but builds the examples with `sud.CompoundCorpus.v1`, which
stamps Compound on the predicted doc exactly as `sa_tokenizer` does at inference.

Usage: eval_sa_compound.py <model-dir> <test.spacy> [--plain|--reader NAME] [--out metrics/en/metrics.json]
       --plain      uses the stock reader (no Compound), i.e. reproduces `spacy evaluate`.
       --reader     compound (default) | norm | oracle — an arm must be scored through the SAME
                    reader it was trained through, or it runs with one of its inputs deleted, which
                    is the whole reason this script exists. `norm` additionally supplies the
                    sandhi-reversed NORM; `oracle` additionally supplies gold LEMMA and gold FEATS
                    and is therefore not a measurement of anything shippable.
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
# Import the AGGREGATOR, not a hand-picked list. This file has now died with E893 twice in one day
# — once for sud.MultiHashEmbedFeats.v1 and once for sud.LemmaVecEmbed.v1 — each time AFTER the
# training run it was meant to score, because `spacy train` gets `--code scripts/seg_code.py` and
# this script kept its own shorter list. A list that has to be remembered per script is a list that
# gets missed; that is the reason seg_code.py exists at all.
import seg_code  # noqa: E402,F401
from gold_tok_corpus import (CompoundCorpus, GoldTokNormCorpus, LemmaOracleCorpus,  # noqa: E402
                             NormCorpus, OracleCorpus)
from spacy.training.corpus import Corpus  # noqa: E402

model, test = sys.argv[1], sys.argv[2]
plain = "--plain" in sys.argv
out = None
if "--out" in sys.argv:
    out = sys.argv[sys.argv.index("--out") + 1]
READERS = {"compound": CompoundCorpus, "norm": NormCorpus, "oracle": OracleCorpus,
           "lemma_oracle": LemmaOracleCorpus, "gold_tok_norm": GoldTokNormCorpus}
# `gold_tok_norm` takes no gold_preproc: it always yields whole multi-sentence docs, which is the
# point of it — pass it to score an arm trained through the same reader, or SENTS_F is a free 100.
NO_GOLD_PREPROC = {"gold_tok_norm"}
which = "plain" if plain else "compound"
if "--reader" in sys.argv:
    which = sys.argv[sys.argv.index("--reader") + 1]
    if which not in READERS:
        sys.exit(f"--reader must be one of {sorted(READERS)}, not {which!r}")

nlp = spacy.load(model)
kw = {} if which in NO_GOLD_PREPROC and not plain else {"gold_preproc": True}
reader = (Corpus if plain else READERS[which])(test, **kw)
examples = list(reader(nlp))
scores = nlp.evaluate(examples)

keys = ["tag_acc", "pos_acc", "morph_acc", "lemma_acc", "dep_uas", "dep_las", "sents_f"]
print(f"{model}  (reader: {'stock, no Compound' if plain else which})")
for k in keys:
    v = scores.get(k)
    if isinstance(v, float):
        print(f"  {k:10s} {v:.4f}")
per = scores.get("morph_per_feat") or {}
# every feature, commonest first — Voice / VerbForm / Mood used to be omitted, and they are exactly
# where a longer suffix window is predicted to move things (passive -yate, -mānaḥ, future -ṣyati).
for f in sorted(per, key=lambda k: -per[k].get("f", 0)):
    d = per[f]
    print(f"    {f:9s} P {d['p']:.3f}  R {d['r']:.3f}  F {d['f']:.3f}")
if "morph_micro_f" in scores:
    print(f"  morph_micro_f {scores['morph_micro_f']:.4f}")
if out:
    pathlib.Path(out).write_text(json.dumps(
        {k: v for k, v in scores.items() if not callable(v)}, ensure_ascii=False, default=str))
