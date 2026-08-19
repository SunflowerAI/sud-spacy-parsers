#!/usr/bin/env python3
"""Evaluate a ja arm with the tokeniser-supplied ``Inflection`` feature present.

`spacy evaluate --gold-preproc` builds the predicted doc from gold words with the stock
`spacy.Corpus`, which never runs the tokeniser -- so for an arm that reads ``Inflection`` as an
INPUT feature, the feature is simply absent and the tagger runs out of distribution. That is not a
measurement of the model, it is a measurement of the model with one of its inputs deleted. Exactly
the reason `eval_sa_compound.py` exists for sa's ``Compound``.

This scores the same gold-preproc setup but builds the examples with `sud.InflEvalCorpus.v1`, which
stamps ``Inflection`` on the predicted doc exactly as `spacy.ja.JapaneseTokenizer` does at
inference. The corpus must have been through `stamp_ja_inflection.py`.

An arm must be scored through the SAME reader it was trained through. `--plain` (stock reader, no
Inflection) is offered for one purpose: measuring how far an Inflection-conditioned arm falls when
the channel is deleted, which is what an unwitting `spacy evaluate --gold-preproc` would report.

Usage: eval_ja_infl.py <model-dir> <test.spacy> [--plain] [--out metrics.json] [--label NAME]
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
import seg_code  # noqa: E402,F401  (registers the readers, tokenisers and factories)
from gold_tok_corpus import InflEvalCorpus, InflTagEvalCorpus  # noqa: E402
from spacy.training.corpus import Corpus  # noqa: E402

if len(sys.argv) < 3:
    sys.exit(__doc__.strip().splitlines()[-1])
model, test = sys.argv[1], sys.argv[2]
plain = "--plain" in sys.argv
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else model

READERS = {"infl": InflEvalCorpus, "infltag": InflTagEvalCorpus, "plain": Corpus}
which = "plain" if plain else "infl"
if "--reader" in sys.argv:
    which = sys.argv[sys.argv.index("--reader") + 1]
    if which not in READERS:
        sys.exit(f"--reader must be one of {sorted(READERS)}, not {which!r}")
# An arm must be scored through the reader it was TRAINED through. Scoring an infltag arm through
# `infl` deletes only the tag channel, which is worse than deleting both: it still looks like it ran.
nlp = spacy.load(model)
reader = READERS[which](test, gold_preproc=True)
examples = list(reader(nlp))

# Report the channel's coverage in what was actually scored. A silent drop to 0 % is the whole
# failure this script exists to prevent, so it is printed on every run rather than assumed.
have = sum(1 for eg in examples for t in eg.predicted if t.morph.get("Inflection"))
tot = sum(len(eg.predicted) for eg in examples)

scores = nlp.evaluate(examples)
print(f"{label}  (reader: {which})")
print(f"  Inflection on predicted   {have}/{tot} = {have / max(tot, 1):.2%}")
for k in ("tag_acc", "pos_acc", "morph_acc", "lemma_acc", "dep_uas", "dep_las"):
    v = scores.get(k)
    if isinstance(v, float):
        print(f"  {k:10s} {v:.4f}")
if out:
    pathlib.Path(out).write_text(json.dumps(
        {k: v for k, v in scores.items() if not callable(v)}, ensure_ascii=False, default=str))
