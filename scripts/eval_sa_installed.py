#!/usr/bin/env python
"""Score an INSTALLED sa wheel through its full deployed pipeline.

WHY NOT `eval_sa_compound.py`. That script imports `seg_code`, which registers `sa_compound` and
friends for arms loaded from a DIRECTORY. An installed wheel bundles its own copy of that code and
registers the same factories from its package `__init__`, so importing both raises E004 ("a factory
for 'sa_compound' already exists"). This module therefore imports ONLY the corpus reader it needs
and lets the wheel register everything else -- which is also exactly the code path a user gets.

⚠ Uses `sud.CompoundCorpus.v1`, not the stock reader. `spacy evaluate --gold-preproc` builds the
predicted doc without ever running the tokeniser, so an arm that reads MORPH as an INPUT feature is
scored with one of its inputs deleted. That is not a measurement of the model.

    eval_sa_installed.py <installed-package-name> <test.spacy> [--out metrics/sa/x.json]
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spacy
from gold_tok_corpus import CompoundCorpus  # registers sud.CompoundCorpus.v1, nothing else

pkg, test = sys.argv[1], sys.argv[2]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None

nlp = spacy.load(pkg)
print("%s  pipeline=%s" % (pkg, nlp.pipe_names))
examples = list(CompoundCorpus(test, gold_preproc=True)(nlp))
scores = nlp.evaluate(examples)
for k in ("tag_acc", "pos_acc", "morph_acc", "lemma_acc", "sents_f"):
    v = scores.get(k)
    if isinstance(v, float):
        print("  %-10s %.4f" % (k, v))

# ⚠ `nlp.evaluate` returns NO dep_uas/dep_las for this arm and does not complain: the arc-factored
# pipe defines no `score` method, so the scorer simply has nothing to call and the keys are absent.
# Reading that as "the parser scored zero" or not noticing at all are both easy; compute them here.
# ⚠ Call each component on the Doc rather than `proc.pipe(docs)`: this pipeline mixes trainable
# pipes with plain function components (SaCompound has no `.pipe`), and every spaCy component is
# callable on a single Doc whereas only some expose a batched `pipe`.
docs = []
for _eg in examples:
    _d = _eg.predicted.copy()
    for _name, proc in nlp.pipeline:
        _d = proc(_d)
    docs.append(_d)
uas = las = n = 0
for doc, eg in zip(docs, examples):
    ref = eg.reference
    if len(doc) != len(ref):
        continue                      # gold_preproc keeps tokenisation 1:1; skip any that drifted
    for t, g in zip(doc, ref):
        if g.dep_ in ("", "-"):
            continue
        n += 1
        hit = (t.head.i - t.i) == (g.head.i - g.i)
        uas += hit
        las += hit and t.dep_.lower() == g.dep_.lower()
print("  %-10s %.4f" % ("dep_uas", uas / max(n, 1)))
print("  %-10s %.4f" % ("dep_las", las / max(n, 1)))
print("  scored %d tokens over %d sentences" % (n, len(docs)))
if out:
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(
        {k: v for k, v in scores.items() if not callable(v)}, ensure_ascii=False, default=str))
