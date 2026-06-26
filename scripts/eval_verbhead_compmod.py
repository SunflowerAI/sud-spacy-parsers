#!/usr/bin/env python3
"""How well does each trained parser distinguish comp:obl from mod on VERB-headed nodes?

Runs each model on gold tokens (gold-preproc), restricts to gold tokens whose gold head is a
VERB and whose gold label is comp:obl or mod (the disambiguation target), and scores the
predicted label there. Isolates the verb-head decision from the noun-head dilution that drags
down the headline comp:obl F of the extended models. Label-only (gold structure defines the
subset; predicted label compared regardless of predicted attachment)."""
import glob
import sys

import spacy
from spacy.tokens import Doc, DocBin

sys.path.insert(0, "scripts")
import lzh_tokenizer  # noqa: F401  (registers lzh language + char tokeniser)
import ar_tokenizer   # noqa: F401  (registers CAMeL tokeniser; lazy — no camel import here)

ARMS = [
    ("en  ext", "training_en_ewt_ext", "corpus_en_ewt_ext/*-test.relabeled_ext.spacy"),
    ("fa  verb-rl", "training_fa_rl", "corpus_fa_rl/*-test.relabeled.spacy"),
    ("fa  ext", "training_fa_ext", "corpus_fa_ext/*-test.relabeled_ext.spacy"),
    ("ar  verb-rl", "training_ar_rl", "corpus_ar_rl/*-test.relabeled.spacy"),
    ("ar  ext", "training_ar_ext", "corpus_ar_ext/*-test.relabeled_ext.spacy"),
    ("la  verb-rl", "training_la_rl", "corpus_la_rl/*-test.relabeled.spacy"),
    ("la  ext", "training_la_ext", "corpus_la_ext/*-test.relabeled_ext.spacy"),
    ("ja  verb-rl", "training_ja_rl", "corpus_ja_rl/*-test.relabeled.spacy"),
    ("ja  ext", "training_ja_ext", "corpus_ja_ext/*-test.relabeled_ext.spacy"),
    ("lzh verb-rl", "training_lzh_rl", "corpus_lzh_rl/*-test.relabeled.spacy"),
    ("lzh ext", "training_lzh_ext", "corpus_lzh_ext/*-test.relabeled_ext.spacy"),
    ("sa  ext", "training_sa_ext", "corpus_sa_ext/*-test.relabeled_ext.spacy"),
]


def base(dep):
    return dep.split("@")[0]


def run(model_dir, corpus_glob):
    nlp = spacy.load(model_dir + "/model-best")
    pipes = [nlp.get_pipe(n) for n in nlp.pipe_names]
    golds = list(DocBin().from_disk(glob.glob(corpus_glob)[0]).get_docs(nlp.vocab))
    # confusion on gold VERB-headed {comp:obl, mod} nodes
    tp = fp = fn = correct = total = 0
    for g in golds:
        d = Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
        for p in pipes:
            d = p(d)
        for i, t in enumerate(g):
            if t.head.i == i or t.head.pos_ != "VERB":
                continue
            gl = base(t.dep_)
            if gl not in ("comp:obl", "mod"):
                continue
            pl = base(d[i].dep_)
            total += 1
            if pl == gl:
                correct += 1
            if pl == "comp:obl" and gl == "comp:obl":
                tp += 1
            elif pl == "comp:obl" and gl != "comp:obl":
                fp += 1
            elif pl != "comp:obl" and gl == "comp:obl":
                fn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return total, correct / total if total else 0.0, P, R, F


print(f"{'arm':12} {'N':>5} {'acc':>6} {'comp:obl P':>11} {'R':>6} {'F':>6}")
for label, mdir, cg in ARMS:
    try:
        n, acc, P, R, F = run(mdir, cg)
        print(f"{label:12} {n:5d} {acc:6.3f} {P:11.3f} {R:6.3f} {F:6.3f}")
    except Exception as e:
        print(f"{label:12} ERROR {type(e).__name__}: {str(e)[:50]}")
