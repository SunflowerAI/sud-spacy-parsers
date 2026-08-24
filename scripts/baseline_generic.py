#!/usr/bin/env python3
"""Trivial baselines for the zero-shot test languages, measured BEFORE any trained arm is quoted.

This repo has a standing rule about this and it was bought with a real mistake: an LLM clause-linking
classifier read 56.5 % accuracy and was reported as a result until the majority baseline turned out
to be 58.5 %. A raw LAS on an unseen language is exactly the same shape of number -- it looks like a
finding until a constant sits next to it.

Five baselines, all scored through the SAME `score()`, the same punctuation exclusion and the same
gold sentence boundaries as the arms, because a number from a different harness is not a comparison
(NEGATIVE-RESULTS.md: the lzh "7 LAS" and "+2.51 zh raw LAS" claims were both harness artefacts):

    left        every token attaches to its left neighbour, first token is root
    right       the mirror
    upos-pair   the most frequent (head direction, label) for each (head UPOS, dep UPOS), with the
                table estimated on the TRAINING sample only -- estimating it on the test language
                would be an oracle and would not be a baseline at all
    gold-head   gold attachment, most frequent label per UPOS pair. The ceiling on the LABELLING
                half alone, so a low LAS can be read as bad heads or bad labels rather than both
    typology    attach left if the language's OV bit is set, right if VO

⚠ **THE TYPOLOGY BASELINE IS THE ONE THAT MATTERS.** It is the typology channel with no parser at
all: the same eight bits, used by two lines of code. If it recovers most of what `g2_typ` recovers
over `g2_typ_der`, then the neural channel is an expensive way to express a head-direction prior and
the headline is much less interesting than it looks. Measuring it after the fact would be the kind
of check that gets skipped once the numbers already look good.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from spacy.training import Example  # noqa: E402

from eval_generic import score  # noqa: E402  (the SAME scorer the arms are read through)

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]


def load_docs(corpus_dir, lang, split, vocab):
    p = os.path.join(corpus_dir, f"{lang}-{split}.spacy")
    if not os.path.exists(p):
        return []
    return list(DocBin().from_disk(p).get_docs(vocab))


def sentences(docs):
    """One Doc per gold sentence. The arms are scored under gold boundaries, so these must be too."""
    return [s.as_doc(copy_user_data=False) for d in docs for s in d.sents]


def learn_tables(corpus_dir, langs, vocab):
    """(label per UPOS pair+direction, direction per UPOS pair, global majority label), from TRAIN."""
    pair_dir = collections.defaultdict(collections.Counter)
    pair_lab = collections.defaultdict(collections.Counter)
    overall = collections.Counter()
    for lang in langs:
        for doc in load_docs(corpus_dir, lang, "train", vocab):
            for t in doc:
                if t.dep_ in ("punct", "p") or t.head.i == t.i:
                    continue
                key = (t.head.pos_, t.pos_)
                pair_dir[key]["left" if t.head.i < t.i else "right"] += 1
                pair_lab[key][t.dep_] += 1
                overall[t.dep_] += 1
    return pair_lab, pair_dir, overall.most_common(1)[0][0]


def build(gold, heads, labels, vocab):
    """A predicted Doc with the given head indices and labels, aligned to `gold` token for token.

    Built through the constructor rather than by assignment: writing `is_sent_start` after the heads
    raises E043, and writing the heads alone leaves SENT_START unset, so `doc.sents` raises E030 in
    the scorer. Passing all three at once is the only order that gives a doc which is both parsed
    and segmented.
    """
    n = len(gold)
    return Doc(vocab,
               words=[t.text for t in gold],
               spaces=[bool(t.whitespace_) for t in gold],
               heads=list(heads),
               deps=[labels[i] if heads[i] != i else "root" for i in range(n)],
               sent_starts=[True] + [False] * (n - 1))


def chain_heads(n, direction):
    """A left- or right-branching chain. The first (or last) token is the root."""
    if direction == "left":
        return [max(i - 1, 0) for i in range(n)]
    return [min(i + 1, n - 1) if i < n - 1 else n - 1 for i in range(n)]


def run(gold_sents, vocab, kind, pair_lab, pair_dir, fallback, bits=None):
    examples = []
    for gold in gold_sents:
        n = len(gold)
        if n == 0:
            continue
        if kind == "left":
            heads = chain_heads(n, "left")
        elif kind == "right":
            heads = chain_heads(n, "right")
        elif kind == "typology":
            if bits is None:
                raise ValueError("the typology baseline needs the language's bits")
            # OV set and VO not -> head-final, so a dependent attaches to its RIGHT neighbour.
            ov, vo = bits[0], bits[1]
            heads = chain_heads(n, "right" if (ov and not vo) else "left")
        elif kind in ("upos-pair", "gold-head"):
            if kind == "gold-head":
                heads = [t.head.i for t in gold]
            else:
                heads = []
                for i, t in enumerate(gold):
                    # Direction from the training table for this UPOS pair, with the neighbour on
                    # that side as the head. No search, no scoring -- this is a baseline.
                    left = pair_dir.get((gold[max(i - 1, 0)].pos_, t.pos_), collections.Counter())
                    right = pair_dir.get((gold[min(i + 1, n - 1)].pos_, t.pos_),
                                         collections.Counter())
                    go_left = left.get("left", 0) >= right.get("right", 0)
                    heads.append(max(i - 1, 0) if go_left else min(i + 1, n - 1))
        else:
            raise ValueError(kind)
        labels = []
        for i, t in enumerate(gold):
            h = heads[i]
            lab = pair_lab.get((gold[h].pos_, t.pos_), collections.Counter())
            labels.append(lab.most_common(1)[0][0] if lab else fallback)
        pred = build(gold, heads, labels, vocab)
        examples.append(Example(pred, gold))
    return score(examples)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--typology", default="assets_typ/typology_v2.json")
    ap.add_argument("--json", default="metrics/generic_v2/baseline.json")
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    typ = json.loads(pathlib.Path(a.typology).read_text(encoding="utf-8"))["languages"]
    train = sorted(k for k, v in man.items() if v["pool"] == "train")
    test = sorted(k for k, v in man.items() if v["pool"] == "test")

    nlp = spacy.blank("xx")
    vocab = nlp.vocab
    print(f"estimating the UPOS-pair tables on {len(train)} TRAINING languages")
    pair_lab, pair_dir, fallback = learn_tables(a.corpus, train, vocab)
    print(f"  {len(pair_lab)} UPOS pairs seen; global majority label {fallback!r}")

    kinds = ["left", "right", "upos-pair", "gold-head", "typology"]
    out = {"meta": {"corpus": a.corpus, "n_train_langs": len(train), "n_test_langs": len(test),
                    "gold_sents": True, "fallback_label": fallback},
           "languages": {}}
    print(f"\n{'lang':6s} " + " ".join(f"{k:>11s}" for k in kinds) + "   tokens")
    macro = collections.defaultdict(list)
    for lang in test:
        gold = sentences(load_docs(a.corpus, lang, "test", vocab))
        if not gold:
            print(f"{lang:6s}  (no test docs)")
            continue
        bits = typ[lang]["bits"]
        row = {}
        for kind in kinds:
            row[kind] = run(gold, vocab, kind, pair_lab, pair_dir, fallback, bits)
            macro[kind].append(row[kind]["las"])
        out["languages"][lang] = {"bits": bits,
                                  "tokens": sum(len(s) for s in gold),
                                  **{k: {"uas": row[k]["uas"], "las": row[k]["las"]}
                                     for k in kinds}}
        print(f"{lang:6s} " + " ".join(f"{100 * row[k]['las']:11.2f}" for k in kinds)
              + f"   {sum(len(s) for s in gold):6d}")

    out["macro_las"] = {k: sum(v) / len(v) for k, v in macro.items() if v}
    print("\nMACRO LAS over test languages")
    for k in kinds:
        print(f"  {k:12s} {100 * out['macro_las'][k]:6.2f}")
    # `gold-head` is a CEILING, not a baseline -- it is handed the right answer for the head half --
    # so the bar the arms must clear is the best of the others.
    real = {k: v for k, v in out["macro_las"].items() if k != "gold-head"}
    best = max(real, key=lambda k: real[k])
    print(f"\nBar for the trained arms: {best} at {100 * real[best]:.2f} macro LAS.")
    print(f"Labelling ceiling (gold heads): {100 * out['macro_las']['gold-head']:.2f}.")
    out["bar"] = {"baseline": best, "macro_las": real[best]}

    pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
