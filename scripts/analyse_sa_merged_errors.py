#!/usr/bin/env python3
"""Where a Sanskrit arm's errors sit: NON-PROJECTIVE arcs, and ADJECTIVAL AGREEMENT.

Two questions the headline LAS cannot answer, asked of any (model, corpus, reader) triple so the
clause-merged arms and the single-sentence ones can be put side by side.

NON-PROJECTIVITY. An arc (h, d) is non-projective iff some token strictly between h and d is not a
descendant of h. Reported as: what share of GOLD arcs are non-projective, and LAS restricted to
those arcs against LAS on the projective rest. Merging clauses into one tree is expected to RAISE
the non-projective share — the merge creates arcs that span a clause boundary — so the share is
reported for the corpus, not assumed constant across corpora.

⚠ Non-projectivity is a property of the GOLD arc, and the two corpora differ, so the non-projective
SUBSETS differ too. Comparing an arm's non-projective LAS on the merged corpus against another
arm's on the unmerged one compares two different question sets; only same-corpus rows are a
comparison.

AGREEMENT. `sud_constrained_parse.agreement_violation` is the definition of record (an adj-like
child of a NOUN under `mod` disagreeing in Case / Number / Gender, genitive children exempt). Two
numbers: how often the GOLD has such an arc — the irreducible floor, since the treebank contains
them — and how often the PREDICTION invents one. A model can only be judged against the floor.

⚠ Scored on PREDICTED morphology, which is what a decoder would have. Gold FEATS would flatter
every arm and measure a constraint nobody can enforce at inference.

    analyse_sa_merged_errors.py MODEL CORPUS.spacy [--reader norm|gold_tok_norm]
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401

import spacy  # noqa: E402
from gold_tok_corpus import GoldTokNormCorpus, NormCorpus  # noqa: E402
from sud_constrained_parse import agreement_violation  # noqa: E402


def nonprojective(heads):
    """Indices of the tokens whose arc to their head is non-projective (0-based, root excluded)."""
    n = len(heads)
    kids = collections.defaultdict(list)
    for c, h in enumerate(heads):
        if c != h:
            kids[h].append(c)

    desc = {}

    def descend(i):
        if i in desc:
            return desc[i]
        out = {i}
        desc[i] = out                       # guard against a cycle in a predicted tree
        for c in kids[i]:
            out |= descend(c)
        desc[i] = out
        return out

    for i in range(n):
        descend(i)
    bad = []
    for d, h in enumerate(heads):
        if d == h:
            continue
        lo, hi = (h, d) if h < d else (d, h)
        if any(k not in desc[h] for k in range(lo + 1, hi)):
            bad.append(d)
    return set(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("corpus")
    ap.add_argument("--reader", default="gold_tok_norm",
                    choices=["norm", "gold_tok_norm"])
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    reader = (NormCorpus(a.corpus, gold_preproc=True) if a.reader == "norm"
              else GoldTokNormCorpus(a.corpus))
    egs = list(reader(nlp))

    tot = collections.Counter()
    for eg in egs:
        ref = eg.reference
        pred = nlp(eg.predicted.copy())
        if len(pred) != len(ref):
            tot["skipped (length mismatch)"] += 1
            continue
        # arcs are compared per SENTENCE-free index: both docs are the same token sequence
        g_heads = [t.head.i for t in ref]
        p_heads = [t.head.i for t in pred]
        g_np = nonprojective(g_heads)
        for i in range(len(ref)):
            if g_heads[i] == i:
                continue
            key = "np" if i in g_np else "proj"
            tot[key] += 1
            if p_heads[i] == g_heads[i]:
                tot[key + "_uas"] += 1
                if pred[i].dep_ == ref[i].dep_:
                    tot[key + "_las"] += 1
        # agreement, on PREDICTED morphology in both cases
        for i, t in enumerate(pred):
            if t.dep_ == "mod" and t.head.i != i and agreement_violation(t, t.head):
                tot["pred_agree_viol"] += 1
            tot["pred_mod"] += 1 if t.dep_ == "mod" else 0
        for i, t in enumerate(ref):
            if t.dep_ == "mod" and t.head.i != i and agreement_violation(pred[i], pred[t.head.i]):
                tot["gold_agree_viol"] += 1
            tot["gold_mod"] += 1 if t.dep_ == "mod" else 0

    def pct(n, d):
        return f"{100.0 * n / d:.2f}" if d else "n/a"

    arcs = tot["np"] + tot["proj"]
    print(f"{a.model}  on  {a.corpus}   (reader: {a.reader}, {len(egs)} examples)")
    print(f"  arcs {arcs}   non-projective in GOLD {tot['np']} ({pct(tot['np'], arcs)} %)")
    print(f"  LAS  projective     {pct(tot['proj_las'], tot['proj'])}   "
          f"(UAS {pct(tot['proj_uas'], tot['proj'])})")
    print(f"  LAS  NON-projective {pct(tot['np_las'], tot['np'])}   "
          f"(UAS {pct(tot['np_uas'], tot['np'])})")
    print(f"  agreement-violating `mod` arcs: gold {tot['gold_agree_viol']}/{tot['gold_mod']} "
          f"({pct(tot['gold_agree_viol'], tot['gold_mod'])} %)   "
          f"predicted {tot['pred_agree_viol']}/{tot['pred_mod']} "
          f"({pct(tot['pred_agree_viol'], tot['pred_mod'])} %)")
    if tot["skipped (length mismatch)"]:
        print(f"  ⚠ skipped {tot['skipped (length mismatch)']} examples on length mismatch")


if __name__ == "__main__":
    main()
