#!/usr/bin/env python
"""Benchmark a binary LLM decision on the clause-linking residue — against real gold.

The residue that `cross_unit_rules.py` leaves at unit boundaries has no gold, so it cannot be used
to judge anything. But the residue has an in-unit ANALOGUE that does: clause-to-clause links inside
a 句讀 unit whose signature no derived rule covers. Those are annotated. So the benchmark is built
there, on dev+test (the rules are derived from train), and the LLM is scored against the
annotators' own decisions.

The question is deliberately BINARY, following the comp/mod pipeline that worked rather than the
constrained multi-way relabel that did not (NEGATIVE-RESULTS: 21 353 decisions, none applied,
36.4 % agreement between two passes). `mod`, `subj` and `unk` are excluded so the two classes are
genuinely exclusive:

    complement   comp:obj / comp:obl / comp:pred  — B is an argument of A's verb
    independent  parataxis / conj:coord           — B is a separate following clause

**Read the result as a lower bound on difficulty, not as cross-unit accuracy.** In-unit links are
not a random sample of cross-unit ones — an editor's mark selects against tight complementation —
so this measures whether the model can do the task at all, on the only data where it can be marked.
A model that cannot beat the majority baseline HERE will not do better at a unit boundary.

Usage:
    cross_unit_bench.py [-n 200] [--seed 0] [--model qwen3:8b] [--no-llm]
"""
import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disambiguate_pp  # noqa: E402
from cross_unit_rules import INTRA, harvest, read, signatures, subtree_first  # noqa: E402

COMPLEMENT = {"comp:obj", "comp:obl", "comp:pred"}
INDEPENDENT = {"parataxis", "conj:coord"}
BASE = "assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-%s.relabeled_ext.udep_ruled.conllu"

PREFIX = """You are annotating the syntax of Classical Chinese (文言文).

You will see two clauses from one sentence, A then B, where B follows A.
Decide how B relates to the verb of A.

complement  = B is an ARGUMENT of A's verb: what is said, thought, known, ordered, asked or
              perceived; the state of affairs that A's verb takes as its object.
independent = B is a SEPARATE following clause: a further action in sequence, a coordinate
              statement, a consequence. A's verb does not take B as an argument.

Answer with exactly one word: complement or independent.

"""


def clause_text(toks, ids):
    return "".join(t[1] for t in toks if t[0] in ids)


def build(paths, rules):
    """Uncovered in-unit clause-to-clause links, with gold, rendered as (A, B, label)."""
    out = []
    for path in paths:
        for _, toks in read(path):
            idx = {t[0]: t for t in toks}
            kids = collections.defaultdict(list)
            for t in toks:
                if t[6] != "0":
                    kids[t[6]].append(t)
            for t in toks:
                if t[6] == "0" or t[7] in INTRA:
                    continue
                h = idx.get(t[6])
                if h is None or t[3] not in ("VERB", "AUX") or h[3] not in ("VERB", "AUX"):
                    continue
                if int(t[0]) < int(h[0]):
                    continue
                label = ("complement" if t[7] in COMPLEMENT else
                         "independent" if t[7] in INDEPENDENT else None)
                if label is None:
                    continue
                first = subtree_first(t, toks, kids)
                if any(sig in rules for sig in signatures(first, t, h)):
                    continue                      # a rule already covers this — not residue
                sub, fr = {t[0]}, [t[0]]
                while fr:
                    nx = []
                    for k in fr:
                        for c in kids.get(k, []):
                            sub.add(c[0]); nx.append(c[0])
                    fr = nx
                a = clause_text(toks, {t[0] for t in toks} - sub)
                b = clause_text(toks, sub)
                if a and b:
                    out.append((a, b, label, t[7]))
    return out


def few_shot(items, k=6):
    """Contrastive shots: alternate the two labels so neither is presented as the default."""
    comp = [x for x in items if x[2] == "complement"]
    ind = [x for x in items if x[2] == "independent"]
    shots = []
    for i in range(k // 2):
        if i < len(comp):
            shots.append(comp[i])
        if i < len(ind):
            shots.append(ind[i])
    return "".join(f"A: {a}\nB: {b}\nAnswer: {lab}\n\n" for a, b, lab, _ in shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    rules, _ = harvest(read(BASE % "train"), 0.90, 20)
    train_items = build([BASE % "train"], rules)
    bench = build([BASE % "dev"], rules) + build([BASE % "test"], rules)
    dist = collections.Counter(x[2] for x in bench)
    maj, majn = dist.most_common(1)[0]
    print(f"benchmark: {len(bench)} uncovered in-unit links (dev+test), "
          f"train pool {len(train_items)}")
    print(f"gold: " + ", ".join(f"{k} {v}" for k, v in dist.most_common()))
    print(f"MAJORITY BASELINE ({maj}): {100*majn/len(bench):.1f} %\n")
    print("gold deprels behind the two classes:")
    for k, v in collections.Counter(x[3] for x in bench).most_common(6):
        print(f"   {k:<14} {v}")

    if args.no_llm:
        return
    random.seed(args.seed)
    sample = random.sample(bench, min(args.n, len(bench)))
    prompt_prefix = PREFIX + few_shot(train_items)
    disambiguate_pp.MODEL = args.model
    print(f"\nquerying {args.model} on {len(sample)} items "
          f"(static prefix {len(prompt_prefix)} chars, cached by Ollama)…")

    hit = 0
    conf = collections.Counter()
    for i, (a, b, gold, _) in enumerate(sample, 1):
        ans = disambiguate_pp.query(f"{prompt_prefix}A: {a}\nB: {b}\nAnswer:")
        pred = "complement" if ans.startswith("comp") else \
               "independent" if ans.startswith("ind") else "?"
        conf[(gold, pred)] += 1
        hit += pred == gold
        if i % 50 == 0:
            print(f"   {i}/{len(sample)}  running accuracy {100*hit/i:.1f} %")
    smaj = collections.Counter(g for g, _, _, _ in
                               [(x[2], 0, 0, 0) for x in sample]).most_common(1)[0][1]
    print(f"\nLLM accuracy      {100*hit/len(sample):.1f} %")
    print(f"majority on sample {100*smaj/len(sample):.1f} %")
    print("\nconfusion (gold -> predicted):")
    for (g, p), v in conf.most_common():
        print(f"   {g:<12} -> {p:<12} {v}")


if __name__ == "__main__":
    main()
