#!/usr/bin/env python3
"""Is Literary Chinese word order rigid enough to CONSTRAIN the parser — and does it already obey?

Two questions, and the second is the one that decides. NEGATIVE-RESULTS.md records the Sanskrit
compound-internal-head constraint as a dead end for exactly this reason: the constraint was true,
and the parser already satisfied it, so enforcing it changed nothing. A constraint pays only where
the model VIOLATES it.

  1. RIGIDITY   per deprel, what share of gold arcs point left vs right? A relation at 99.9 % one
                way is a candidate constraint; one at 60/40 is a real choice the parser must make.
  2. VIOLATION  how often does the parser's own output break a rule that is (near-)exceptionless in
                gold? And when it does, is it WRONG there — i.e. would forbidding it help?

⚠ The population that matters is not "violations" but "violations that are also errors". A parser
that violates a 99.9 % rule on the 0.1 % of tokens where gold violates it too is right to.
"""
import argparse
import collections
import importlib.util
import pathlib


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="assets_lzh/SUD_Classical_Chinese-Kyoto/"
                    "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")
    ap.add_argument("--model", default="training_lzh_seg/model-best")
    ap.add_argument("--corpus", default="corpus_lzh_trad/lzh_kyoto-sud-test."
                                        "relabeled_ext.udep_ruled.punct.rulemerged.spacy")
    ap.add_argument("--threshold", type=float, default=0.99)
    a = ap.parse_args()

    # 1. rigidity, from the training gold
    d = collections.defaultdict(collections.Counter)
    for line in pathlib.Path(a.train).open(encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        h = int(f[6])
        if h == 0:
            continue
        d[f[7]]["L" if h < int(f[0]) else "R"] += 1
    print(f"{'deprel':<18}{'n':>8}{'left':>8}{'right':>8}{'dominant':>10}")
    rules = {}
    for rel, c in sorted(d.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n < 200:
            continue
        share = max(c["L"], c["R"]) / n
        side = "L" if c["L"] >= c["R"] else "R"
        mark = "  <- RULE" if share >= a.threshold else ""
        print(f"{rel:<18}{n:>8}{c['L']/n:>8.1%}{c['R']/n:>8.1%}{share:>9.1%}{mark}")
        if share >= a.threshold:
            rules[rel] = side
    print(f"\n{len(rules)} relations are >= {a.threshold:.0%} one-directional: {sorted(rules)}")

    # 2. does the parser violate them, and is it wrong when it does?
    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc, DocBin
    nlp = spacy.load(a.model)
    gold = list(DocBin().from_disk(a.corpus).get_docs(nlp.vocab))
    gold = [s.as_doc() for g in gold for s in g.sents]
    preds = [Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
             for g in gold]
    preds = list(nlp.pipe(preds, batch_size=64))
    viol = collections.Counter()
    viol_wrong = collections.Counter()
    gold_viol = collections.Counter()
    n_tok = 0
    for p, g in zip(preds, gold):
        for tp, tg in zip(p, g):
            if tg.dep_ == "punct":
                continue
            n_tok += 1
            if tg.head.i != tg.i and tg.dep_ in rules:
                side = "L" if tg.head.i < tg.i else "R"
                if side != rules[tg.dep_]:
                    gold_viol[tg.dep_] += 1
            if tp.head.i == tp.i or tp.dep_ not in rules:
                continue
            side = "L" if tp.head.i < tp.i else "R"
            if side != rules[tp.dep_]:
                viol[tp.dep_] += 1
                if tp.head.i != tg.head.i or tp.dep_ != tg.dep_:
                    viol_wrong[tp.dep_] += 1
    tv, tw = sum(viol.values()), sum(viol_wrong.values())
    print(f"\nover {n_tok} scored test tokens:")
    print(f"   parser output VIOLATING a rule      {tv:5d}  {tv/n_tok:.3%} of tokens")
    print(f"   ...and wrong there                  {tw:5d}  {tw/max(tv,1):.1%} of the violations")
    print(f"   GOLD itself violates a rule         {sum(gold_viol.values()):5d}"
          f"  {sum(gold_viol.values())/n_tok:.3%}")
    if viol:
        print("\n   by relation (parser violations / of which wrong / gold violations):")
        for rel, c in viol.most_common(8):
            print(f"      {rel:<16}{c:5d}{viol_wrong[rel]:8d}{gold_viol[rel]:9d}")


if __name__ == "__main__":
    main()
