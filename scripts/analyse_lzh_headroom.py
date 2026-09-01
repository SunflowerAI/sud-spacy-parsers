#!/usr/bin/env python3
"""Where is the lzh parser's remaining headroom? Four decompositions, cheapest diagnostics first.

Context: lzh is ATTACHMENT-bound (76.5 % of its errors are wrong heads, 23.5 % right-head-wrong-
label), three lexical channels have failed against controls, and a depth/width sweep moved nothing
outside seed noise. So the question is not "what other feature" but "what SHAPE do the wrong
attachments have".

  1. NON-PROJECTIVITY. spaCy's arc-eager transition parser has no swap action, so a non-projective
     gold arc is UNREACHABLE — a hard ceiling, not a training problem. Latin's docs record 63 % of
     its attachment errors sitting inside a non-projective sentence; nobody has measured lzh.
  2. ARC LENGTH and DIRECTION. A transition parser degrades with distance; if the errors are long
     arcs, that is a different problem from short ones.
  3. THE WRONG ROOTS. 679 tokens are given ROOT wrongly or denied it — sentence structure rather
     than word-level attachment, and the population `sent_join` is adjacent to.
  4. THE GOVERNOR'S IDENTITY. What is the parser attaching to instead?
"""
import argparse
import collections
import importlib.util
import pathlib


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def nonproj_arcs(heads):
    """Indices whose incoming arc crosses another, over one sentence's local heads."""
    bad = set()
    n = len(heads)
    for i in range(n):
        if heads[i] == i:
            continue
        a, b = sorted((i, heads[i]))
        for j in range(n):
            if heads[j] == j or j == i:
                continue
            c, d = sorted((j, heads[j]))
            if a < c < b < d or c < a < d < b:
                bad.add(i)
                break
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build_sud/lzh/lzh_sud_kyoto-0.2.0/lzh_sud_kyoto/"
                                       "lzh_sud_kyoto-0.2.0")
    ap.add_argument("--corpus", default="corpus_lzh_trad/lzh_kyoto-sud-test."
                                        "relabeled_ext.udep_ruled.punct.rulemerged.spacy")
    a = ap.parse_args()
    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc, DocBin

    nlp = spacy.load(a.model)
    docs = list(DocBin().from_disk(a.corpus).get_docs(nlp.vocab))
    gold = [s.as_doc() for d in docs for s in d.sents]
    preds = [Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
             for g in gold]
    preds = list(nlp.pipe(preds, batch_size=64))

    n = err = 0
    np_tok = np_err = np_sent_tok = np_sent_err = 0
    by_len = collections.defaultdict(lambda: [0, 0])
    root_conf = collections.Counter()
    gov_pos = collections.Counter()
    for g, p in zip(gold, preds):
        heads = [t.head.i - g[0].i for t in g]
        bad = nonproj_arcs(heads)
        sent_has_np = bool(bad)
        for k, (tg, tp) in enumerate(zip(g, p)):
            if tg.dep_ == "punct":
                continue
            n += 1
            wrong = tp.head.i != tg.head.i
            err += wrong
            if k in bad:
                np_tok += 1
                np_err += wrong
            if sent_has_np:
                np_sent_tok += 1
                np_sent_err += wrong
            d = abs(tg.head.i - tg.i) if tg.head.i != tg.i else 0
            b = "root" if d == 0 else ("1" if d == 1 else "2" if d == 2 else
                                       "3-5" if d <= 5 else "6-10" if d <= 10 else "11+")
            by_len[b][0] += 1
            by_len[b][1] += wrong
            if tg.head.i == tg.i and wrong:
                root_conf["gold ROOT, parser attached it"] += 1
            if tg.head.i != tg.i and tp.head.i == tp.i:
                root_conf["gold attached, parser made it ROOT"] += 1
            if wrong:
                gov_pos[(tg.head.pos_, tp.head.pos_)] += 1
    print(f"{n} scored tokens, {err} wrong heads ({err/n:.2%})\n")
    print("1. NON-PROJECTIVITY (a hard ceiling: arc-eager has no swap)")
    print(f"   gold arcs that are non-projective   {np_tok:6d}  {np_tok/n:6.2%} of tokens")
    print(f"   ...of which the parser gets wrong   {np_err:6d}  {np_err/max(np_tok,1):6.2%}"
          f"   (= {np_err/max(err,1):5.1%} of ALL errors)")
    print(f"   tokens in a sentence containing one {np_sent_tok:6d}  {np_sent_tok/n:6.2%}")
    print(f"   ...error rate there                 {np_sent_err/max(np_sent_tok,1):6.2%}"
          f"   vs {(err-np_sent_err)/max(n-np_sent_tok,1):6.2%} elsewhere")
    print("\n2. GOLD ARC LENGTH")
    for b in ("root", "1", "2", "3-5", "6-10", "11+"):
        t, w = by_len[b]
        if t:
            print(f"   {b:<6}{t:7d} tokens  {t/n:6.2%}   error {w/t:6.2%}"
                  f"   ({w/max(err,1):5.1%} of all errors)")
    print("\n3. ROOT confusions")
    for k, v in root_conf.most_common():
        print(f"   {k:<36}{v:6d}  {v/max(err,1):5.1%} of errors")
    print("\n4. governor UPOS: gold -> parser (top mis-attachments)")
    for (gp, pp), c in gov_pos.most_common(8):
        print(f"   {gp:<8} -> {pp:<8}{c:6d}")


if __name__ == "__main__":
    main()
