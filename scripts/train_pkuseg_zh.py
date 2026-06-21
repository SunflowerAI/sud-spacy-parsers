#!/usr/bin/env python
"""Train a GSD-compatible pkuseg segmenter for Chinese, so spaCy's tokeniser
reproduces the treebank's word boundaries instead of jieba's.

  1. convert SUD_Chinese-GSD train/dev CoNLL-U -> space-segmented .utf8
  2. pkuseg.train(...) -> a model directory
  3. measure token-F1 of pkuseg-on-GSD vs gold on the test split, next to jieba

Run: .venv/bin/python scripts/train_pkuseg_zh.py
"""
import os, sys, argparse, glob

def gold_sents(path):
    """Yield lists of gold token forms (zh GSD has no MWT / no spaces)."""
    toks = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if line == "":
            if toks:
                yield toks
            toks = []
        else:
            c = line.split("\t")
            if "-" not in c[0] and "." not in c[0]:
                toks.append(c[1])
    if toks:
        yield toks

def write_utf8(conllu, out):
    with open(out, "w", encoding="utf-8") as f:
        for toks in gold_sents(conllu):
            f.write(" ".join(toks) + "\n")

def spans(toklist):
    out, p = [], 0
    for t in toklist:
        out.append((p, p + len(t)))
        p += len(t)
    return set(out)

def token_f1(pred_sents, gold_sents_):
    g = s = m = 0
    for pred, gold in zip(pred_sents, gold_sents_):
        gs, ps = spans(gold), spans(pred)
        m += len(gs & ps); g += len(gs); s += len(ps)
    P, R = (m / s if s else 0), (m / g if g else 0)
    return 2 * P * R / (P + R) if P + R else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treebank", default="assets_zh/SUD_Chinese-GSDSimp")
    ap.add_argument("--out", default="models/zh_gsdsimp_pkuseg")
    ap.add_argument("--utf8", default="corpus_zh_pkuseg_simp")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.utf8, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    def find(split):
        return glob.glob(f"{args.treebank}/*{split}.conllu")[0]
    train_c, dev_c, test_c = find("train"), find("dev"), find("test")
    train_u, dev_u = f"{args.utf8}/train.utf8", f"{args.utf8}/dev.utf8"
    write_utf8(train_c, train_u)
    write_utf8(dev_c, dev_u)

    import spacy_pkuseg as pkuseg
    print(f"training pkuseg on {args.treebank} ({args.iters} iters) ...", file=sys.stderr)
    pkuseg.train(train_u, dev_u, args.out, train_iter=args.iters)

    # user dict = multi-char word types from train
    vocab = sorted({t for s in gold_sents(train_c) for t in s if len(t) > 1})
    udict = f"{args.utf8}/userdict.txt"
    open(udict, "w", encoding="utf-8").write("\n".join(vocab) + "\n")

    import spacy
    gold = list(gold_sents(test_c))
    raw = ["".join(t) for t in gold]

    def f1_for(seg, **init):
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": seg}}})
        if seg == "pkuseg":
            nlp.tokenizer.initialize(**init)
        return token_f1([[t.text for t in nlp(r)] for r in raw], gold)

    print(f"\ntest sentences: {len(gold)}")
    print(f"token-F1  pkuseg-on-treebank            = {f1_for('pkuseg', pkuseg_model=args.out):.4f}")
    print(f"token-F1  pkuseg-on-treebank + userdict = {f1_for('pkuseg', pkuseg_model=args.out, pkuseg_user_dict=udict):.4f}")
    print(f"token-F1  jieba                         = {f1_for('jieba'):.4f}")

if __name__ == "__main__":
    main()
