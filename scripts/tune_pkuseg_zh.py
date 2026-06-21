#!/usr/bin/env python
"""Push pkuseg-on-GSD higher: user dictionary of treebank word types,
more iterations, and fine-tuning from pkuseg's pretrained model."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from train_pkuseg_zh import gold_sents, token_f1, GSD, UTF8

TRAIN = f"{GSD}/zh_gsd-sud-train.conllu"
DEV   = f"{UTF8}/dev.utf8"
TRAINU = f"{UTF8}/train.utf8"
TEST  = f"{GSD}/zh_gsd-sud-test.conllu"

def main():
    import spacy, spacy_pkuseg as pkuseg

    gold = list(gold_sents(TEST))
    raw = ["".join(t) for t in gold]

    # user dict = multi-char word types from train
    vocab = sorted({t for s in gold_sents(TRAIN) for t in s if len(t) > 1})
    udict = f"{UTF8}/gsd_userdict.txt"
    open(udict, "w", encoding="utf-8").write("\n".join(vocab) + "\n")
    print(f"user dict: {len(vocab)} word types", file=sys.stderr)

    def evaluate(model, user_dict=None, label=""):
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "pkuseg"}}})
        if user_dict:
            nlp.tokenizer.initialize(pkuseg_model=model, pkuseg_user_dict=user_dict)
        else:
            nlp.tokenizer.initialize(pkuseg_model=model)
        pred = [[t.text for t in nlp(r)] for r in raw]
        print(f"  token-F1 {label:38s} = {token_f1(pred, gold):.4f}")

    print("== existing 20-iter model ==")
    evaluate("models/zh_gsd_pkuseg", None, "pkuseg(20)")
    evaluate("models/zh_gsd_pkuseg", udict, "pkuseg(20) + GSD user dict")

    print("== retrain 100 iters ==", file=sys.stderr)
    pkuseg.train(TRAINU, DEV, "models/zh_gsd_pkuseg_100", train_iter=100)
    evaluate("models/zh_gsd_pkuseg_100", None, "pkuseg(100)")
    evaluate("models/zh_gsd_pkuseg_100", udict, "pkuseg(100) + GSD user dict")

if __name__ == "__main__":
    main()
