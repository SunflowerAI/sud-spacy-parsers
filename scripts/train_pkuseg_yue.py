#!/usr/bin/env python
"""Train a Cantonese word segmenter by fine-tuning the Mandarin pkuseg model.

SUD_Cantonese-HK is word-tokenised with no spaces, so raw-text segmentation is statistical (the
zh situation). The yue treebank is tiny (804 train sentences), so a pkuseg trained from scratch
on it is weak; instead we **warm-start from the dual-script Mandarin segmenter**
(`models/zh_gsdboth_pkuseg`, trained on the trad+simp GSD union, so it has seen traditional
characters — Cantonese-HK is traditional) and fine-tune on the yue gold tokens
(`spacy_pkuseg.train(..., init_model=...)`).

Reports word-level token-F1 on the yue test split for: the char tokeniser (current raw fallback),
jieba, raw Mandarin pkuseg (no fine-tune), pkuseg-from-scratch-on-yue, and the fine-tuned model
(± a yue word-type user dict). Run: .venv/bin/python scripts/train_pkuseg_yue.py
"""
import os, sys, glob, importlib.util

# reuse the zh helpers (gold_sents / write_utf8 / spans / token_f1)
_sz = importlib.util.spec_from_file_location("zp", "scripts/train_pkuseg_zh.py")
zp = importlib.util.module_from_spec(_sz); _sz.loader.exec_module(zp)

TREEBANK = "assets_yue/SUD_Cantonese-HK"
UTF8 = "corpus_yue_pkuseg"
ZH_INIT = "models/zh_gsdboth_pkuseg"          # dual-script Mandarin segmenter to fine-tune from
ITERS = 100


def find(split):
    return glob.glob(f"{TREEBANK}/yue_hk-sud-{split}.conllu")[0]


def char_f1(gold):
    """token-F1 of the one-char-per-token fallback (every multi-char gold word is missed)."""
    pred = [[c for c in "".join(t)] for t in gold]
    return zp.token_f1(pred, gold)


def main():
    os.makedirs(UTF8, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    train_c, dev_c, test_c = find("train"), find("dev"), find("test")
    train_u, dev_u = f"{UTF8}/train.utf8", f"{UTF8}/dev.utf8"
    zp.write_utf8(train_c, train_u)
    zp.write_utf8(dev_c, dev_u)

    import spacy_pkuseg as pkuseg
    scratch_dir = "models/yue_hk_pkuseg_scratch"
    finetune_dir = "models/yue_hk_pkuseg"
    print(f"training pkuseg from scratch on yue ({ITERS} iters) ...", file=sys.stderr)
    pkuseg.train(train_u, dev_u, scratch_dir, train_iter=ITERS)
    print(f"fine-tuning pkuseg from {ZH_INIT} on yue ({ITERS} iters) ...", file=sys.stderr)
    pkuseg.train(train_u, dev_u, finetune_dir, train_iter=ITERS, init_model=ZH_INIT)

    # yue multi-char word-type user dict (from train)
    vocab = sorted({t for s in zp.gold_sents(train_c) for t in s if len(t) > 1})
    udict = f"{UTF8}/userdict.txt"
    open(udict, "w", encoding="utf-8").write("\n".join(vocab) + "\n")

    import spacy
    gold = list(zp.gold_sents(test_c))
    raw = ["".join(t) for t in gold]

    def f1_for(seg, **init):
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": seg}}})
        if seg == "pkuseg":
            nlp.tokenizer.initialize(**init)
        return zp.token_f1([[t.text for t in nlp(r)] for r in raw], gold)

    print(f"\ntest sentences: {len(gold)}")
    print(f"token-F1  char (one-per-char fallback)   = {char_f1(gold):.4f}")
    print(f"token-F1  jieba                          = {f1_for('jieba'):.4f}")
    print(f"token-F1  Mandarin pkuseg (no fine-tune) = {f1_for('pkuseg', pkuseg_model=ZH_INIT):.4f}")
    print(f"token-F1  yue pkuseg from scratch        = {f1_for('pkuseg', pkuseg_model=scratch_dir):.4f}")
    print(f"token-F1  yue pkuseg fine-tuned          = {f1_for('pkuseg', pkuseg_model=finetune_dir):.4f}")
    print(f"token-F1  yue pkuseg fine-tuned+userdict = {f1_for('pkuseg', pkuseg_model=finetune_dir, pkuseg_user_dict=udict):.4f}")


if __name__ == "__main__":
    main()
