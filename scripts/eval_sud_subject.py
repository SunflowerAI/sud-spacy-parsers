#!/usr/bin/env python3
"""Compare the trained `sud_tagger` against the `sud_subject_rule` frame table, end to end.

Neither approach dominates across languages, so the choice has to be measured per language --
and measured on the same footing, which is the point of this script. The two are NOT comparable
on gold trees: the rule reads the parser's deprel and the head's predicted UPOS, so its accuracy
falls with parse quality, while the trained pipe reads only surface forms through its own encoder.
Both are therefore run over gold TOKENS with everything else predicted.

    scripts/eval_sud_subject.py --lang lzh
    scripts/eval_sud_subject.py --lang en --split dev

Reports P/R/F for each arm over tokens that carry `Subject=` in the treebank.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import seg_code  # noqa: E402,F401  (custom tokenisers + sud_tagger)
import sud_subject_rule  # noqa: E402,F401  (registers the rule factory)
from sud_misc import get_misc  # noqa: E402

TEST = {
    "en":  "assets/en_ewt-sud-{split}.relabeled_ext.conllu",
    "zh":  "assets_zh/SUD_Chinese-GSDBoth/zh_gsdboth-sud-{split}.relabeled_ext.conllu",
    "yue": "assets_yue/SUD_Cantonese-HK/yue_hk-sud-{split}.relabeled_ext.conllu",
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-{split}.relabeled_ext.conllu",
    "fa":  "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-{split}.relabeled_ext.conllu",
    "la":  "assets_la/la_ittbproiel-sud-{split}.relabeled_ext.conllu",
    "sa":  "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.csl_rev.conllu",
}
LEMMA_ARM = {"sa": "training_sa_lemma3_noannot/model-best"}


def sentences(path):
    block = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if block:
                yield block
            block = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        block.append(f)
    if block:
        yield block


def gold_subject(fields):
    for item in fields[9].split("|"):
        if item.startswith("Subject="):
            return item.split("=", 1)[1]
    return None


def score(nlp, rows_iter):
    tp = fp = fn = skipped = 0
    for rows in rows_iter:
        doc = Doc(nlp.vocab, words=[f[1] or "_" for f in rows])
        doc = nlp(doc)
        if len(doc) != len(rows):
            skipped += 1
            continue
        for tok, f in zip(doc, rows):
            pred = get_misc(tok, "Subject")
            gold = gold_subject(f)
            if pred and gold and pred == gold:
                tp += 1
            else:
                fp += bool(pred)
                fn += bool(gold)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp + fn, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(TEST))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    path = TEST[args.lang].format(split=args.split)
    if not pathlib.Path(path).exists():
        sys.exit(f"missing {path}")
    rows = list(sentences(path))

    print(f"{args.lang} {args.split}   (gold tokens, everything else predicted)")

    trained_dir = f"training_{args.lang}_sud/model-best"
    if pathlib.Path(trained_dir).exists():
        nlp = spacy.load(trained_dir)
        P, R, F, n, sk = score(nlp, rows)
        print(f"  sud_tagger (trained)  gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}"
              + (f"  [{sk} sents skipped]" if sk else ""))
    else:
        print(f"  sud_tagger (trained)  -- {trained_dir} missing")

    lemma = LEMMA_ARM.get(args.lang, f"training_{args.lang}_lemma/model-best")
    if pathlib.Path(lemma).exists():
        nlp = spacy.load(lemma)
        if "sud_subject_rule" in nlp.pipe_names:
            nlp.remove_pipe("sud_subject_rule")
        nlp.add_pipe("sud_subject_rule", last=True, config={"lang": args.lang})
        P, R, F, n, sk = score(nlp, rows)
        print(f"  sud_subject_rule      gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}"
              + (f"  [{sk} sents skipped]" if sk else ""))
    else:
        print(f"  sud_subject_rule      -- {lemma} missing")


if __name__ == "__main__":
    main()
