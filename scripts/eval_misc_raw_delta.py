#!/usr/bin/env python3
"""What the gold-token MISC evals cannot see: the effect of a TOKENISER change.

`eval_sud_{subject,shared,idiom}.py` all build `Doc(vocab, words=[gold forms])`, so the tokeniser is
bypassed and a segmenter swap is invisible to every number they print. But `sud_subject_rule` keys on
`(head.lemma_, deprel, head.pos_)`, and merging 孔 + 子 into 孔子 changes that lemma -- so the layer's
RAW behaviour can move while the gold-token metrics sit still. Standing hazard 5 is about the MISC
layer reading the base's own predictions; this measures the half the harness hides.

Compares the shipped pipeline against the same pipeline with the OLD one-character tokeniser,
over the raw text of the test split, and reports how often each MISC feature fires.
"""
import argparse, collections, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: F401  (registers lzh + the tokenizers)
import spacy
import sud_misc                       # the MISC layer lives HERE, not in token.morph
from lzh_tokenizer import CharTokenizer

FEATS = ("Subject", "Shared", "Idiom", "InIdiom")

def raw_sentences(conllu):
    out, cur = [], []
    for line in pathlib.Path(conllu).open(encoding="utf-8"):
        if line.startswith("#"): continue
        if not line.strip():
            if cur: out.append("".join(cur)); cur = []
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]: continue
        cur.append(f[1])
    if cur: out.append("".join(cur))
    return out

def tally(nlp, texts):
    c = collections.Counter(); toks = 0
    for t in texts:
        d = nlp(t); toks += len(d)
        for tok in d:
            # ⚠ NOT `tok.morph`. sud_misc.py opens a spaCy EXTENSION for this layer on purpose, so
            # that "a predicted SUD feature never has to compete for room with a morphological
            # one". Reading token.morph here returned an empty table with no error at all.
            for k in FEATS:
                v = sud_misc.get_misc(tok, k)
                if v: c[f"{k}={v}"] += 1
    return c, toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conllu", required=True)
    a = ap.parse_args()
    texts = raw_sentences(a.conllu)
    print(f"{len(texts):,} raw sentences from {a.conllu}")
    nlp = spacy.load(a.model)
    new, ntok = tally(nlp, texts)
    old_nlp = spacy.load(a.model)
    old_nlp.tokenizer = CharTokenizer(old_nlp.vocab)   # what shipped before
    old, otok = tally(old_nlp, texts)
    print(f"tokens: one-char tokeniser {otok:,} -> segmenter {ntok:,} "
          f"({otok-ntok:,} fewer, {1-ntok/otok:.2%} merged away)")
    keys = sorted(set(old) | set(new))
    print(f"\n{'MISC feature':>22} {'one-char':>9} {'segmenter':>10} {'delta':>7}")
    for k in keys:
        print(f"{k:>22} {old[k]:>9} {new[k]:>10} {new[k]-old[k]:>+7}")

if __name__ == "__main__":
    main()
