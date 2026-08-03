#!/usr/bin/env python3
"""Raw end-to-end evaluation of a zh arm under a chosen character segmenter.

`spacy evaluate --gold-preproc` scores the parser on GOLD tokens, which is the right way to compare
parsers but says nothing about the tokeniser. This runs the pipeline on the gold docs' raw TEXT, so
the segmenter's errors propagate into tagging, sentence segmentation and parsing — the number a user
of the wheel actually gets.

    eval_zh_raw.py training_zh_lemma/model-best corpus_zh_both/..test.spacy models/zh_seg_jbdec \
        [--lexicon models/zh_lex_corpus.txt]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("segmenter", nargs="?", default=None,
                    help="character segmenter directory; omit to use the model's own tokenizer")
    ap.add_argument("--lexicon", default="models/zh_lex_corpus.txt")
    a = ap.parse_args()

    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example

    nlp = spacy.load(a.model)
    if a.segmenter:
        from char_seg_tokenizer import CharSegTokenizer
        tok = CharSegTokenizer(nlp.vocab)
        tok.load_segmenter(a.segmenter, lexicon=a.lexicon)
        nlp.tokenizer = tok

    docs = list(DocBin().from_disk(a.data).get_docs(nlp.vocab))
    examples = [Example(nlp(d.text), d) for d in docs]
    sc = nlp.evaluate(examples)
    name = a.segmenter or "(model's own tokenizer)"
    print(f"{name}  on {len(docs)} docs")
    for k in ("token_acc", "token_f", "tag_acc", "pos_acc", "lemma_acc",
              "sents_f", "dep_uas", "dep_las"):
        if sc.get(k) is not None:
            print(f"  {k:12s} {sc[k]:.4f}")


if __name__ == "__main__":
    main()
