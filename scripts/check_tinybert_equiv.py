#!/usr/bin/env python
"""Prove the numpy (torch-free) TinyBERT tok2vec equals the torch one, and that the full pipeline's parse agrees.

    check_tinybert_equiv.py TORCH_MODEL_DIR NUMPY_MODEL_DIR TEST.spacy

Both directories are loaded in THIS (torch) environment. Reports: (1) max |difference| of the tok2vec output on
every test sentence plus stress docs (a >256-piece doc that must chunk, the empty doc, whitespace-only tokens,
a literal [SEP], CJK, accents); (2) the full-pipeline analysis compared token for token (head, dep, tag, pos,
lemma, morph, sent boundaries), with the list of every differing sentence.
"""
import sys, pathlib
import numpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: F401
import spacy
from spacy.tokens import Doc, DocBin


def main():
    tdir, ndir, corpus = sys.argv[1:4]
    nt, nn = spacy.load(tdir), spacy.load(ndir)
    print("torch  pipeline:", nt.pipe_names)
    print("numpy  pipeline:", nn.pipe_names)
    gold = list(DocBin().from_disk(corpus).get_docs(nt.vocab))
    sents = [([t.text for t in d], [bool(t.whitespace_) for t in d]) for d in gold]
    flat = [w for s, _ in sents for w in s]
    stress = [
        (flat[:900], None),                                  # > 256 pieces: forces chunking
        (flat[:2500], None),
        ([], None),                                          # empty doc
        (["Hello", " ", "world", "\n", "​", "　", "!"], None),   # whitespace-only tokens -> [UNK]
        (["a", "[SEP]", "[CLS]", "b", "[UNK]", "[MASK]"], None),
        (["Café", "naïve", "北京", "Über", "İstanbul", "don't", "U.S.", "x" * 150], None),
        (["The"] * 255, None), (["The"] * 256, None), (["The"] * 257, None), (["supercalifragilisticexpialidocious"] * 90, None),
    ]
    docs_t = [Doc(nt.vocab, words=w, spaces=s) for w, s in sents + stress]
    docs_n = [Doc(nn.vocab, words=w, spaces=s) for w, s in sents + stress]
    mt, mn = nt.get_pipe("tok2vec").model, nn.get_pipe("tok2vec").model
    worst, worst_i = 0.0, -1
    for i in range(0, len(docs_t), 64):
        a, b = mt.predict(docs_t[i:i + 64]), mn.predict(docs_n[i:i + 64])
        for j, (x, y) in enumerate(zip(a, b)):
            assert x.shape == y.shape, (i + j, x.shape, y.shape)
            if len(x):
                d = float(numpy.abs(x - y).max())
                if d > worst:
                    worst, worst_i = d, i + j
    print(f"(1) tok2vec: {len(docs_t)} docs ({len(sents)} test sentences + {len(stress)} stress); "
          f"max |torch-numpy| = {worst:.3e} (doc {worst_i}); stress shapes OK")
    diff_sents, tok_diff, n_tok = [], 0, 0
    out_t = list(nt.pipe([Doc(nt.vocab, words=w, spaces=s) for w, s in sents + stress], batch_size=64))
    out_n = list(nn.pipe([Doc(nn.vocab, words=w, spaces=s) for w, s in sents + stress], batch_size=64))
    for k, (a, b) in enumerate(zip(out_t, out_n)):
        sig = lambda d: [(t.head.i, t.dep_, t.tag_, t.pos_, t.lemma_, str(t.morph), t.is_sent_start, str(getattr(t._, "sud_misc", None))) for t in d]
        sa, sb = sig(a), sig(b)
        n_tok += len(sa)
        bad = [i for i, (p, q) in enumerate(zip(sa, sb)) if p != q]
        if bad or len(sa) != len(sb):
            tok_diff += len(bad)
            diff_sents.append((k, len(bad)))
    kinds = "heads+deps+tags+pos+lemma+morph+sent-start+sud_misc"
    print(f"(2) full pipeline ({kinds}): {len(out_t)} docs / {n_tok} tokens; differing docs {len(diff_sents)}, "
          f"differing tokens {tok_diff}", diff_sents[:20])
    for text in ("", "   ", "a \n\n b", "The cat sat."):      # the pipeline itself, incl. the empty-batch path in Tok2Vec.predict
        da, db = nt(text), nn(text)
        assert [(t.text, t.head.i, t.dep_, t.tag_) for t in da] == [(t.text, t.head.i, t.dep_, t.tag_) for t in db], repr(text)
    print("(3) nlp(text) on '', '   ', 'a \\n\\n b', 'The cat sat.': identical")
    miscs = [p for p in nt.pipe_names if p.startswith("sud_")]
    for p in miscs:
        pass


if __name__ == "__main__":
    main()
