#!/usr/bin/env python3
"""Predict an aligned-space gloss vector from a SHIPPED monolingual arm's own tok2vec.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. Not a parser input -- a predicted gloss carries no
information the parser does not already have, so it cannot raise LAS. It is ANNOTATION ASSISTANCE:
a linguist glossing text in a language that already has a parser gets a ranked proposal to correct
instead of writing each gloss from scratch. The metric is therefore keystrokes saved -- top-k
accuracy against a human gloss -- not LAS.

WHY THE ARM'S OWN TOK2VEC. The generic parser's encoder is deliberately lexically blind (UPOS, FEATS,
a language row) and could say nothing about what a novel word MEANS. A shipped monolingual arm's
tok2vec is the opposite: `attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]` over the wordform, in
context, already trained on the language. That representation is what this reads.

⚠ A DELIBERATE DEPARTURE FROM THE FREEZE RECIPE, flagged rather than done quietly. The recipe gives
every new layer its OWN small HashEmbedCNN rather than a listener, because co-training is dominated.
This is neither: the whole arm is FROZEN and a head is fitted on its frozen OUTPUT. A fresh encoder
would have to relearn Arabic orthography from a linguist's small glossed sample when the shipped arm
already knows it, and the head is tiny (width x 128). No parsing weight is touched, so the wheel's
behaviour cannot regress.

⚠ GATED TO CONTENT WORDS. Function morphemes are glossed as Leipzig abbreviations -- ACC, NMLZ, 3SG
-- which have no vector in the English space at all. A proposal for them would be confidently wrong
in exactly the place an annotator is least able to skim past it.

⚠ SCORED ON UNSEEN LEMMAS, and the split is BY LEMMA, not by token. A seen lemma is a lexicon
lookup, and a lexicon over already-glossed text already reaches cos 0.81-0.86 -- measuring on a
random token split would report mostly that, and flatter the model by 20-40 points of coverage it
contributes nothing to.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
SPLIT = re.compile(r"[-.:=,;/\[\]()<>+~]+|_")
TAB = "\t"


def gloss_words(g):
    return [p for part in SPLIT.split(g.replace("_", " ")) for p in part.split()
            if p.isalpha() and not (p.isupper() and len(p) > 1)]


def read(conllu_glob):
    """(tokens, lemmas, upos, glosses) per sentence."""
    for fn in sorted(glob.glob(conllu_glob)):
        w, l, u, g = [], [], [], []
        for ln in open(fn, encoding="utf-8", errors="replace"):
            if not ln.strip():
                if w:
                    yield w, l, u, g
                w, l, u, g = [], [], [], []
                continue
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split(TAB)
            if len(c) < 10 or "-" in c[0] or "." in c[0]:
                continue
            w.append(c[1])
            l.append(c[2] if c[2] not in ("", "_") else c[1])
            u.append(c[3])
            m = re.search(r"(?:^|\|)Gloss=([^|]*)", c[9])
            g.append(m.group(1) if m and m.group(1) not in ("", "_") else None)
        if w:
            yield w, l, u, g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="installed package name, e.g. ar_sud_padt")
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--table", default="assets_vec/generic_vec_v3.npz")
    ap.add_argument("--vec-lang", default="en", help="row-set the gloss words are looked up in")
    ap.add_argument("--max-sents", type=int, default=4000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--disable", nargs="*", default=["clause_parser"],
                    help="pipes to switch off while reading tok2vec output. lzh's `clause_parser` "
                         "REBUILDS the Doc and drops doc.tensor -- the third time that component "
                         "has lost an annotation it did not think to carry (lemma/morph, then "
                         "token extensions, now the tensor). Nothing downstream of it is needed "
                         "here, so it is switched off rather than fixed in passing.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import importlib
    from spacy.tokens import Doc
    from sud_generic_embed_v3 import load_vectors

    T = load_vectors(a.table)
    en_rows = [i for k, i in T.idx.items() if k[0] == a.vec_lang]
    en_keys = [k[1] for k in T.idx if k[0] == a.vec_lang]
    EN = T.V[np.asarray(en_rows)]
    print(f"english rows: {len(EN):,}")

    nlp = importlib.import_module(a.arm).load()
    for d in a.disable:
        if d in nlp.pipe_names:
            nlp.disable_pipe(d)
            print(f"  disabled {d} (it rebuilds the Doc and drops doc.tensor)")
    print(f"arm {a.arm}: {nlp.pipe_names}")

    def gvec(g):
        vs = [T.V[r] for p in gloss_words(g) if (r := T.row(a.vec_lang, p)) is not None]
        if not vs:
            return None
        m = np.mean(vs, 0)
        n = np.linalg.norm(m)
        return m / n if n else None

    X, Y, LEM, GOLD = [], [], [], []
    for i, (w, l, u, g) in enumerate(read(a.conllu)):
        if i >= a.max_sents:
            break
        doc = Doc(nlp.vocab, words=w)
        doc = nlp(doc)
        if doc.tensor is None or not len(doc.tensor):
            sys.exit("the arm produced no doc.tensor -- no tok2vec output to read")
        for j, tok in enumerate(doc):
            if u[j] not in CONTENT or not g[j]:
                continue
            v = gvec(g[j])
            if v is None:
                continue
            X.append(doc.tensor[j]); Y.append(v); LEM.append(l[j].lower()); GOLD.append(g[j])
    X = np.asarray(X, dtype="float32"); Y = np.stack(Y)
    print(f"{len(X):,} content tokens with a human gloss and a vector; tok2vec width {X.shape[1]}")

    # SPLIT BY LEMMA -- see the docstring warning.
    lemmas = sorted(set(LEM))
    rng = np.random.default_rng(0)
    rng.shuffle(lemmas)
    test_lem = set(lemmas[:max(1, len(lemmas) // 5)])
    te = np.array([i for i, l in enumerate(LEM) if l in test_lem])
    tr = np.array([i for i, l in enumerate(LEM) if l not in test_lem])
    print(f"{len(lemmas):,} lemma types -> train {len(tr):,} tokens / test {len(te):,} tokens "
          f"on {len(test_lem):,} UNSEEN lemmas")

    # closed-form ridge: a tiny head on frozen features
    Xt = np.hstack([X[tr], np.ones((len(tr), 1), "float32")])
    W = np.linalg.solve(Xt.T @ Xt + a.ridge * np.eye(Xt.shape[1], dtype="float32"), Xt.T @ Y[tr])
    Xe = np.hstack([X[te], np.ones((len(te), 1), "float32")])
    P = Xe @ W
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

    cos = (P * Y[te]).sum(1)
    mean_v = Y[tr].mean(0); mean_v /= np.linalg.norm(mean_v)
    print(f"\ncos(predicted, human gloss) on UNSEEN lemmas   {cos.mean():+.4f}")
    print(f"  constant-mean baseline                       {(Y[te] @ mean_v).mean():+.4f}")

    # top-k: is any word of the human gloss among the k nearest English rows?
    def topk_hit(pred, gold, k):
        s = EN @ pred
        idx = np.argpartition(-s, k)[:k]
        prop = {en_keys[i] for i in idx}
        return bool(prop & {w.lower() for w in gloss_words(gold)})
    hits = {k: 0 for k in (1, a.topk, 20)}
    base = {k: 0 for k in (1, a.topk, 20)}
    n = min(len(te), 3000)
    for i in range(n):
        for k in hits:
            hits[k] += topk_hit(P[i], GOLD[te[i]], k)
            base[k] += topk_hit(mean_v, GOLD[te[i]], k)
    print(f"\ntop-k: the human gloss appears among the k nearest English words ({n:,} unseen-lemma tokens)")
    for k in sorted(hits):
        print(f"  top-{k:<3d} head {hits[k]/n:6.1%}   constant baseline {base[k]/n:6.1%}")

    if a.json:
        json.dump({"arm": a.arm, "n_tokens": len(X), "n_unseen_lemma_tokens": int(len(te)),
                   "cos": float(cos.mean()), "cos_baseline": float((Y[te] @ mean_v).mean()),
                   "topk": {str(k): hits[k] / n for k in hits},
                   "topk_baseline": {str(k): base[k] / n for k in base}},
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
