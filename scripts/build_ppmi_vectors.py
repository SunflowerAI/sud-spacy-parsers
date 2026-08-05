#!/usr/bin/env python3
"""Train type vectors from a raw corpus by PPMI + truncated SVD.

WHY NOT fastText/word2vec.  Below roughly 50M tokens, count-based PPMI+SVD is competitive with or
better than SGNS (Levy, Goldberg & Dagan 2015), trains in minutes on CPU with no new dependency, and
is DETERMINISTIC -- which matters here because every accuracy difference we care about is around a
point and single-seed neural runs vary by more than that. It also has no hyperparameters worth
tuning beyond the window and the shift.

WHY WE NEED IT AT ALL.  Only en/zh/ja/ko have spaCy vector models. fastText covers id/fa/ar/la but
is CC BY-SA 3.0, which is incompatible with the CC BY-NC-SA Latin wheel and would attach share-alike
to any other wheel shipping a derived table. sa and lzh have corpora of their own that are BIGGER
than what fastText's Wikipedia release used, so training our own is both cleaner and better:

    sa   DCS, 6.7M tokens, already segmented and lemmatised, CC BY 4.0
    lzh  Kanseki Repository, CC BY-SA 4.0 -- and lzh needs NO segmenter, because the released
         tokeniser is one Han character per token, so the type inventory IS the character set

THE SHIFT.  PPMI is `max(0, log p(w,c)/(p(w)p(c)) - log k)`. The `- log k` is SPPMI (shifted PPMI),
which Levy & Goldberg show corresponds to SGNS with k negative samples; k=1 is plain PPMI. Shifting
sparsifies the matrix and usually helps on small corpora, so it is exposed rather than hardcoded.

EIGENVALUE WEIGHTING.  `W = U * S**p`. p=0.5 (the default) is the symmetric factorisation that
Levy et al. found beats p=1 on similarity tasks; p=0 discards the spectrum entirely.

    build_ppmi_vectors.py --corpus corpus_sa_tokens.txt --dim 300 --out vectors_sa_ppmi.vec
"""
import argparse
import collections
import math
import pathlib
import sys

import numpy as np


def read_sentences(path, limit=0):
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            toks = line.split()
            if toks:
                yield toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="one sentence per line, space-separated tokens")
    ap.add_argument("--out", required=True, help="word2vec text format (.vec)")
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--max-vocab", type=int, default=200000)
    ap.add_argument("--shift", type=float, default=1.0,
                    help="k in SPPMI = max(0, PMI - log k). 1 = plain PPMI")
    ap.add_argument("--eig", type=float, default=0.5, help="exponent p in W = U * S**p")
    ap.add_argument("--cds", type=float, default=0.75,
                    help="context distribution smoothing: p(c) raised to this power before PMI, "
                         "which damps the pull of rare contexts (Levy et al.'s single biggest "
                         "count-model win)")
    ap.add_argument("--limit", type=int, default=0, help="first N lines only (smoke test)")
    args = ap.parse_args()

    print(f"  pass 1: counting types in {args.corpus}")
    freq = collections.Counter()
    n_sent = n_tok = 0
    for toks in read_sentences(args.corpus, args.limit):
        freq.update(toks)
        n_sent += 1
        n_tok += len(toks)
    kept = [w for w, c in freq.most_common(args.max_vocab) if c >= args.min_count]
    idx = {w: i for i, w in enumerate(kept)}
    V = len(kept)
    print(f"    {n_sent:,} sentences, {n_tok:,} tokens, {len(freq):,} types "
          f"-> {V:,} kept (min-count {args.min_count})")
    if V < 2:
        raise SystemExit("vocabulary too small")

    print(f"  pass 2: co-occurrence, window +/-{args.window}")
    cooc = collections.Counter()
    for toks in read_sentences(args.corpus, args.limit):
        ids = [idx.get(t, -1) for t in toks]
        n = len(ids)
        for i, wi in enumerate(ids):
            if wi < 0:
                continue
            lo, hi = max(0, i - args.window), min(n, i + args.window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                wj = ids[j]
                if wj >= 0:
                    cooc[(wi, wj)] += 1.0
    print(f"    {len(cooc):,} non-zero cells "
          f"({len(cooc)/max(V*V,1):.4%} dense)")

    rows = np.fromiter((k[0] for k in cooc), dtype=np.int32, count=len(cooc))
    cols = np.fromiter((k[1] for k in cooc), dtype=np.int32, count=len(cooc))
    vals = np.fromiter(cooc.values(), dtype=np.float64, count=len(cooc))
    total = vals.sum()
    w_sum = np.bincount(rows, weights=vals, minlength=V)
    c_sum = np.bincount(cols, weights=vals, minlength=V)
    c_smooth = c_sum ** args.cds
    c_smooth_total = c_smooth.sum()

    print(f"  SPPMI (shift k={args.shift}, cds={args.cds})")
    pmi = (np.log(vals) - np.log(w_sum[rows]) - np.log(c_smooth[cols])
           + math.log(c_smooth_total))
    sppmi = pmi - math.log(args.shift) if args.shift != 1.0 else pmi
    keep = sppmi > 0
    print(f"    {keep.sum():,}/{len(sppmi):,} cells positive ({keep.mean():.1%})")

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    M = csr_matrix((sppmi[keep], (rows[keep], cols[keep])), shape=(V, V))
    dim = min(args.dim, V - 1)
    print(f"  truncated SVD -> {dim} dims")
    U, S, _Vt = svds(M, k=dim)
    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    W = U * (S ** args.eig)[None, :]
    print(f"    top singular values {S[0]:.2f} .. {S[-1]:.2f}; "
          f"row norms {np.linalg.norm(W,axis=1).min():.3f}-{np.linalg.norm(W,axis=1).max():.3f}")

    out = pathlib.Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"{V} {dim}\n")
        for i, w in enumerate(kept):
            fh.write(w + " " + " ".join(f"{x:.5f}" for x in W[i]) + "\n")
    print(f"  wrote {out}  ({V:,} x {dim}, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
