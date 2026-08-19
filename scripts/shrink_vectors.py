#!/usr/bin/env python3
"""PCA a `.vec` down to fewer dimensions, and optionally emit the shuffled control alongside.

WHY, and what it does NOT change. `StaticVectors` already projects the table down to the tok2vec
width (96) with a learned linear map, so the 300 dimensions are not extra model capacity -- they are
purely stored bytes: 17 193 x 300 float32 is 20 MB on top of a 12 MB wheel, which is the objection
that sank the `md` fastText vectors (`NEGATIVE-RESULTS.md`: LAS +0.2-0.9, within noise, for 9-16x
the size). At 96 dimensions the same table is ~6.6 MB. The question this answers is whether the
measured gain survives that cut.

⚠ PRUNE BY DIMENSION, NEVER BY VOCABULARY. `vectors_lzh_apt96` is what the other mistake looks
like: pruned to a training vocabulary, it covers 0 % of treebank-unseen forms and holds no
punctuation at all. Rows are the value; dimensions are the cost.

PCA, not whitened, matching `build_pca_table.py` in the aptness repo -- whitening would destroy the
distance structure the vectors carry.

THE CONTROL IS SHUFFLED AFTER REDUCTION, so it has exactly the arm's row-norm distribution at
exactly the arm's dimensionality. Shuffling first and reducing after would give the PCA a different
matrix to fit and the two tables would no longer differ in one variable.

Usage:

    .venv/bin/python scripts/shrink_vectors.py vectors_lzh_ids_leakfree.vec \\
        --dims 96 --out vectors_lzh_ids_leakfree96.vec \\
        --shuffled-out vectors_lzh_shuffled96.vec
"""
import argparse
import random

import numpy


def read_vec(path):
    keys, rows = [], []
    with open(path, encoding="utf-8") as fh:
        _n, dim = fh.readline().split()
        for line in fh:
            p = line.rstrip("\n").split(" ")
            if len(p) != int(dim) + 1:
                continue
            keys.append(p[0])
            rows.append([float(x) for x in p[1:]])
    return keys, numpy.asarray(rows, dtype="float64")


def write_vec(path, keys, M):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(keys)} {M.shape[1]}\n")
        for k, v in zip(keys, M):
            fh.write(k + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vec")
    ap.add_argument("--dims", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffled-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    keys, M = read_vec(a.vec)
    print(f"  read {a.vec}: {len(keys)} x {M.shape[1]}")
    Xc = M - M.mean(0, keepdims=True)
    _, S, Vt = numpy.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(float((S ** 2).sum()), 1e-9)
    R = Xc @ Vt[:a.dims].T
    print(f"  PCA -> {a.dims} dims, retained variance {float(var[:a.dims].sum()):.1%}")

    write_vec(a.out, keys, R)
    print(f"  wrote {a.out}")

    if a.shuffled_out:
        idx = list(range(len(keys)))
        random.Random(a.seed).shuffle(idx)
        fixed = sum(1 for i, j in enumerate(idx) if i == j)
        write_vec(a.shuffled_out, keys, R[idx])
        print(f"  wrote {a.shuffled_out}  ({fixed} rows landed on themselves)")


if __name__ == "__main__":
    main()
