#!/usr/bin/env python3
"""PCA a distributional lemma-vector table down to a parser-sized block.

`vectors_sa_lemma_ppmi.vec` is PPMI over LEMMAS, and it is genuinely distributional rather than
orthographic — the distinction that matters, because spaCy's `_md` tables are pruned/subword, so
their similarity structure is largely spelling and a parser already sees that through PREFIX,
SUFFIX, SHAPE and its character-window encoder. Nearest neighbours here share no characters at all:

    gam  -> yā, āgam, prayā, vraj, prāp, dṛś        (motion verbs)
    vac  -> brū, pracch, prativac, vacana, vākya    (speech verbs)
    agni -> jātavedas, samindh, indh, samidh, havya (the ritual frame of fire)

⚠ PCA IS LOSSY HERE, MEASURABLY. 300 -> 96 keeps 44.2 % of the variance. The table above is what has
to survive the compression, so `--report` prints the same neighbourhoods after PCA rather than
trusting the variance number.

    build_lemma_vectors.py --in vectors_sa_lemma_ppmi.vec --dim 96 --out scripts/sa_lemmavec_96.npz
"""
import argparse
import numpy as np


def load(path):
    keys, rows = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) > 10:
                keys.append(p[0])
                rows.append(np.asarray(p[1:], dtype="f"))
    M = np.vstack(rows)
    return keys, M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="vectors_sa_lemma_ppmi.vec")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--out", default="scripts/sa_lemmavec_96.npz")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--probe", default="gam,vac,agni,rājan,putra",
                    help="comma-separated lemmas whose neighbourhoods --report prints; the "
                         "defaults are Sanskrit, so any other language MUST pass its own or the "
                         "report silently prints nothing and the PCA goes unchecked")
    a = ap.parse_args()
    from sklearn.decomposition import PCA

    keys, M = load(a.src)
    pca = PCA(n_components=a.dim, random_state=0).fit(M)
    V = pca.transform(M).astype("f")
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)   # unit rows: keeps the block's scale
    np.savez_compressed(a.out, keys=np.array(keys, dtype=object), vectors=V)
    print(f"{len(keys)} lemmas x {a.dim}d -> {a.out}  "
          f"(variance kept {pca.explained_variance_ratio_.sum():.3f})")
    if a.report:
        idx = {k: i for i, k in enumerate(keys)}
        missing = []
        for w in [x.strip() for x in a.probe.split(",") if x.strip()]:
            if w in idx:
                s = V @ V[idx[w]]
                print(f"  {w:<10} -> " + ", ".join(keys[j] for j in np.argsort(-s)[1:7]))
            else:
                missing.append(w)
        if missing:
            print(f"  (not in the table: {', '.join(missing)})")


if __name__ == "__main__":
    main()
