#!/usr/bin/env python3
"""Estimate the systematic SOURCE-LEMMA -> ENGLISH-GLOSS offset in the shared space.

The v3 channel trains on aligned lemma vectors and is deployed on English gloss vectors, and seed 0
showed that the substitution loses the gain (+4.51 -> -0.86 macro LAS) even though it is
geometrically sound (cos +0.46 against +0.11 shuffled). Measuring the shift says why: it is not
noise around the source vector, it is a DISPLACEMENT.

On 239,748 Arabic pairs:

    cos(source, gloss)              +0.4597  (sd 0.1845)
    ||mean shift|| / mean ||shift||  34.6 %   -- a third of it is one constant direction
    residual top-8 of 128 dims       36.3 %   of variance (isotropic would be 6.2 %)
    cos after removing the mean      +0.5017

So an isotropic-noise augmentation would be modelling the wrong thing, and subtracting the mean is a
free inference-time correction that needs no retraining.

⚠ ESTIMATED ON A TRAINING LANGUAGE, APPLIED TO HELD-OUT ONES, AND THAT IS THE POINT TO BE SCEPTICAL
ABOUT. Whether one language's offset generalises is not settled by this script; it is settled by
whether subtracting it improves held-out LAS, which is an end-to-end test. A cosine between two
languages' offsets would be weaker evidence, and the obvious second language here is not available:
Ancient Hebrew (hbo) carries glosses but is NOT in fastText's aligned-44, so its lemmas have no rows
-- looking them up in the MODERN Hebrew table produces a number that means nothing.

⚠ NO TEST DATA IS INVOLVED. The pairs come from a training language's own treebank.
"""
from __future__ import annotations
import argparse, glob, pathlib, re, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sud_generic_embed_v3 import load_vectors        # noqa: E402

SPLIT = re.compile(r"[-.:=,;/\[\]()<>+~]+|_")


def gloss_vec(table, g):
    vs = []
    for p in (q for part in SPLIT.split(g.replace("_", " ")) for q in part.split()):
        if p.isalpha() and not (p.isupper() and len(p) > 1):
            r = table.row("en", p)
            if r is not None:
                vs.append(table.V[r])
    if not vs:
        return None
    m = np.mean(vs, 0)
    n = np.linalg.norm(m)
    return m / n if n else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="assets_vec/generic_vec_v3.npz")
    ap.add_argument("--lang", default="ar")
    ap.add_argument("--corpus", default="assets_sud218/sud-treebanks-v2.18/SUD_Arabic-PADT")
    ap.add_argument("--out", default="assets_vec/gloss_shift_v3.npy")
    ap.add_argument("--sample-out", default="assets_vec/gloss_shift_sample_v3.npy",
                    help="a SAMPLE of real shift vectors, for training-time augmentation. The mean "
                         "alone is the wrong object to augment with: it is only 34.6 %% of the "
                         "shift, and subtracting it at inference was measured to HURT (-0.86 -> "
                         "-1.05 macro LAS). The residual is where the shift lives, and it is "
                         "strongly anisotropic -- top-8 of 128 dims hold 36.3 %% of its variance "
                         "against 6.2 %% if it were isotropic -- so Gaussian noise would model "
                         "something the data is not. Sampling real displacements reproduces the "
                         "deployment distribution without training on a single gloss.")
    ap.add_argument("--sample-n", type=int, default=20000)
    a = ap.parse_args()

    T = load_vectors(a.table)
    S, G = [], []
    for fn in sorted(glob.glob(f"{a.corpus}/*.conllu")):
        for ln in open(fn, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip():
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 10 or not c[0].isdigit():
                continue
            m = re.search(r"(?:^|\|)Gloss=([^|]*)", c[9])
            if not m or m.group(1) in ("", "_"):
                continue
            r = T.row(a.lang, c[2] if c[2] not in ("", "_") else c[1])
            if r is None:
                continue
            gv = gloss_vec(T, m.group(1))
            if gv is not None:
                S.append(T.V[r]); G.append(gv)
    if len(S) < 1000:
        sys.exit(f"only {len(S)} pairs -- too few to estimate a 128-d offset")
    S, G = np.stack(S), np.stack(G)
    D = G - S
    mu = D.mean(0).astype("float32")
    Gc = G - mu
    Gc /= np.linalg.norm(Gc, axis=1, keepdims=True)
    print(f"{len(S):,} pairs from {a.lang}")
    print(f"  cos(source, gloss)            {(S*G).sum(1).mean():+.4f}")
    print(f"  cos after removing mean shift {(S*Gc).sum(1).mean():+.4f}")
    print(f"  systematic fraction            {np.linalg.norm(mu)/np.linalg.norm(D,axis=1).mean():.1%}")
    np.save(a.out, mu)
    print(f"wrote {a.out}")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(D), min(a.sample_n, len(D)), replace=False)
    np.save(a.sample_out, D[idx].astype("float32"))
    print(f"wrote {a.sample_out}: {len(idx):,} real shift vectors")


if __name__ == "__main__":
    main()
