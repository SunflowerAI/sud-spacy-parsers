#!/usr/bin/env python
"""Measure whether sud_sibling_score.py's local-search reranking improves an ALREADY-TRAINED
arc-factored checkpoint -- no retraining needed, since the sibling table is a gold corpus statistic
(scripts/sud_sibling_score.py's own module docstring has the full motivation and design rationale).

Mirrors analyse_arcfactored.py's own load/predict path (not importing its `main`, to avoid touching
a MUST-STAY-IN-SYNC file for a one-off measurement script) so the FIRST-ORDER numbers this prints
are directly comparable to analyse_arcfactored.py's own headline LAS for the same checkpoint.
"""
import argparse
import collections
import pathlib
import sys

import numpy as np
import spacy
from spacy.tokens import Doc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_arcfactored as tr  # noqa: E402
from sud_cle import mst  # noqa: E402
from sud_sibling_score import build_sibling_table, rerank_with_siblings  # noqa: E402
from analyse_arcfactored import load_arcfactored, window_mask  # noqa: E402

NEG = -1e4


def scores_and_first_order(meta, P, X, doc):
    """Replicates predict_from_X_joint_label's own math up to the point of decoding, but returns
    the intermediate (S, chosen) matrices too -- MUST STAY IN SYNC with
    analyse_arcfactored.py's predict_from_X_joint_label, which this mirrors exactly."""
    n = X.shape[0]; h = meta["hidden"]
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    Hr = np.vstack([np.zeros((1, h), H.dtype), H])
    LH = np.maximum(X @ P["Lh"], 0); LD = np.maximum(X @ P["Ld"], 0)
    LHr = np.vstack([np.zeros((1, h), LH.dtype), LH])
    arc_raw = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    bil = np.einsum("hg,lgk,dk->hdl", LHr, P["V"], LD, optimize=True)
    lin1 = LHr @ P["v"][:, :h].T
    lin2 = LD @ P["v"][:, h:].T
    label_raw = bil + lin1[:, None, :] + lin2[None, :, :] + P["cb"][None, None, :]
    bkt = tr.dist_buckets(n, meta["window"])
    combined = arc_raw[:, :, None] + label_raw + P["dist"].T[bkt]
    if meta.get("agreement") and "agree" in P:
        combined = combined + P["agree"].T[tr.agreement_buckets(doc)]
    if meta.get("pos") and "pos" in P:
        combined = combined + P["pos"].T[tr.pos_buckets(doc)]
    if meta.get("lemvec") and "lemvec" in P:
        lv = tr.lemma_vecs(doc, meta["lemvec_table"])
        combined = combined + (lv @ P["lemvec"].T)[:, None, :]
    if meta.get("morphhash") and "morphhash" in P:
        mh = tr.morph_hash_buckets(doc)
        combined = combined + P["morphhash"].T[mh][None, :, :]
    if meta.get("direction") and "direction" in P:
        combined = combined + P["direction"].T[tr.direction_buckets(n, meta["window"])]
    mask = window_mask(n, meta["window"]).T
    combined = np.where(mask[:, :, None], combined, NEG)
    S, chosen = combined.max(-1), combined.argmax(-1)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    labels = chosen[heads, np.arange(n)]
    return S, chosen, heads, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/la_frozen_full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--sib-weight", type=float, default=1.0,
                    help="scale the sibling table before reranking -- 0.0 reduces to first-order "
                         "decoding exactly, for sanity-checking the scale against the biaffine's "
                         "own raw arc-score magnitude")
    a = ap.parse_args()

    import json
    meta = json.loads((pathlib.Path(a.model) / "meta.json").read_text())
    P = dict(np.load(pathlib.Path(a.model) / "biaffine.npz"))
    nlp = spacy.load(meta["src"])
    _, upstream = tr.encoder_and_upstream(nlp)
    encoder, _ = tr.encoder_and_upstream(nlp)

    train_gold = tr.load("la", "train", nlp, None)
    sib_table = build_sibling_table(train_gold, meta["labels"]) * a.sib_weight
    print(f"  sibling table built from {len(train_gold)} gold train docs, "
          f"{len(meta['labels'])} labels, absmax {np.abs(sib_table).max():.3f}", flush=True)

    test_gold = tr.load("la", "test", nlp, a.limit or None)
    if meta.get("presegment"):
        test_gold = tr.explode_sentences(test_gold)
        print(f"  PRESEGMENT: exploded test into {len(test_gold)} single-sentence items", flush=True)

    tot = base_ok = rerank_ok = 0
    by_deprel = collections.defaultdict(lambda: [0, 0, 0])   # n, base_ok, rerank_ok
    for g in test_gold:
        words = [t.text for t in g]
        n = len(g)
        if n == 0:
            continue
        d1 = Doc(nlp.vocab, words=words)
        tr.annotate_upstream(nlp, d1, upstream)
        X = tr.per_doc(encoder.predict([d1]), [d1])[0]
        S, chosen, base_heads, base_labels = scores_and_first_order(meta, P, X, d1)
        rr_heads, rr_labels = rerank_with_siblings(S, chosen, base_heads, sib_table,
                                                    k=a.k, max_passes=a.passes)
        gold_h = [0 if t.head.i == t.i else t.head.i + 1 for t in g]
        for i, t in enumerate(g):
            tot += 1
            b_ok = int(base_heads[i] == gold_h[i] and meta["labels"][base_labels[i]] == t.dep_)
            r_ok = int(rr_heads[i] == gold_h[i] and meta["labels"][rr_labels[i]] == t.dep_)
            base_ok += b_ok; rerank_ok += r_ok
            row = by_deprel[t.dep_]
            row[0] += 1; row[1] += b_ok; row[2] += r_ok

    print(f"\n  {tot} tokens   baseline LAS {base_ok*100/tot:.2f}   "
          f"reranked LAS {rerank_ok*100/tot:.2f}   delta {(rerank_ok-base_ok)*100/tot:+.2f}\n")
    print(f"  {'deprel':15s} {'n':>6s} {'base':>7s} {'rerank':>7s} {'delta':>7s}")
    for dep, (n_, b, r) in sorted(by_deprel.items(), key=lambda kv: -kv[1][0]):
        print(f"  {dep:15s} {n_:6d} {b*100/n_:6.2f}% {r*100/n_:6.2f}% {(r-b)*100/n_:+6.2f}")


if __name__ == "__main__":
    main()
