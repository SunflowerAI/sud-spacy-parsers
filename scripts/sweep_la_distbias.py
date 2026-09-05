#!/usr/bin/env python
"""Can the proximity (signed-distance) bias just be TUNED, rather than restructured?

Sweeps a scalar multiplier on the trained `P["dist"]` array, on the ALREADY-TRAINED checkpoint --
no retraining, so this bounds the value of "tune its strength" before investing in anything bigger
(a per-label/joint arc+label scoring change). Reports overall UAS and, specifically, `conj:coord`
and `cc` accuracy (the relations `diagnose_la_deprel_errors.py` found are dominated by the bias
overriding a correct FAR head with a plausible NEAR one), at each scale.

⚠ WHY A GLOBAL SCALAR CANNOT BE THE REAL FIX, EVEN IF IT HELPS A LITTLE. The bias is ONE shared
array over (direction, magnitude) buckets, read at ARC-SCORING time -- before any label is chosen.
`mod`/`det`/`comp:obj` genuinely want a near head most of the time; `conj:coord` genuinely wants
whichever token started the chain, however far. A single scale factor moves ALL of them the same
way: turning it down helps the far-preferring relations and hurts the near-preferring ones (already
seen zeroing it outright: dist1 -24, root +12). So this sweep is asking "where is the least-bad
single setting", not "can the two be reconciled" -- for the latter, the bias would need to be
LABEL-CONDITIONED, which means scoring (head, dependent, label) jointly rather than arc-then-label.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import Doc
from spacy.util import registry
from sud_cle import mst
import train_arcfactored as tr

NEG = -1e4
MODEL = "models/la_arcfactored_preseg"


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    i = np.arange(n)
    m[:, 1:] = np.abs(i[:, None] - i[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


def decode(meta, P, X, scale):
    n = X.shape[0]; h = meta["hidden"]
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    Hr = np.vstack([np.zeros((1, h), "float32"), H])
    S = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    if "dist" in P:
        S = S + scale * P["dist"][tr.dist_buckets(n, meta["window"])]
    S = np.where(window_mask(n, meta["window"]).T, S, NEG)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    LH = np.maximum(X @ P["Lh"], 0); LD = np.maximum(X @ P["Ld"], 0)
    hv = np.where((heads > 0)[:, None], LH[np.maximum(heads - 1, 0)], 0.0)
    sc = (np.einsum("nh,lhg,ng->nl", hv, P["V"], LD)
          + np.concatenate([hv, LD], 1) @ P["v"].T + P["cb"])
    return heads, [meta["labels"][i] for i in sc.argmax(1)]


def main():
    meta = json.loads((pathlib.Path(MODEL) / "meta.json").read_text())
    P = dict(np.load(pathlib.Path(MODEL) / "biaffine.npz"))
    presegment = bool(meta.get("presegment"))
    nlp = spacy.load(meta["src"])
    _, upstream = tr.encoder_and_upstream(nlp)

    embed = registry.architectures.get("spacy.MultiHashEmbed.v2")(
        width=96, attrs=meta["joint_attrs"], rows=meta["joint_rows"], include_static_vectors=False)
    from thinc.api import chain as _c, LSTM, with_padded
    enc = _c(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
    enc.set_dim("nO", 96)
    gold = tr.load("la", "test", nlp)
    probe = Doc(nlp.vocab, words=[t.text for t in gold[0]])
    tr.annotate_upstream(nlp, probe, upstream)
    enc.initialize(X=[probe])
    enc.from_bytes((pathlib.Path(MODEL) / "encoder.bin").read_bytes())

    # -- pre-encode everything once; the sweep only changes DECODING, not encoding --
    items = []       # (X, gold_heads, gold_labels)
    for g in gold:
        sents = list(g.sents) if presegment else [g[:]]
        for s in sents:
            off = s.start
            d = Doc(nlp.vocab, words=[t.text for t in s])
            tr.annotate_upstream(nlp, d, upstream)
            X = enc.predict([d])[0]
            gh = np.array([0 if t.head.i == t.i else (t.head.i - off) + 1 for t in s])
            gl = [t.dep_ for t in s]
            items.append((X, gh, gl))

    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    print(f"  {'scale':>6}{'UAS':>8}{'LAS':>8}{'conj:coord':>12}{'cc':>8}{'mod':>8}{'det':>8}")
    for sc in scales:
        tot_n = tot_uas = tot_las = 0
        by_dep = {}
        for X, gh, gl in items:
            heads, labels = decode(meta, P, X, sc)
            for i in range(len(gh)):
                tot_n += 1
                u_ok = heads[i] == gh[i]
                l_ok = u_ok and labels[i] == gl[i]
                tot_uas += u_ok; tot_las += l_ok
                d = by_dep.setdefault(gl[i], [0, 0])
                d[0] += 1; d[1] += u_ok
        def acc(dep):
            d = by_dep.get(dep, [1, 0])
            return d[1] * 100 / d[0]
        print(f"  {sc:>6.2f}{tot_uas*100/tot_n:>8.2f}{tot_las*100/tot_n:>8.2f}"
              f"{acc('conj:coord'):>12.2f}{acc('cc'):>8.2f}{acc('mod'):>8.2f}{acc('det'):>8.2f}")


if __name__ == "__main__":
    main()
