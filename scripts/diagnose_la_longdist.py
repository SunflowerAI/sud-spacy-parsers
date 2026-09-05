#!/usr/bin/env python
"""Why does the la arc-factored decoder lose to the transition parser specifically on long-distance
arcs (dist 5-9: -8.5, -4.3, -8.4, -5.6), when a BiLSTM has, in principle, unbounded context?

Two independent hypotheses, tested on the ALREADY-TRAINED checkpoint (no retraining, so this
isolates each mechanism cleanly rather than confounding it with a different training run):

  (A) THE DISTANCE BIAS IS FIGHTING THE ARC SCORER. `train_arcfactored.py` inherits lzh's signed
      distance buckets -- a scalar bias per (direction, magnitude) bucket, which lzh already showed
      helps the common short case and actively HURTS long ones, because a bias shared by every pair
      at that distance cannot discriminate WHICH distant token is the head. Test: zero `P["dist"]`
      at inference and re-decode with the SAME encoder output -- if long-distance accuracy jumps,
      the bias is the culprit, not the representation.

  (B) THE BiLSTM's STATE IS DILUTED BY THE OTHER NINE SENTENCES IN THE DOC. This decoder trains and
      evaluates over whole `convert -n 10` documents (la mean 145 tokens, up to 449), with no reset
      at sentence boundaries -- so a token's hidden state also carries forward content from
      whichever OTHER sentences happen to sit in the same document, encoded by a small (width 96,
      depth 2) BiLSTM. A transition parser has no such interference: each sentence gets a fresh
      stack. Test: re-encode each GOLD sentence on its OWN (fresh BiLSTM state, no other sentences
      present), using the SAME trained weights, and re-decode per sentence -- if long-distance
      accuracy improves under this regime, dilution is (at least part of) the story, and training on
      pre-segmented sentences is a real candidate fix, not just an inference-time trick.

Both run over the SAME test docs and the SAME trained biaffine + BiLSTM weights, so the only
variable in each comparison is the one being tested.
"""
import collections, json, pathlib, sys
import numpy as np

sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import Doc
from spacy.util import registry
from sud_cle import mst
import train_arcfactored as tr

NEG = -1e4
MODEL = "models/la_arcfactored"


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    i = np.arange(n)
    m[:, 1:] = np.abs(i[:, None] - i[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


def decode(meta, P, X, use_dist=True):
    n = X.shape[0]; h = meta["hidden"]
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    Hr = np.vstack([np.zeros((1, h), "float32"), H])
    S = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    if use_dist and "dist" in P:
        S = S + P["dist"][tr.dist_buckets(n, meta["window"])]
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
    nlp = spacy.load(meta["src"])
    encoder_ignored, upstream = tr.encoder_and_upstream(nlp)  # only need `upstream` here

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

    def cut():
        return collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # n, whole, nodist, presegment, combined

    by_dist = cut()
    tot = [0, 0, 0, 0, 0]
    for g in gold:
        n = len(g)
        gh = np.array([0 if t.head.i == t.i else t.head.i + 1 for t in g])
        # -- whole-doc encoding (the trained regime) --
        d_whole = Doc(nlp.vocab, words=[t.text for t in g])
        tr.annotate_upstream(nlp, d_whole, upstream)
        X_whole = enc.predict([d_whole])[0]
        h_whole, _ = decode(meta, P, X_whole, use_dist=True)
        h_nodist, _ = decode(meta, P, X_whole, use_dist=False)
        # -- per-sentence encoding: fresh BiLSTM state per gold sentence, same weights --
        h_seg = np.zeros(n, dtype="int64")
        h_seg_nodist = np.zeros(n, dtype="int64")
        for s in g.sents:
            off = s.start
            d_s = Doc(nlp.vocab, words=[t.text for t in s])
            tr.annotate_upstream(nlp, d_s, upstream)
            X_s = enc.predict([d_s])[0]
            h_s, _ = decode(meta, P, X_s, use_dist=True)
            h_s_nd, _ = decode(meta, P, X_s, use_dist=False)
            for i in range(len(s)):
                h_seg[off + i] = 0 if h_s[i] == 0 else off + h_s[i]
                h_seg_nodist[off + i] = 0 if h_s_nd[i] == 0 else off + h_s_nd[i]
        for i in range(n):
            dist_key = f"dist {min(abs((gh[i] or i + 1) - 1 - i), 9)}" if gh[i] else "dist root"
            ok_whole = h_whole[i] == gh[i]
            ok_nodist = h_nodist[i] == gh[i]
            ok_seg = h_seg[i] == gh[i]
            ok_combined = h_seg_nodist[i] == gh[i]
            c = by_dist[dist_key]
            c[0] += 1; c[1] += ok_whole; c[2] += ok_nodist; c[3] += ok_seg; c[4] += ok_combined
            tot[0] += 1; tot[1] += ok_whole; tot[2] += ok_nodist; tot[3] += ok_seg; tot[4] += ok_combined
    print(f"  {tot[0]} tokens (UAS, same trained weights throughout)")
    print(f"    whole-doc + dist bias (as reported)   {tot[1]*100/tot[0]:.2f}")
    print(f"    whole-doc, dist bias ZEROED           {tot[2]*100/tot[0]:.2f}   "
          f"{(tot[2]-tot[1])*100/tot[0]:+.2f}")
    print(f"    per-sentence encoding (fresh state)   {tot[3]*100/tot[0]:.2f}   "
          f"{(tot[3]-tot[1])*100/tot[0]:+.2f}")
    print(f"    per-sentence + dist bias ZEROED       {tot[4]*100/tot[0]:.2f}   "
          f"{(tot[4]-tot[1])*100/tot[0]:+.2f}\n")
    print(f"  {'':<12}{'n':>7}{'whole':>10}{'no-dist':>12}{'per-sent':>12}{'both':>12}")
    order = [f"dist {i}" for i in range(10)] + ["dist root"]
    for k in order:
        if k not in by_dist: continue
        n_, w, nd, sg, cb = by_dist[k]
        if n_ < 30: continue
        print(f"  {k:<12}{n_:>7}{w*100/n_:>10.2f}{nd*100/n_:>12.2f}{sg*100/n_:>12.2f}{cb*100/n_:>12.2f}"
              f"   (no-dist {(nd-w)*100/n_:+.2f}, per-sent {(sg-w)*100/n_:+.2f}, both {(cb-w)*100/n_:+.2f})")


if __name__ == "__main__":
    main()
