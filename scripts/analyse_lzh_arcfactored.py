#!/usr/bin/env python
"""WHERE does the arc-factored parser lose to the transition parser? Decompose, don't guess.

Six cuts, each testing a specific hypothesis about the gap:

  ROOT / SENTENCE BOUNDARY -- the leading suspect. An arc-factored decoder discovers sentence
      boundaries IMPLICITLY, as tokens that attach to the virtual root across a 77-token document;
      the transition parser has a dedicated BREAK action for exactly this. If the gap concentrates
      here it is a segmentation problem wearing a parsing costume, and headline UAS cannot see it.
  ARC LENGTH        -- graph-based decoders usually WIN on long arcs and lose on local ones.
  CROSSING ARCS     -- the whole motivation. Arc-factored should beat a pseudo-projective transition
      system here; if it does not, the architecture is not delivering its one structural advantage.
  PER-DEPREL        -- which relations carry the loss.
  LABEL GIVEN HEAD  -- separates the arc scorer from the label scorer.
  SENTENCE LENGTH   -- does the loss grow with the segmentation burden?
"""
import argparse, json, pathlib, sys, collections
import numpy as np
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import DocBin, Doc
from spacy.util import registry
from sud_cle import mst
# ⚠ MUST STAY IN SYNC with the trainer's scorer, which now adds a signed-distance bias.
import importlib.util as _iu
_sp = _iu.spec_from_file_location("_tr", "scripts/train_lzh_arcfactored.py")
_tr = _iu.module_from_spec(_sp); _sp.loader.exec_module(_tr)

NEG = -1e4
C = ("corpus_lzh_resplit_ctl/lzh_kyoto-sud-test."
     "relabeled_ext.udep_ruled.punct.rulemerged.resplit.spacy")


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    i = np.arange(n)
    m[:, 1:] = np.abs(i[:, None] - i[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


def load_arcfactored(path, nlp, bilstm=False, tagfeat=False):
    meta = json.loads((pathlib.Path(path) / "meta.json").read_text())
    P = dict(np.load(pathlib.Path(path) / "biaffine.npz"))
    # ⚠ MUST MATCH THE TRAINER'S ENCODER EXACTLY, or from_bytes loads into the wrong shapes.
    if bilstm or meta.get("bilstm"):
        from thinc.api import chain as _c, LSTM, with_padded
        # ⚠ THE ATTRIBUTE LIST IS PART OF THE SAVED STRUCTURE. A model trained with TAG has five
        # embed tables; building four here fails deserialisation with "mismatched structure".
        _at = ["NORM", "PREFIX", "SUFFIX", "SHAPE"] + (["TAG"] if tagfeat else [])
        _rw = [5000, 1000, 2500, 2500] + ([500] if tagfeat else [])
        embed = registry.architectures.get("spacy.MultiHashEmbed.v2")(
            width=96, attrs=_at, rows=_rw, include_static_vectors=False)
        enc = _c(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
        enc.set_dim("nO", 96)
    else:
        enc = registry.architectures.get("spacy.HashEmbedCNN.v2")(
            width=96, depth=4, embed_size=2000, window_size=1, maxout_pieces=3,
            subword_features=True, pretrained_vectors=None)
    enc.initialize(X=[Doc(nlp.vocab, words=["王", "愛", "民"])])
    enc.from_bytes((pathlib.Path(path) / "encoder.bin").read_bytes())
    return meta, P, enc


def predict(meta, P, enc, doc, nlp):
    X = enc.predict([doc])[0]
    n = X.shape[0]; h = meta["hidden"]
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    Hr = np.vstack([np.zeros((1, h), "float32"), H])
    S = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    if "dist" in P:
        S = S + P["dist"][_tr.dist_buckets(n, meta["window"])]
    S = np.where(window_mask(n, meta["window"]).T, S, NEG)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    LH = np.maximum(X @ P["Lh"], 0); LD = np.maximum(X @ P["Ld"], 0)
    hv = np.where((heads > 0)[:, None], LH[np.maximum(heads - 1, 0)], 0.0)
    sc = (np.einsum("nh,lhg,ng->nl", hv, P["V"], LD)
          + np.concatenate([hv, LD], 1) @ P["v"].T + P["cb"])
    return heads, [meta["labels"][i] for i in sc.argmax(1)]


def crossing_set(heads):
    n = len(heads)
    idx = [i for i in range(n) if heads[i] != i]
    arcs = {i: (min(i, heads[i]), max(i, heads[i])) for i in idx}
    bad = set()
    for i in idx:
        a = arcs[i]
        for j in idx:
            c = arcs[j]
            if a[0] < c[0] < a[1] < c[1] or c[0] < a[0] < c[1] < a[1]:
                bad.add(i); break
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/lzh_arcfactored")
    ap.add_argument("--baseline", default="training_lzh_depmorph_resplit/model-best")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bilstm", action="store_true")
    ap.add_argument("--tagfeat", action="store_true")
    a = ap.parse_args()
    nlp = spacy.load(a.baseline)
    t2v, parser = nlp.get_pipe("tok2vec"), nlp.get_pipe("parser")
    meta, P, enc = load_arcfactored(a.model, nlp, a.bilstm, a.tagfeat)
    print(f"  arc-factored checkpoint: epoch {meta['epoch']}, dev LAS {meta['las']:.2f}")
    gold = list(DocBin().from_disk(C).get_docs(nlp.vocab))
    if a.limit: gold = gold[:a.limit]
    cut = lambda: collections.defaultdict(lambda: [0, 0, 0])   # n, af_ok, tp_ok
    root, length, cross, dep, sent = cut(), cut(), cut(), cut(), cut()
    lab_given_head = [0, 0, 0, 0]
    tot = [0, 0, 0]
    for g in gold:
        words = [t.text for t in g]; n = len(g)
        d0 = Doc(nlp.vocab, words=words); t2v(d0); parser(d0)
        d1 = Doc(nlp.vocab, words=words); t2v(d1)
        if a.tagfeat:
            nlp.get_pipe("tagger")(d1)          # predicted XPOS, exactly as in training
        ah, al = predict(meta, P, enc, d1, nlp)
        gh = [0 if t.head.i == t.i else t.head.i + 1 for t in g]
        th = [0 if t.head.i == t.i else t.head.i + 1 for t in d0]
        gcross = crossing_set([t.head.i for t in g])
        slen = {}
        for s in g.sents:
            for t in s: slen[t.i] = len(s)
        for i, t in enumerate(g):
            af = ah[i] == gh[i] and al[i] == t.dep_
            tp = th[i] == gh[i] and d0[i].dep_ == t.dep_
            tot[0] += 1; tot[1] += af; tot[2] += tp
            k = "IS a sentence root" if gh[i] == 0 else "not a root"
            for c, key in ((root, k),
                           (length, f"dist {min(abs((gh[i] or i+1)-1-i), 9)}" if gh[i] else "dist root"),
                           (cross, "gold CROSSING arc" if i in gcross else "gold projective arc"),
                           (dep, t.dep_),
                           (sent, f"sent len {min(slen.get(i,1)//5*5, 20)}+")):
                c[key][0] += 1; c[key][1] += af; c[key][2] += tp
            if ah[i] == gh[i]:
                lab_given_head[0] += 1; lab_given_head[1] += al[i] == t.dep_
            if th[i] == gh[i]:
                lab_given_head[2] += 1; lab_given_head[3] += d0[i].dep_ == t.dep_
    print(f"  {tot[0]} tokens   arc-factored LAS {tot[1]*100/tot[0]:.2f}   "
          f"transition LAS {tot[2]*100/tot[0]:.2f}   gap {(tot[1]-tot[2])*100/tot[0]:+.2f}\n")
    def show(c, title, top=None):
        print(f"  {title}")
        rows = sorted(c.items(), key=lambda kv: -kv[1][0])[:top]
        for k, (n_, af, tp) in rows:
            if n_ < 30: continue
            print(f"    {k:<22} n={n_:<6} arc-fact {af*100/n_:6.2f}  transition {tp*100/n_:6.2f}"
                  f"   {(af-tp)*100/n_:+6.2f}")
    show(root, "BY ROOT STATUS (the segmentation hypothesis)")
    show(cross, "BY GOLD PROJECTIVITY (the motivating advantage)")
    show(length, "BY ARC LENGTH")
    show(sent, "BY SENTENCE LENGTH")
    show(dep, "BY DEPREL (10 most frequent)", 10)
    print(f"\n  LABEL ACCURACY GIVEN A CORRECT HEAD: arc-factored "
          f"{lab_given_head[1]*100/max(lab_given_head[0],1):.2f}  "
          f"transition {lab_given_head[3]*100/max(lab_given_head[2],1):.2f}")


if __name__ == "__main__":
    main()
