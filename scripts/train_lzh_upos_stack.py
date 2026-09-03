#!/usr/bin/env python
"""Fit the `lzh_upos_stack` selector and write its artefacts into a model directory.

⚠ FITTED IN THE RELEASED ORDERING. The tagger must be the UPOS-reading one that ships, not an
independent tagger: the selector learns when to trust a tagger, and the two taggers disagree with
the morphologiser at different rates (1942 vs 3591 on test) and are right within those
disagreements at different rates (27.4 % vs 37.4 %).

⚠ THE COMPONENT SEES ONLY THE DOC, so the features here are limited to what a Doc carries -- XPOS,
UPOS, the form. The experimental version also used the two models' softmax confidences; they are
deliberately excluded so the shipped component needs no access to other pipes' internals.
"""
import argparse, collections, sys
import numpy as np
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import DocBin, Doc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

C = ("corpus_lzh_resplit_ctl/lzh_kyoto-sud-%s."
     "relabeled_ext.udep_ruled.punct.rulemerged.resplit.spacy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="training_lzh_tagger_upos/model-best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    nlp = spacy.load(a.arm)
    print(f"  arm pipeline: {nlp.pipe_names}")
    x2u = collections.defaultdict(collections.Counter)
    amb = collections.defaultdict(collections.Counter)
    for g in DocBin().from_disk(C % "train").get_docs(nlp.vocab):
        for t in g:
            x2u[t.tag_][t.pos_] += 1
            amb[t.text][t.pos_] += 1
    best = {k: c.most_common(1)[0][0] for k, c in x2u.items()}
    xpur = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in x2u.items()}
    ambn = {w: len(c) for w, c in amb.items()}
    nv = sorted({w for w, c in amb.items() if c["NOUN"] >= 20 and c["VERB"] >= 20})
    upos = sorted({u for c in x2u.values() for u in c})
    ui = {u: i for i, u in enumerate(upos)}

    def rows(split, limit=None):
        docs = list(DocBin().from_disk(C % split).get_docs(nlp.vocab))
        if limit:
            docs = docs[:limit]
        R, Y, P = [], [], []
        for g in docs:
            p = nlp(Doc(nlp.vocab, words=[t.text for t in g],
                        spaces=[bool(t.whitespace_) for t in g]))
            for i, gt in enumerate(g):
                tu = best.get(p[i].tag_)
                if tu is None or tu == p[i].pos_:
                    continue
                R.append([xpur.get(p[i].tag_, .5), float(gt.text in set(nv)),
                          float(np.log1p(ambn.get(gt.text, 1))),
                          float(ui.get(p[i].pos_, len(upos))), float(ui.get(tu, len(upos)))])
                Y.append(1 if tu == gt.pos_ else 0)
                P.append(f"{p[i].pos_}->{tu}")
        return R, np.array(Y), P

    Rtr, Ytr, Ptr = rows("train", a.limit)
    pairs = sorted(set(Ptr))
    pi = {q: i for i, q in enumerate(pairs)}

    def mat(R, P):
        X = np.zeros((len(R), 5 + len(pairs)), "float32")
        for i, (r, q) in enumerate(zip(R, P)):
            X[i, :5] = r
            if q in pi:
                X[i, 5 + pi[q]] = 1.0
        return X

    Xtr = mat(Rtr, Ptr)
    print(f"  {len(Ytr)} training disagreements, {len(pairs)} pairs, {Xtr.shape[1]} features")
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=0.3, max_iter=1500,
                        random_state=a.seed, early_stopping=True).fit(sc.transform(Xtr), Ytr)
    import lzh_upos_stack as S
    comp = S.LzhUposStack()
    comp.W = [(w.astype("float32"), b.astype("float32"))
              for w, b in zip(clf.coefs_, clf.intercepts_)]
    comp.mean = sc.mean_.astype("float32")
    comp.scale = np.where(sc.scale_ == 0, 1.0, sc.scale_).astype("float32")
    comp.x2u, comp.xpur, comp.ambn = best, xpur, ambn
    comp.nv, comp.pairs, comp.upos = set(nv), pairs, upos
    comp.to_disk(a.out)
    print(f"  wrote {a.out}  ({len(comp.W)} layers, {sum(w.size for w, _ in comp.W):,} weights)")


if __name__ == "__main__":
    main()
