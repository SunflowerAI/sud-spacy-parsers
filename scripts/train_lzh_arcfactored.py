#!/usr/bin/env python
"""Arc-factored (biaffine + Chu-Liu/Edmonds) parser for lzh, over the FROZEN shared encoder.

WHY. The transition parser's beam holds a better parse than greedy in 40 % of docs (+1.35 LAS at
oracle) and no reranker reached it: tagger agreement, the only feature with signal, is HIGHER on the
wrong parse in the cases that matter, because model confidence measures internal COHERENCE, not
correctness. Inspecting those cases, the beam's wins are one recurring phenomenon -- greedy roots a
NOUN where the verb belongs -- and the beam's candidates are simultaneously POLLUTED by
pseudo-projective decorated labels (`punct||discourse@sp`) that greedy got right.

An arc-factored decoder addresses both: it scores a whole tree rather than a local action sequence,
and it needs no pseudo-projectivisation, so no decorated labels exist to get wrong.

⚠ WINDOWED at k=30. Measured on lzh train: 100.00 % of all arcs and 99.98 % of CROSSING arcs fall
within 30 tokens. Docs here average 77 tokens (max 234) because `convert -n 10` groups ten
sentences, so unwindowed O(n^2) scoring would grow with the length of the CALL -- standing hazard 10.

⚠ NODE 0 IS A VIRTUAL ROOT and every token may attach to it, so CLE returns a FOREST and sentence
boundaries survive. This project's parsers double as sentencisers.
"""
import argparse, json, pathlib, sys, time
import numpy as np

sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import DocBin, Doc
from thinc.api import Adam, NumpyOps
from sud_cle import mst

NEG = -1e4

# ⚠ SIGNED DISTANCE BUCKETS -- the fix for the defect the error decomposition found.
# The biaffine scores a (head, dependent) pair from the two token vectors ALONE, and the encoder is
# a CNN with window 1 depth 4, i.e. a receptive field of +-4 tokens. Beyond that the model cannot
# tell a candidate head 2 tokens away from one 20 tokens away. Measured at epoch 1 against the
# transition parser, which gets distance for free from its stack/buffer state:
#     dist 1  -15.30    dist 3  -26.37    dist 5  -40.12    dist 7  -50.00
# Root attachment, by contrast, was already +2.70 AHEAD, and label-given-head only -2.4 behind, so
# distance is where essentially the whole deficit sits.
_EDGES = (1, 2, 3, 4, 5, 7, 11, 19, 31)


def dist_buckets(n, window):
    """(n+1, n) int bucket ids over [virtual root | tokens] x dependents."""
    h = np.arange(n + 1)[:, None] - 1          # row 0 is the virtual root
    d = np.arange(n)[None, :]
    delta = h - d
    mag = np.abs(delta)
    b = np.zeros_like(delta)
    for i, e in enumerate(_EDGES):
        b = np.where(mag >= e, i + 1, b)
    b = b * 2 - (delta < 0).astype(int)        # separate the two directions
    b = np.maximum(b, 0) + 1                   # shift, leaving 0 free for the root row
    b[0, :] = 0                                # every arc FROM the virtual root shares one bucket
    return b


N_DIST_BINS = 2 * len(_EDGES) + 2

C = ("corpus_lzh_resplit_ctl/lzh_kyoto-sud-%s."
     "relabeled_ext.udep_ruled.punct.rulemerged.resplit.spacy")


def load(split, nlp, limit=None):
    docs = list(DocBin().from_disk(C % split).get_docs(nlp.vocab))
    return docs[:limit] if limit else docs


def vectors(nlp, docs, batch=64):
    t2v = nlp.get_pipe("tok2vec")
    out = []
    for i in range(0, len(docs), batch):
        chunk = [Doc(nlp.vocab, words=[t.text for t in d]) for d in docs[i:i + batch]]
        out.extend(t2v.model.predict(chunk))
    return out


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    idx = np.arange(n)
    m[:, 1:] = np.abs(idx[:, None] - idx[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


class Biaffine:
    """Scores head->dep over a window; labels scored only for the selected arc."""

    def __init__(self, w, h, nlab, seed=0):
        r = np.random.default_rng(seed)
        s = lambda *d: (r.normal(size=d) * (1.0 / np.sqrt(d[0]))).astype("float32")
        self.p = {"Wh": s(w, h), "bh": np.zeros(h, "float32"),
                  "Wd": s(w, h), "bd": np.zeros(h, "float32"),
                  "U": np.zeros((h, h), "float32"), "u": np.zeros(h, "float32"),
                  "dist": np.zeros(N_DIST_BINS, "float32"),
                  "Lh": s(w, h), "Ld": s(w, h),
                  "V": np.zeros((nlab, h, h), "float32"), "v": np.zeros((nlab, 2 * h), "float32"),
                  "cb": np.zeros(nlab, "float32")}
        self.h, self.nlab = h, nlab

    def backprop_inputs(self, dH, dD, dLH, dLD, H, D, LH, LD):
        """Gradient wrt the ENCODER's output -- needed only for --joint."""
        return ((dH * (H > 0)) @ self.p["Wh"].T + (dD * (D > 0)) @ self.p["Wd"].T
                + (dLH * (LH > 0)) @ self.p["Lh"].T + (dLD * (LD > 0)) @ self.p["Ld"].T)

    use_dist = True

    def arc_scores(self, X, k, drop=0.0, rng=None):
        n = X.shape[0]
        H = np.maximum(X @ self.p["Wh"] + self.p["bh"], 0)
        D = np.maximum(X @ self.p["Wd"] + self.p["bd"], 0)
        # inverted dropout on the projections, as in Dozat & Manning: the masks must be kept and
        # reapplied in backprop, or the gradient is for a different network than the forward pass.
        self.mh = self.md = None
        if drop > 0 and rng is not None:
            keep = 1.0 - drop
            self.mh = (rng.random(H.shape) < keep).astype("float32") / keep
            self.md = (rng.random(D.shape) < keep).astype("float32") / keep
            H = H * self.mh; D = D * self.md
        Hr = np.vstack([np.zeros((1, self.h), "float32"), H])
        S = (Hr @ self.p["U"]) @ D.T + (self.p["u"] @ D.T)[None, :]
        self.bkt = dist_buckets(n, k)
        if self.use_dist:
            S = S + self.p["dist"][self.bkt]
        S = np.where(window_mask(n, k).T, S, NEG)
        return S, H, D, Hr

    def label_scores(self, X, heads):
        n = X.shape[0]
        LH = np.maximum(X @ self.p["Lh"], 0); LD = np.maximum(X @ self.p["Ld"], 0)
        hv = np.where((heads > 0)[:, None], LH[np.maximum(heads - 1, 0)], 0.0)
        bil = np.einsum("nh,lhg,ng->nl", hv, self.p["V"], LD)
        lin = np.concatenate([hv, LD], 1) @ self.p["v"].T
        return bil + lin + self.p["cb"], LH, LD, hv


def softmax_ce(S, gold):
    """per-dependent CE over candidate heads; S is (n+1, n) with the virtual root in row 0"""
    Z = S.T                                     # (n, n+1)
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    n = Z.shape[0]
    loss = -np.log(np.maximum(P[np.arange(n), gold], 1e-9)).sum()
    dZ = P.copy(); dZ[np.arange(n), gold] -= 1.0
    return loss, dZ.T                           # (n+1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="training_lzh_depmorph_resplit/model-best")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    # ⚠ THE FREEZE RECIPE IS WRONG FOR THIS DECODER, which is why --joint exists. The shared
    # encoder was fitted feeding a TRANSITION system's state-based feedforward net over stack and
    # buffer slots; nothing in that objective required a head-projection . dep-projection dot
    # product to be meaningful. Frozen, this parser plateaus at UAS 47.8 against the transition
    # parser's ~80 while its LOSS keeps falling -- fitting the objective without it converting into
    # head accuracy, i.e. a representation that cannot express what is being asked.
    # ⚠ THE FIRST JOINT RUN OVERFIT: loss fell 2.16 -> 0.85 while UAS moved 72.3 -> 73.1 over the
    # last five epochs. It had no dropout, no LR schedule and a batch size of ONE, against spaCy's
    # mature loop which has all three -- so "arc-factored loses by 8 LAS" was measuring the TRAINER,
    # not the architecture. These three flags exist to make that comparison honest.
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--decay", type=float, default=1.0, help="multiply lr by this each epoch")
    # ⚠ THE LAST CONFOUND IN THE COMPARISON. The transition parser's encoder is CO-TRAINED with a
    # tagger (`pipeline = [tok2vec, tagger, parser]`), so it gets multi-task regularisation for
    # free; this parser's encoder saw parse loss alone. Both training regimes tried so far converge
    # to ~73 UAS -- constant-LR/no-dropout and decayed-LR/dropout alike -- so the ceiling is not the
    # optimiser. Co-training is the remaining structural difference.
    # ⚠ ADDED AFTER LOSING AN ARM. The dropout run was stopped at epoch 9 still climbing (73.68
    # UAS) and could not be resumed or even ERROR-ANALYSED, because nothing was ever written to
    # disk. A training script with no checkpoint produces a number and not a model.
    # ⚠ THE RECEPTIVE FIELD, NOT THE DISTANCE PRIOR, IS THE BINDING CONSTRAINT. Adding signed
    # distance buckets bought +9 UAS at epoch 0 -- but the decomposition showed ALL of it came from
    # distance 1-2 (67 % of arcs), while distance 5+ stayed at 0-17 % against the transition
    # parser's 43-66 %. A per-bucket bias supplies a PRIOR ("heads are usually near"); it cannot
    # DISCRIMINATE among distant candidates, because a scalar shared by every pair at that distance
    # says nothing about WHICH token seven away is the head. With a CNN of window 1 and depth 4 the
    # receptive field is +-4, so for a head seven away neither representation contains any trace of
    # the other. A BiLSTM has unbounded context in both directions and is what Dozat & Manning use.
    # ⚠ THE DISTANCE BUCKETS ARE A TRADE, NOT A FREE WIN, which only a per-slice decomposition
    # showed. Measured against the same model without them:
    #     overall  60.50 -> 63.01     dist 1  67.76 -> 73.45   (helped: 67 % of arcs are near)
    #     root     83.10 -> 76.50     dist 4  43.38 -> 39.15
    #                                 dist 5  25.51 -> 19.03
    #                                 dist 7   7.58 ->  1.52   (hurt: the prior misleads)
    # A scalar per distance says "heads are usually near" and nothing about WHICH distant token is
    # the head, so it wins the common case and loses the rare one -- and costs 6.6 points of root
    # accuracy. With a BiLSTM supplying real long-range context the prior is worse than redundant.
    ap.add_argument("--batch", type=int, default=1,
                    help="docs per optimiser step; 1 was the original and wastes BLAS throughput")
    # ⚠ PREDICTED XPOS, NEVER GOLD. The lzh pipeline is [tok2vec, TAGGER, parser], so the tagger's
    # output is genuinely available to the parser at runtime -- this is a legitimate feature, not a
    # leak. But it must be the tagger's PREDICTION at training time too; training on gold tags and
    # meeting predicted ones at inference is the skew that has cost this project 3 points before.
    ap.add_argument("--tagfeat", action="store_true",
                    help="add predicted XPOS as an encoder input attribute")
    ap.add_argument("--no-dist", action="store_true",
                    help="disable the signed-distance buckets")
    ap.add_argument("--bilstm", action="store_true",
                    help="MultiHashEmbed -> BiLSTM encoder instead of the CNN")
    ap.add_argument("--save", default="", help="directory to write the best model into")
    ap.add_argument("--cotrain", action="store_true",
                    help="add an XPOS tagging head on the shared encoder, as the transition arm has")
    ap.add_argument("--tag-weight", type=float, default=1.0)
    ap.add_argument("--joint", action="store_true",
                    help="train a fresh encoder jointly with the biaffine instead of freezing")
    a = ap.parse_args()
    nlp = spacy.load(a.src)
    tr = load("train", nlp, a.limit or None)
    te = load("test", nlp, (a.limit // 4) if a.limit else None)
    print(f"  train {len(tr)} docs, test {len(te)}", flush=True)
    labs = sorted({t.dep_ for d in tr for t in d})
    li = {l: i for i, l in enumerate(labs)}
    tags = sorted({t.tag_ for d in tr for t in d}) if a.cotrain else []
    ti = {t: i for i, t in enumerate(tags)}
    if a.cotrain:
        print(f"  CO-TRAIN: {len(tags)} XPOS tags share the encoder with the parser", flush=True)
    print(f"  {len(labs)} deprel labels; window {a.window}", flush=True)
    enc = None
    if a.joint:
        from spacy.util import registry
        from thinc.api import chain as _chain
        if a.bilstm:
            _attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"] + (["TAG"] if a.tagfeat else [])
            _rows = [5000, 1000, 2500, 2500] + ([500] if a.tagfeat else [])
            embed = registry.architectures.get("spacy.MultiHashEmbed.v2")(
                width=96, attrs=_attrs, rows=_rows, include_static_vectors=False)
            # ⚠ THINC'S NATIVE LSTM, NOT `spacy.TorchBiLSTMEncoder.v1`. That one pulls in
            # PyTorch, which is 437 MB installed against this project's documented 250 MB
            # SERVERLESS BUDGET (docs/packaging-and-release.md, where zh went to trouble to drop
            # 36 MB to fit). It would also break the standing constraint that the wheel carries
            # "no transformer, no torch and no inference cost" -- the reason SikuBERT was distilled
            # to a static table in the first place. thinc is 4.2 MB and already a hard dependency.
            from thinc.api import LSTM, with_padded
            lstm = with_padded(LSTM(96, 96, bi=True, depth=2))
            enc = _chain(embed, lstm)
            enc.set_dim("nO", 96)
        else:
            enc = registry.architectures.get("spacy.HashEmbedCNN.v2")(
                width=96, depth=4, embed_size=2000, window_size=1, maxout_pieces=3,
                subword_features=True, pretrained_vectors=None)
        enc.initialize(X=[Doc(nlp.vocab, words=[t.text for t in d]) for d in tr[:64]])
        print(f"  JOINT: training a fresh {'BiLSTM (depth 2)' if a.bilstm else 'CNN (window 1, depth 4)'}"
              f" width-96 encoder with the biaffine", flush=True)
        Xtr = Xte = None
        w = 96
    else:
        Xtr, Xte = vectors(nlp, tr), vectors(nlp, te)
        w = Xtr[0].shape[1]
    def _mk(docs):
        out = [Doc(nlp.vocab, words=[t.text for t in d]) for d in docs]
        if a.tagfeat:
            _t2v, _tag = nlp.get_pipe("tok2vec"), nlp.get_pipe("tagger")
            for i in range(0, len(out), 64):
                ch = out[i:i + 64]
                for dd in ch: _t2v(dd); _tag(dd)
        return out
    plain_tr, plain_te = _mk(tr), _mk(te)
    if a.tagfeat:
        print("  TAG feature: predicted XPOS added to the encoder input", flush=True)
    m = Biaffine(w, a.hidden, len(labs))
    m.use_dist = not a.no_dist
    if a.no_dist:
        print("  distance buckets DISABLED", flush=True)
    ops = NumpyOps(); opt = Adam(a.lr)
    gold_tr = [(np.array([0 if t.head.i == t.i else t.head.i + 1 for t in d]),
                np.array([li.get(t.dep_, 0) for t in d]),
                np.array([ti.get(t.tag_, 0) for t in d]) if a.cotrain else None) for d in tr]
    if a.cotrain:
        rr = np.random.default_rng(7)
        m.p["Wt"] = (rr.normal(size=(w, len(tags))) / np.sqrt(w)).astype("float32")
        m.p["bt"] = np.zeros(len(tags), "float32")
    drng = np.random.default_rng(1234)
    best = (-1.0, -1)
    for ep in range(a.epochs):
        opt.learn_rate = a.lr * (a.decay ** ep)
        order = np.random.default_rng(ep).permutation(len(tr))
        tot = 0.0; t0 = time.time()
        bi = 0
        while bi < len(order):
            chunk = order[bi:bi + a.batch]; bi += a.batch
            c = bi
            if enc is not None:
                Xs_b, bp_enc = enc([plain_tr[j] for j in chunk], is_train=True)
            else:
                Xs_b, bp_enc = [Xtr[j] for j in chunk], None
            gacc = {}; dX_b = []
            for bslot, di in enumerate(chunk):
              X = Xs_b[bslot]
              gh, gl, gt = gold_tr[di]; n = X.shape[0]
              S, H, D, Hr = m.arc_scores(X, a.window, a.dropout, drng)
              loss, dS = softmax_ce(S, gh)
              LS, LH, LD, hv = m.label_scores(X, gh)
              Z = LS - LS.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
              loss += -np.log(np.maximum(P[np.arange(n), gl], 1e-9)).sum()
              dL = P.copy(); dL[np.arange(n), gl] -= 1.0
              tot += loss / max(n, 1)
              g = {}
              HW = Hr @ m.p["U"]; dHW = dS @ D
              g["U"] = Hr.T @ dHW
              g["u"] = dS.sum(0) @ D
              g["dist"] = (np.bincount(m.bkt.ravel(), weights=dS.ravel(),
                                     minlength=N_DIST_BINS).astype("float32")
                         if m.use_dist else np.zeros(N_DIST_BINS, "float32"))
              dD = dS.T @ HW + np.outer(dS.sum(0), m.p["u"])
              dH = (dHW @ m.p["U"].T)[1:]
              # ⚠ REAPPLY THE DROPOUT MASKS. H and D here are the POST-mask activations, so
              # (H > 0) still selects the right units, but the mask scaling must be carried back.
              if m.mh is not None:
                dH = dH * m.mh; dD = dD * m.md
              g["Wh"] = X.T @ (dH * (H > 0)); g["bh"] = (dH * (H > 0)).sum(0)
              g["Wd"] = X.T @ (dD * (D > 0)); g["bd"] = (dD * (D > 0)).sum(0)
              g["V"] = np.einsum("nl,nh,ng->lhg", dL, hv, LD)
              g["v"] = dL.T @ np.concatenate([hv, LD], 1)
              g["cb"] = dL.sum(0)
              dhv = np.einsum("nl,lhg,ng->nh", dL, m.p["V"], LD) + dL @ m.p["v"][:, :m.h]
              dLD = np.einsum("nl,lhg,nh->ng", dL, m.p["V"], hv) + dL @ m.p["v"][:, m.h:]
              dXt = None
              if a.cotrain:
                T = X @ m.p["Wt"] + m.p["bt"]
                Zt = T - T.max(1, keepdims=True); Pt = np.exp(Zt); Pt /= Pt.sum(1, keepdims=True)
                tot += a.tag_weight * float(-np.log(np.maximum(Pt[np.arange(n), gt], 1e-9)).mean())
                dT = Pt.copy(); dT[np.arange(n), gt] -= 1.0; dT *= a.tag_weight
                g["Wt"] = X.T @ dT; g["bt"] = dT.sum(0)
                dXt = dT @ m.p["Wt"].T
              dLH = np.zeros_like(LH)
              src = gh - 1; ok = gh > 0
              np.add.at(dLH, src[ok], dhv[ok])
              g["Lh"] = X.T @ (dLH * (LH > 0)); g["Ld"] = X.T @ (dLD * (LD > 0))
              if bp_enc is not None:
                dXin = m.backprop_inputs(dH, dD, dLH, dLD, H, D, LH, LD)
                if dXt is not None:
                    dXin = dXin + dXt          # the tagging head shapes the SAME encoder
                dX_b.append(dXin.astype("float32"))
              for kk in g:
                gacc[kk] = gacc.get(kk, 0) + g[kk]
            # ⚠ ONE optimiser step PER BATCH, after accumulating every doc's gradient. Stepping
            # per doc (the original) is batch size 1 however the data is grouped.
            if bp_enc is not None and dX_b:
                bp_enc(dX_b)
                enc.finish_update(opt)
            for kk in gacc:
                m.p[kk], _ = opt(("bi", kk), m.p[kk], (gacc[kk] / len(chunk)).astype("float32"))
            if c and c % 1500 == 0:
                print(f"    ep{ep} {c}/{len(tr)} loss {tot/(c+1):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        # ---- evaluate with CLE ----
        uas = las = ntok = 0
        Xte_ep = ([enc.predict([pd])[0] for pd in plain_te] if enc is not None else Xte)
        for d, X in zip(te, Xte_ep):
            n = X.shape[0]
            S, *_ = m.arc_scores(X, a.window)      # eval: no dropout
            # `mst` takes a SQUARE matrix over [virtual root | tokens]; arc_scores emits
            # (n+1 heads, n dependents), so pad a dependent column for the root, which may
            # never take a head.
            Sq = np.full((n + 1, n + 1), NEG, dtype="float64")
            Sq[:, 1:] = S
            # ⚠ mst returns n+1 entries with the VIRTUAL ROOT at index 0; the real tokens are
            # 1..n. Everything downstream is per-token, so drop the root entry. Its VALUES are
            # already in the same [0=root, 1..n=token] convention as the gold heads.
            heads = mst(Sq)[1:]
            LS, *_ = m.label_scores(X, heads)
            pl = LS.argmax(1)
            for i, t in enumerate(d):
                gh_ = 0 if t.head.i == t.i else t.head.i + 1
                ntok += 1
                if heads[i] == gh_:
                    uas += 1
                    if labs[pl[i]] == t.dep_: las += 1
        print(f"  epoch {ep}: loss {tot/len(tr):.4f}   UAS {uas*100/ntok:.2f}   LAS {las*100/ntok:.2f}"
              f"   (lr {opt.learn_rate:.2e})", flush=True)
        if a.save and las > best[0]:
            best = (las, ep)
            out = pathlib.Path(a.save); out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "biaffine.npz", **{k: v for k, v in m.p.items()})
            if enc is not None:
                (out / "encoder.bin").write_bytes(enc.to_bytes())
            (out / "meta.json").write_text(json.dumps(
                {"labels": labs, "tags": tags, "window": a.window, "hidden": a.hidden,
                 "epoch": ep, "uas": uas*100/ntok, "las": las*100/ntok, "joint": bool(a.joint),
                 "cotrain": bool(a.cotrain)}, ensure_ascii=False, indent=1))
            print(f"    saved -> {a.save} (best LAS {las*100/ntok:.2f})", flush=True)


if __name__ == "__main__":
    main()
