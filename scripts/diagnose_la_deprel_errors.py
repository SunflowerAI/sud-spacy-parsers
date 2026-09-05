#!/usr/bin/env python
"""What KIND of mistake does the la arc-factored decoder make, deprel by deprel?

The decomposition (`analyse_arcfactored.py`) already showed the worst-hit relations on the
presegmented checkpoint: comp:obl -11.76, cc -7.37, conj:coord -6.32 -- all well below the
aggregate -3.20. This asks two sharper questions per relation, not just "how much worse":

  (1) IS IT JUST THE LONG-ARC STORY AGAIN? If comp:obl/coordination arcs are systematically longer
      than average, their loss could be entirely explained by the already-diagnosed distance
      weakness, with nothing relation-specific going on.
  (2) WHEN WRONG, DOES THE MODEL PREFER A NEARER CANDIDATE THAN THE CORRECT ONE? The signed
      distance bias is a "heads are usually near" prior (train_arcfactored.py). If comp:obl -- the
      classic PP-ATTACHMENT AMBIGUITY relation, verb vs. a closer noun -- shows a strong bias
      toward attaching NEARER than gold specifically on ITS errors, that is the bias fighting a
      relation where the correct answer is disproportionately often the FARTHER candidate.

Prints, per relation: n, gold LAS, arc-fact accuracy, mean gold arc length, and among WRONG
predictions: mean predicted-vs-gold distance delta (negative = model went nearer than gold) and a
few concrete worked examples.
"""
import argparse, collections, json, pathlib, sys
import numpy as np

sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import Doc
from spacy.util import registry
from sud_cle import mst
import train_arcfactored as tr

NEG = -1e4


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    i = np.arange(n)
    m[:, 1:] = np.abs(i[:, None] - i[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


def decode(meta, P, X, doc=None):   # doc unused: only the joint-label agreement term needs it
    n = X.shape[0]; h = meta["hidden"]
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    Hr = np.vstack([np.zeros((1, h), "float32"), H])
    S = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    if "dist" in P:
        S = S + P["dist"][tr.dist_buckets(n, meta["window"])]
    S = np.where(window_mask(n, meta["window"]).T, S, NEG)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    LH = np.maximum(X @ P["Lh"], 0); LD = np.maximum(X @ P["Ld"], 0)
    hv = np.where((heads > 0)[:, None], LH[np.maximum(heads - 1, 0)], 0.0)
    sc = (np.einsum("nh,lhg,ng->nl", hv, P["V"], LD)
          + np.concatenate([hv, LD], 1) @ P["v"].T + P["cb"])
    return heads, [meta["labels"][i] for i in sc.argmax(1)]


def decode_joint_label(meta, P, X, doc=None):
    """MUST STAY IN SYNC with sud_joint_biaffine.JointBiaffine.forward/decode_scores (including the lemcase bilinear term).

    `doc` is required when meta["agreement"], meta["pos"], meta["feat_names"] or meta["pron"] is
    true -- see analyse_arcfactored.py's twin function for why skipping it would silently evaluate
    the checkpoint one or more terms short."""
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
    dist_term = P["dist"].T[bkt]
    combined = arc_raw[:, :, None] + label_raw + dist_term
    if meta.get("agreement") and "agree" in P:
        assert doc is not None, "agreement-enabled checkpoint needs the doc to compute the bias"
        combined = combined + P["agree"].T[tr.agreement_buckets(doc)]
    if meta.get("direction") and "direction" in P:
        combined = combined + P["direction"].T[tr.direction_buckets(n, meta["window"])]
    if meta.get("pos") and "pos" in P:
        assert doc is not None, "pos-enabled checkpoint needs the doc to compute the bias"
        combined = combined + P["pos"].T[tr.pos_buckets(doc)]
    if meta.get("lemvec") and "lemvec" in P:
        assert doc is not None, "lemvec-enabled checkpoint needs the doc to compute the bias"
        lv = tr.lemma_vecs(doc, meta["lemvec_table"])
        combined = combined + (lv @ P["lemvec"].T)[:, None, :]
    if meta.get("morphhash") and "morphhash" in P:
        assert doc is not None, "morphhash-enabled checkpoint needs the doc to compute the bias"
        mh = tr.morph_hash_buckets(doc)
        combined = combined + P["morphhash"].T[mh][None, :, :]
    if meta.get("feat_names"):
        assert doc is not None, "feat-enabled checkpoint needs the doc to compute the bias"
        for name in meta["feat_names"]:
            key = f"feat_{name}"
            if key not in P:
                continue
            vocab_index = {tuple(v): i for i, v in enumerate(meta["feat_vocab"][name])}
            fb = tr.feat_buckets(doc, name, vocab_index)
            combined = combined + P[key].T[fb][None, :, :]
    if meta.get("pron") and "pron" in P:
        assert doc is not None, "pron-enabled checkpoint needs the doc to compute the bias"
        pb = tr.preverbal_buckets(doc, meta["window"])
        combined = combined + P["pron"].T[pb]
    if meta.get("lemvec_dep") and "lemvec_dep" in P:
        assert doc is not None, "lemvec_dep-enabled checkpoint needs the doc to compute the bias"
        lvd = tr.lemma_vecs_dep(doc, meta["lemvec_table"])
        combined = combined + (lvd @ P["lemvec_dep"].T)[None, :, :]
    if meta.get("lemcase") and "lemcase" in P:
        assert doc is not None, "lemcase-enabled checkpoint needs the doc to compute the bias"
        lv_lc = tr.lemma_vecs(doc, meta["lemvec_table"])
        lc_vocab_idx = {tuple(v): i for i, v in enumerate(meta["lemcase_vocab"])}
        lc_bkt = tr.feat_buckets(doc, "Case", lc_vocab_idx)
        Mlc = np.einsum("hk,lkc->hlc", lv_lc, P["lemcase"], optimize=True)
        combined = combined + Mlc[:, :, lc_bkt].transpose(0, 2, 1)
    if meta.get("lemhash") and "lemhash" in P:
        assert doc is not None, "lemhash-enabled checkpoint needs the doc to compute the bias"
        lh_bkt = tr.lemma_hash_buckets(doc)
        combined = combined + P["lemhash"].T[lh_bkt][:, None, :]
    mask = window_mask(n, meta["window"]).T
    combined = np.where(mask[:, :, None], combined, NEG)
    S, chosen = combined.max(-1), combined.argmax(-1)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    labels = chosen[heads, np.arange(n)]
    return heads, [meta["labels"][i] for i in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/la_arcfactored_preseg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    meta = json.loads((pathlib.Path(a.model) / "meta.json").read_text())
    P = dict(np.load(pathlib.Path(a.model) / "biaffine.npz"))
    presegment = bool(meta.get("presegment"))
    nlp = spacy.load(meta["src"])
    _, upstream = tr.encoder_and_upstream(nlp)

    embed = tr.build_joint_embed_from_meta(meta)
    from thinc.api import chain as _c, LSTM, with_padded
    if meta.get("bilstm", True):
        enc = _c(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
        enc.set_dim("nO", 96)
    else:
        enc = _c(embed, registry.architectures.get("spacy.MaxoutWindowEncoder.v2")(
            width=96, depth=4, window_size=1, maxout_pieces=3))
    gold = tr.load("la", "test", nlp, a.limit or None)
    probe = Doc(nlp.vocab, words=[t.text for t in gold[0]])
    tr.annotate_upstream(nlp, probe, upstream)
    enc.initialize(X=[probe])
    enc.from_bytes((pathlib.Path(a.model) / "encoder.bin").read_bytes())
    decode_fn = decode_joint_label if meta.get("joint_label") else decode

    stats = collections.defaultdict(lambda: {"n": 0, "ok": 0, "head_wrong": 0, "label_wrong": 0,
                                              "gold_len": [], "err_delta": [], "examples": [],
                                              "confusions": collections.Counter(),
                                              # under --joint-label, a head-wrong error can be the
                                              # LABEL scorer hijacking the ARC decision: the wrong
                                              # head wins because ITS best label scored high, not
                                              # because the position itself looked right. Tracking
                                              # what label WON at the wrong head tests that directly.
                                              "wrong_head_label": collections.Counter()})

    def process(words, sent, gold_toks, off=0):
        """`sent` is the plain Doc handed to the encoder; `gold_toks` are the matching gold tokens;
        `off` maps sent-local indices back to the ORIGINAL doc for readable examples."""
        n = len(sent)
        X = enc.predict([sent])[0]
        heads, labels = decode_fn(meta, P, X, doc=sent)
        for i, t in enumerate(gold_toks):
            gh = 0 if t.head.i == t.i else (t.head.i - off) + 1
            dep = t.dep_
            s = stats[dep]
            s["n"] += 1
            gold_dist = abs((gh or i + 1) - 1 - i)
            s["gold_len"].append(gold_dist)
            head_ok = heads[i] == gh
            ok = head_ok and labels[i] == dep
            s["ok"] += ok
            if not head_ok:
                s["head_wrong"] += 1
                s["wrong_head_label"][labels[i]] += 1
            elif labels[i] != dep:
                s["label_wrong"] += 1
                s["confusions"][labels[i]] += 1
            if not ok and gh > 0:               # skip roots -- "distance" isn't meaningful there
                pred_dist = abs(heads[i] - 1 - i) if heads[i] > 0 else None
                if pred_dist is not None:
                    s["err_delta"].append(pred_dist - gold_dist)
                if len(s["examples"]) < a.examples:
                    lo, hic = max(0, i - 4), min(n, i + 5)
                    ctx = " ".join(f"[{w.text}]" if j == i else w.text for j, w in enumerate(sent[lo:hic], start=lo))
                    gold_head_txt = sent[gh - 1].text if gh > 0 else "ROOT"
                    pred_head_txt = sent[heads[i] - 1].text if heads[i] > 0 else "ROOT"
                    kind = "LABEL ONLY" if head_ok else "head wrong"
                    s["examples"].append(
                        f"      [{kind:<10}] '{t.text}' gold-> '{gold_head_txt}' (dist {gold_dist}, dep={dep})   "
                        f"pred-> '{pred_head_txt}' (dist {pred_dist}, dep={labels[i]})   ...{ctx}...")

    for g in gold:
        if presegment:
            for s in g.sents:
                d = Doc(nlp.vocab, words=[t.text for t in s])
                tr.annotate_upstream(nlp, d, upstream)
                process([t.text for t in s], d, list(s), off=s.start)
        else:
            d = Doc(nlp.vocab, words=[t.text for t in g])
            tr.annotate_upstream(nlp, d, upstream)
            process([t.text for t in g], d, list(g), off=0)

    rows = sorted(stats.items(), key=lambda kv: kv[1]["ok"] / kv[1]["n"])
    print(f"  {'deprel':<16}{'n':>6}{'acc':>8}{'head-wrong':>12}{'label-only':>12}{'gold-len':>10}{'err-delta':>11}  top confusions (label-only)")
    for dep, s in rows:
        if s["n"] < 30:
            continue
        acc = s["ok"] * 100 / s["n"]
        hw = s["head_wrong"] * 100 / s["n"]
        lo = s["label_wrong"] * 100 / s["n"]
        gl = np.mean(s["gold_len"])
        ed = np.mean(s["err_delta"]) if s["err_delta"] else float("nan")
        conf = ", ".join(f"{k}:{v}" for k, v in s["confusions"].most_common(3))
        whl = ", ".join(f"{k}:{v}" for k, v in s["wrong_head_label"].most_common(3))
        print(f"  {dep:<16}{s['n']:>6}{acc:>8.2f}{hw:>12.2f}{lo:>12.2f}{gl:>10.2f}{ed:>+11.2f}  "
              f"label-only:[{conf}]  wrong-head-won-as:[{whl}]")
    print("\n  head-wrong/label-only are % of n. err-delta = mean(|pred-dist| - |gold-dist|) over"
          " WRONG predictions; negative = model attached NEARER than gold, when wrong."
          "\n  wrong-head-won-as: under --joint-label, what label WON at the wrong head -- tests"
          " whether a confusable label hijacked the ARC decision, not just the label read-off.\n")
    for dep in ["comp:obl", "cc", "conj:coord", "subj", "det"]:
        if dep not in stats or not stats[dep]["examples"]:
            continue
        print(f"  -- {dep} worked examples --")
        for ex in stats[dep]["examples"]:
            print(ex)
        print()


if __name__ == "__main__":
    main()
