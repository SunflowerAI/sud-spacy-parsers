#!/usr/bin/env python
"""WHERE does the arc-factored parser lose to (or beat) the transition parser? Decompose, don't guess.

Generalised from `analyse_lzh_arcfactored.py` -- see that file for why each cut exists. Six cuts:

  ROOT / SENTENCE BOUNDARY -- an arc-factored decoder discovers sentence boundaries IMPLICITLY, as
      tokens that attach to the virtual root across a whole multi-sentence doc; the transition
      parser has a dedicated BREAK action for exactly this.
  ARC LENGTH        -- graph-based decoders usually win on long arcs and lose on local ones.
  CROSSING ARCS     -- the whole motivation for trying this on la/sa in the first place.
  PER-DEPREL        -- which relations carry the loss.
  LABEL GIVEN HEAD  -- separates the arc scorer from the label scorer.
  SENTENCE LENGTH   -- does the loss grow with the segmentation burden?

⚠ THE TRANSITION BASELINE RUNS THE WHOLE REAL PIPELINE, IN ORDER, up to and including "parser" --
not just `tok2vec` then `parser` -- because la's and sa's released parsers read predicted
LEMMA/MORPH set by components that sit BEFORE them (see `train_arcfactored.py`'s module docstring).
Running only tok2vec+parser would silently starve them of that input and understate the baseline
this decoder is meant to beat.

⚠ NOT EVERY TRANSITION BASELINE CAN TAKE A WHOLE MULTI-SENTENCE DOC. la's source
(`training_la_lemvec_sud`) trains through `sud.GoldTokCorpus.v1` -- multi-sentence docs, gold
tokenisation, genuinely taught to place a BREAK -- so feeding it a whole doc tests exactly what it
was built for. sa's source (`training_sa_mp2_sub_s1`) trains through `sud.NormCorpus.v1` with
`gold_preproc=true`, i.e. ONE SENTENCE PER EXAMPLE (CLAUDE.md hazard 11: "every dev example is
already one sentence, so SENTS_F reads 100.00 for a parser that never learned to start one").
First measured here: fed a whole 85-token/10-sentence sa doc raw, it predicted ONE root total
(gold has ten) and ROOT accuracy read 8.08 % -- not a real weakness, an out-of-regime test. `sentencises()`
detects this from the loaded model's own OWN config (`corpora.train.@readers`) rather than a
per-language hardcode, and when it is false the transition baseline is run per GOLD SENTENCE and
stitched back to whole-doc indices -- handing the transition side its ideal case (sentence
boundaries for free) while the arc-factored decoder still gets none, which is the conservative
direction to be wrong in, not a thumb on the scale for this decoder.

⚠ MUST STAY IN SYNC with `train_arcfactored.py`'s scorer (LANGS table, window_mask, dist_buckets,
agreement_buckets, direction_buckets, pos_buckets, morph_hash_buckets, feat_buckets,
preverbal_buckets, lemma_vecs, lemma_vecs_dep, lemma_hash_buckets, lemma_hash_buckets_dep,
sibling_buckets, grandparent_buckets, clausegap_buckets) AND with
`sud_self_attention.attn_forward` -- read directly for `--attn-hd` (JointBiaffine.use_attn_hd,
applied to H/D) and reimplemented inline as `_apply_attn`
for the superseded `--attn` (applied to X; kept only for reproducing its documented negative result).
"""
import argparse, json, pathlib, sys, collections
import numpy as np
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import DocBin, Doc
from spacy.util import registry
from sud_cle import mst
from sud_self_attention import attn_forward   # a pure function, not a trainable class -- consistent
                                               # with this module's "never instantiate the trainable
                                               # classes for prediction" style (mst is also imported
                                               # directly, for the same reason)
import importlib.util as _iu
_sp = _iu.spec_from_file_location("_tr", "scripts/train_arcfactored.py")
assert _sp is not None and _sp.loader is not None
_tr = _iu.module_from_spec(_sp); _sp.loader.exec_module(_tr)

NEG = -1e4


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    i = np.arange(n)
    m[:, 1:] = np.abs(i[:, None] - i[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


def load_arcfactored(path, plain_probe):
    meta = json.loads((pathlib.Path(path) / "meta.json").read_text())
    P = dict(np.load(pathlib.Path(path) / "biaffine.npz"))
    # ⚠ ATTN WEIGHTS ARE PREFIXED (attn_Wq, ...) into the SAME dict, not returned separately -- no
    # key collision with the biaffine's own params (Wh/Wd/U/u/Lh/Ld/V/v/cb/dist/...), and every
    # downstream caller already threads P through unchanged, so this needs no new return value or
    # call-site plumbing; `_apply_attn` reads them back out by the same prefixed names.
    if meta.get("attn"):
        attn_npz = pathlib.Path(path) / "attn.npz"
        assert attn_npz.exists(), f"meta.json says attn=true but {attn_npz} is missing"
        for k, v in np.load(attn_npz).items():
            P[f"attn_{k}"] = v
    if meta.get("joint"):
        from thinc.api import chain as _c
        embed = _tr.build_joint_embed_from_meta(meta)
        if meta.get("bilstm"):
            from thinc.api import LSTM, with_padded
            enc = _c(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
            enc.set_dim("nO", 96)
        else:
            enc = _c(embed, registry.architectures.get("spacy.MaxoutWindowEncoder.v2")(
                width=96, depth=4, window_size=1, maxout_pieces=3))
        enc.initialize(X=[plain_probe])
        enc.from_bytes((pathlib.Path(path) / "encoder.bin").read_bytes())
    else:
        enc = None      # frozen mode: caller extracts vectors via encoder_and_upstream() directly
    return meta, P, enc


def _apply_attn(P, X):
    """Reimplements SelfAttentionMixer.forward's math against the loaded weight dict, exactly the
    style every other term in this module already uses. MUST STAY IN SYNC with
    sud_self_attention.py's own forward()."""
    Wq, Wk, Wv, Wo = P["attn_Wq"], P["attn_Wk"], P["attn_Wv"], P["attn_Wo"]
    w = Wq.shape[0]
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    scores = (Q @ K.T) * (1.0 / np.sqrt(w))
    scores = scores - scores.max(1, keepdims=True)
    E = np.exp(scores); A = E / E.sum(1, keepdims=True)
    return X + (A @ V) @ Wo


def predict_from_X(meta, P, X, doc=None):   # doc unused: only the joint-label agreement term needs it
    if meta.get("attn") and "attn_Wq" in P:
        X = _apply_attn(P, X)
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


def predict_from_X_joint_label(meta, P, X, doc=None):
    """The --joint-label scorer's own forward+decode (sud_joint_biaffine.JointBiaffine), reimplemented
    against the loaded weight dict rather than importing the class, matching predict_from_X's own
    style. MUST STAY IN SYNC with JointBiaffine.forward/decode_scores.

    `doc` is required (and used) when meta["agreement"], meta["pos"], meta["feat_names"] or
    meta["pron"] is true -- all are computed from the doc's tokens, not from X, so skipping it here
    would silently evaluate the checkpoint one or more terms short of what it was trained with."""
    if meta.get("attn") and "attn_Wq" in P:
        X = _apply_attn(P, X)
    n = X.shape[0]; h = meta["hidden"]; nlab = len(meta["labels"])
    H = np.maximum(X @ P["Wh"] + P["bh"], 0)
    D = np.maximum(X @ P["Wd"] + P["bd"], 0)
    if meta.get("attn_hd") and "attn_h_Wq" in P:
        # ⚠ --attn-hd refines H/D AFTER Wh/Wd's own per-token projection, not X beforehand -- see
        # sud_joint_biaffine.py's own note on use_attn_hd. MUST STAY IN SYNC with its forward(),
        # `attn_window` included -- evaluating a windowed checkpoint unmasked would score it against
        # a mechanism it was never trained with.
        aw = meta.get("attn_window")
        H, _ = attn_forward(H, P["attn_h_Wq"], P["attn_h_Wk"], P["attn_h_Wv"], P["attn_h_Wo"],
                             window=aw)
        D, _ = attn_forward(D, P["attn_d_Wq"], P["attn_d_Wk"], P["attn_d_Wv"], P["attn_d_Wo"],
                             window=aw)
    Hr = np.vstack([np.zeros((1, h), H.dtype), H])
    LH = np.maximum(X @ P["Lh"], 0); LD = np.maximum(X @ P["Ld"], 0)
    LHr = np.vstack([np.zeros((1, h), LH.dtype), LH])
    arc_raw = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
    bil = np.einsum("hg,lgk,dk->hdl", LHr, P["V"], LD, optimize=True)
    lin1 = LHr @ P["v"][:, :h].T
    lin2 = LD @ P["v"][:, h:].T
    label_raw = bil + lin1[:, None, :] + lin2[None, :, :] + P["cb"][None, None, :]
    bkt = _tr.dist_buckets(n, meta["window"])
    dist_term = P["dist"].T[bkt]
    combined = arc_raw[:, :, None] + label_raw + dist_term
    if meta.get("agreement") and "agree" in P:
        assert doc is not None, "agreement-enabled checkpoint needs the doc to compute the bias"
        combined = combined + P["agree"].T[_tr.agreement_buckets(doc)]
    if meta.get("direction") and "direction" in P:
        combined = combined + P["direction"].T[_tr.direction_buckets(n, meta["window"])]
    if meta.get("pos") and "pos" in P:
        assert doc is not None, "pos-enabled checkpoint needs the doc to compute the bias"
        combined = combined + P["pos"].T[_tr.pos_buckets(doc)]
    if meta.get("lemvec") and "lemvec" in P:
        assert doc is not None, "lemvec-enabled checkpoint needs the doc to compute the bias"
        lv = _tr.lemma_vecs(doc, meta["lemvec_table"])
        combined = combined + (lv @ P["lemvec"].T)[:, None, :]
    if meta.get("morphhash") and "morphhash" in P:
        assert doc is not None, "morphhash-enabled checkpoint needs the doc to compute the bias"
        mh = _tr.morph_hash_buckets(doc)
        combined = combined + P["morphhash"].T[mh][None, :, :]
    if meta.get("feat_names"):
        assert doc is not None, "feat-enabled checkpoint needs the doc to compute the bias"
        for name in meta["feat_names"]:
            key = f"feat_{name}"
            if key not in P:
                continue
            vocab_index = {tuple(v): i for i, v in enumerate(meta["feat_vocab"][name])}
            fb = _tr.feat_buckets(doc, name, vocab_index)
            combined = combined + P[key].T[fb][None, :, :]
    if meta.get("pron") and "pron" in P:
        assert doc is not None, "pron-enabled checkpoint needs the doc to compute the bias"
        pb = _tr.preverbal_buckets(doc, meta["window"])
        combined = combined + P["pron"].T[pb]
    if meta.get("lemvec_dep") and "lemvec_dep" in P:
        assert doc is not None, "lemvec_dep-enabled checkpoint needs the doc to compute the bias"
        lvd = _tr.lemma_vecs_dep(doc, meta["lemvec_table"])
        combined = combined + (lvd @ P["lemvec_dep"].T)[None, :, :]
    if meta.get("lemcase") and "lemcase" in P:
        assert doc is not None, "lemcase-enabled checkpoint needs the doc to compute the bias"
        lv_lc = _tr.lemma_vecs(doc, meta["lemvec_table"])
        lc_vocab_idx = {tuple(v): i for i, v in enumerate(meta["lemcase_vocab"])}
        lc_bkt = _tr.feat_buckets(doc, "Case", lc_vocab_idx)
        Mlc = np.einsum("hk,lkc->hlc", lv_lc, P["lemcase"], optimize=True)
        combined = combined + Mlc[:, :, lc_bkt].transpose(0, 2, 1)
    if meta.get("lemhash") and "lemhash" in P:
        assert doc is not None, "lemhash-enabled checkpoint needs the doc to compute the bias"
        lh_bkt = _tr.lemma_hash_buckets(doc)
        combined = combined + P["lemhash"].T[lh_bkt][:, None, :]
    if meta.get("lemhashdep") and "lemhashdep" in P:
        assert doc is not None, "lemhashdep-enabled checkpoint needs the doc to compute the bias"
        lhd_bkt = _tr.lemma_hash_buckets_dep(doc)
        combined = combined + P["lemhashdep"].T[lhd_bkt][None, :, :]
    if meta.get("clausegap") and "clausegap" in P:
        assert doc is not None, "clausegap-enabled checkpoint needs the doc to compute the bias"
        cgb = _tr.clausegap_buckets(doc)
        combined = combined + P["clausegap"].T[cgb]
    mask = window_mask(n, meta["window"]).T
    combined = np.where(mask[:, :, None], combined, NEG)
    if (meta.get("sibling") and "sib" in P) or (meta.get("grandparent") and "grand" in P):
        # ⚠ TWO-PASS: `combined` here is exactly PASS 1 (every other bias term already added,
        # masked) -- decode it via CLE, turn that PREDICTED (never gold) tree into the sibling/
        # grandparent bucket grids, add whichever bias this checkpoint has, then decode AGAIN.
        # MUST STAY IN SYNC with train_arcfactored.py's own two-pass procedure (its training loop
        # and eval loop) -- ONE shared first-order decode for both flags, not one each.
        S0, chosen0 = combined.max(-1), combined.argmax(-1)
        Sq0 = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq0[:, 1:] = S0
        heads0 = mst(Sq0)[1:]
        labels0 = chosen0[heads0, np.arange(n)]
        if meta.get("sibling") and "sib" in P:
            sib_bkt = _tr.sibling_buckets(heads0, labels0, n)
            combined = combined + P["sib"].T[sib_bkt]
        if meta.get("grandparent") and "grand" in P:
            grand_bkt = _tr.grandparent_buckets(heads0, labels0, n)
            combined = combined + P["grand"][:, grand_bkt].T[:, None, :]
        combined = np.where(mask[:, :, None], combined, NEG)
    S, chosen = combined.max(-1), combined.argmax(-1)
    Sq = np.full((n + 1, n + 1), NEG, dtype="float64"); Sq[:, 1:] = S
    heads = mst(Sq)[1:]
    labels = chosen[heads, np.arange(n)]
    return heads, [meta["labels"][i] for i in labels]


def sentencises(nlp):
    """Was this model's SOURCE trained on multi-sentence docs (so it can place a BREAK inside a
    whole doc), or on gold_preproc single-sentence examples (so it never learned to)? Read off the
    model's own stored config, never hand-listed per language."""
    try:
        return nlp.config["corpora"]["train"]["@readers"] == "sud.GoldTokCorpus.v1"
    except Exception:
        return False


def transition_predict(nlp, full_chain, g, whole_doc):
    """Run the ACTUAL transition pipeline and return (heads, deps) at whole-doc indices.

    `whole_doc=True` feeds the parser the entire multi-sentence doc, its real deployed regime for
    an arm trained via `sud.GoldTokCorpus.v1`. `whole_doc=False` instead runs it per GOLD SENTENCE
    and stitches the result back -- the fair test for an arm that never saw more than one sentence
    per training example, and generous to it (gold boundaries for free) rather than to this
    decoder.
    """
    n = len(g)
    if whole_doc:
        d0 = Doc(nlp.vocab, words=[t.text for t in g])
        for name in full_chain:
            nlp.get_pipe(name)(d0)
        return [t.head.i for t in d0], [t.dep_ for t in d0]
    heads = [0] * n; deps = ["ROOT"] * n
    for s in g.sents:
        off = s.start
        d0 = Doc(nlp.vocab, words=[t.text for t in s])
        for name in full_chain:
            nlp.get_pipe(name)(d0)
        for i, t in enumerate(d0):
            heads[off + i] = off + t.head.i
            deps[off + i] = t.dep_
    return heads, deps


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
    ap.add_argument("--lang", required=True, choices=sorted(_tr.LANGS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", default="", help="default: LANGS[lang]['src']")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    baseline = a.baseline or _tr.LANGS[a.lang]["src"]
    nlp = spacy.load(baseline)
    encoder, upstream = _tr.encoder_and_upstream(nlp)
    parser_idx = nlp.pipe_names.index("parser")
    full_chain = nlp.pipe_names[:parser_idx + 1]     # everything up to and incl. the parser itself
    gold = _tr.load(a.lang, "test", nlp, a.limit or None)
    probe = Doc(nlp.vocab, words=[t.text for t in gold[0]])
    _tr.annotate_upstream(nlp, probe, upstream)
    meta, P, enc = load_arcfactored(a.model, probe)
    predict_fn = predict_from_X_joint_label if meta.get("joint_label") else predict_from_X
    presegment = bool(meta.get("presegment"))
    # ⚠ A PRESEGMENTED CHECKPOINT NEVER SAW MORE THAN ONE SENTENCE AT A TIME, so it must be scored
    # that way, and the transition baseline must be forced into the SAME regime for the comparison
    # to mean anything -- see `explode_sentences()`'s docstring. An un-presegmented checkpoint keeps
    # using whatever regime its OWN source model was actually trained/deployed under.
    whole_doc = False if presegment else sentencises(nlp)
    print(f"  [{a.lang}] arc-factored upstream: {upstream}   transition full chain: {full_chain}")
    print(f"  arc-factored regime: {'PRESEGMENT (per gold sentence)' if presegment else 'whole multi-sentence doc'}")
    print(f"  transition baseline regime: {'whole multi-sentence doc' if whole_doc else 'per GOLD SENTENCE'}"
          + ("" if presegment else (" (source never learned to segment)" if not sentencises(nlp) else "")))
    print(f"  arc-factored checkpoint: epoch {meta['epoch']}, dev LAS {meta['las']:.2f}"
          f"   ({'joint' if meta.get('joint') else 'frozen'})")
    cut = lambda: collections.defaultdict(lambda: [0, 0, 0])   # n, af_ok, tp_ok
    root, length, cross, dep, sent = cut(), cut(), cut(), cut(), cut()
    lab_given_head = [0, 0, 0, 0]
    tot = [0, 0, 0]
    for g in gold:
        words = [t.text for t in g]; n = len(g)
        th_raw, td_raw = transition_predict(nlp, full_chain, g, whole_doc)
        if presegment:
            ah = [0] * n; al = ["ROOT"] * n
            for s in g.sents:
                off = s.start
                d1 = Doc(nlp.vocab, words=[t.text for t in s])
                _tr.annotate_upstream(nlp, d1, upstream)
                X = enc.predict([d1])[0] if enc is not None else _tr.per_doc(encoder.predict([d1]), [d1])[0]
                ah_s, al_s = predict_fn(meta, P, X, doc=d1)
                for i in range(len(s)):
                    ah[off + i] = 0 if ah_s[i] == 0 else off + ah_s[i]
                    al[off + i] = al_s[i]
        else:
            d1 = Doc(nlp.vocab, words=words)
            _tr.annotate_upstream(nlp, d1, upstream)
            if enc is not None:
                X = enc.predict([d1])[0]
            else:
                X = _tr.per_doc(encoder.predict([d1]), [d1])[0]
            ah, al = predict_fn(meta, P, X, doc=d1)
        gh = [0 if t.head.i == t.i else t.head.i + 1 for t in g]
        th = [0 if th_raw[i] == i else th_raw[i] + 1 for i in range(n)]
        gcross = crossing_set([t.head.i for t in g])
        slen = {}
        for s in g.sents:
            for t in s: slen[t.i] = len(s)
        for i, t in enumerate(g):
            af = ah[i] == gh[i] and al[i] == t.dep_
            tp = th[i] == gh[i] and td_raw[i] == t.dep_
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
                lab_given_head[2] += 1; lab_given_head[3] += td_raw[i] == t.dep_
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
