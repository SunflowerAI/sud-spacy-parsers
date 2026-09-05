#!/usr/bin/env python
"""`sud.ArcFactoredParser.v1` -- the arc-factored (biaffine + Chu-Liu/Edmonds) decoder built and
measured all session in `train_arcfactored.py`, wrapped as an ORDINARY spaCy pipe component so a
trained checkpoint can actually ship in a wheel, rather than only being runnable as a standalone
research script via `analyse_arcfactored.py`. First release target: Sanskrit
(`sa_arcfactored_full`, +6.26 LAS over the shipped transition arm's own official release figure --
see CLAUDE.md/docs/sanskrit.md for the comparison this replaces).

WHY A NEW COMPONENT, NOT A DROP-IN REPLACEMENT OF `spacy.TransitionBasedParser.v2`. This decoder's
forward pass (JointBiaffine) and its decode step (`sud_cle.mst`, Chu-Liu/Edmonds over the WHOLE
doc) are hand-rolled numpy, not a thinc `Model` spaCy's own training/serialization plumbing already
knows how to walk -- so this component owns its OWN `to_disk`/`from_disk` rather than delegating to
`TrainablePipe`'s.

⚠ EVERYTHING THIS DECODER READS MUST BE SEALED INTO THE COMPONENT'S OWN SAVED BYTES, never a
repo-relative path -- CLAUDE.md hazard 4 ("a config path is a host path"), the exact defect that
made `lzh_sud_kyoto-0.3.0` die off the build host. `train_arcfactored.py`'s own LANGS table points
`lemvec_table` at `scripts/<lang>_lemmavec_96.npz` for BUILD-time convenience; `to_disk` here copies
that table's actual keys/vectors into the component's own directory, and `from_disk` never touches
the original path again.

⚠ CALLED ONCE PER `Doc`, WHATEVER IT CONTAINS -- there is no sentence-internal re-segmentation.
This matches the checkpoint's OWN training regime (`--presegment`: one gold SENTENCE per training
item) and, for sa specifically, the EXISTING deployed contract too: `training_sa_mp2_sub_s1` ships
with no senter/sentencizer at all, so a multi-sentence `Doc` was already being parsed as a single
span before this component existed. Feeding this decoder a whole multi-sentence document is
therefore not a new limitation introduced here, but it is also not something to rely on: callers
that need real multi-document throughput should segment upstream (a `sentencizer` pipe, or one
`nlp()` call per sentence) exactly as they would need to for accurate sub-sentence structure from
any parser.

⚠ MUST STAY IN SYNC with `train_arcfactored.py` (bucket functions, the `JointBiaffine` forward
pass) and `sud_joint_biaffine.py` (`JointBiaffine` itself) -- imported directly, not reimplemented,
specifically so there is only one copy of this logic to keep correct. Both ship as `--code`
alongside this file (see `package_sud.sh`'s per-arm `--code` list).
"""
import json
import pathlib
import sys

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sud_cle import mst  # noqa: E402
from sud_joint_biaffine import JointBiaffine  # noqa: E402
import train_arcfactored as _tr  # noqa: E402

NEG = -1e4


def _window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    idx = np.arange(n)
    m[:, 1:] = np.abs(idx[:, None] - idx[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


class ArcFactoredParser:
    """A trained checkpoint, sealed and self-contained. Construction (the factory below) always
    produces an EMPTY shell with no weights -- exactly `sud_lemmavec_embed.py`'s own
    construct-then-restore contract, since spaCy rebuilds a pipeline's components from the config
    BEFORE calling `from_disk` on each one. Only `from_disk` (or `initialize`, for a fresh
    from-scratch build during training) may raise on missing content; `__call__` on an empty shell
    is a programming error, not a runtime one worth guarding softly."""

    def __init__(self, vocab, name="arcfactored_parser"):
        self.vocab = vocab
        self.name = name
        self.meta = None
        self.P = None
        self.encoder = None
        self.upstream = None

    def __call__(self, doc):
        if self.meta is None:
            raise ValueError(
                f"{self.name}: no checkpoint loaded (construction produces an empty shell; "
                f"call from_disk, or load the whole pipeline via spacy.load)")
        n = len(doc)
        if n == 0:
            return doc
        X = self.encoder.predict([doc])[0]
        heads, labels = self._decode(X, doc)
        for i, t in enumerate(doc):
            t.head = doc[i] if heads[i] == 0 else doc[heads[i] - 1]
            t.dep_ = labels[i]
        return doc

    def _decode(self, X, doc):
        """The JointBiaffine forward pass + CLE decode, reimplemented against the loaded weight
        dict -- MUST STAY IN SYNC with `sud_joint_biaffine.JointBiaffine.forward`/`decode_scores`
        and with `analyse_arcfactored.py`'s `predict_from_X_joint_label`, which this mirrors."""
        meta, P = self.meta, self.P
        n = X.shape[0]
        h = meta["hidden"]
        H = np.maximum(X @ P["Wh"] + P["bh"], 0)
        D = np.maximum(X @ P["Wd"] + P["bd"], 0)
        Hr = np.vstack([np.zeros((1, h), H.dtype), H])
        LH = np.maximum(X @ P["Lh"], 0)
        LD = np.maximum(X @ P["Ld"], 0)
        LHr = np.vstack([np.zeros((1, h), LH.dtype), LH])
        arc_raw = (Hr @ P["U"]) @ D.T + (P["u"] @ D.T)[None, :]
        bil = np.einsum("hg,lgk,dk->hdl", LHr, P["V"], LD, optimize=True)
        lin1 = LHr @ P["v"][:, :h].T
        lin2 = LD @ P["v"][:, h:].T
        label_raw = bil + lin1[:, None, :] + lin2[None, :, :] + P["cb"][None, None, :]
        bkt = _tr.dist_buckets(n, meta["window"])
        combined = arc_raw[:, :, None] + label_raw + P["dist"].T[bkt]
        if meta.get("agreement") and "agree" in P:
            combined = combined + P["agree"].T[_tr.agreement_buckets(doc)]
        if meta.get("pos") and "pos" in P:
            combined = combined + P["pos"].T[_tr.pos_buckets(doc)]
        if meta.get("lemvec") and "lemvec" in P:
            lv = self._lemma_vecs(doc)
            combined = combined + (lv @ P["lemvec"].T)[:, None, :]
        if meta.get("morphhash") and "morphhash" in P:
            mh = _tr.morph_hash_buckets(doc)
            combined = combined + P["morphhash"].T[mh][None, :, :]
        if meta.get("feat_names"):
            for name in meta["feat_names"]:
                key = f"feat_{name}"
                if key not in P:
                    continue
                vocab_index = {tuple(v): i for i, v in enumerate(meta["feat_vocab"][name])}
                fb = _tr.feat_buckets(doc, name, vocab_index)
                combined = combined + P[key].T[fb][None, :, :]
        if meta.get("direction") and "direction" in P:
            combined = combined + P["direction"].T[_tr.direction_buckets(n, meta["window"])]
        if meta.get("pron") and "pron" in P:
            combined = combined + P["pron"].T[_tr.preverbal_buckets(doc, meta["window"])]
        if meta.get("lemvec_dep") and "lemvec_dep" in P:
            lvd = self._lemma_vecs_dep(doc)
            combined = combined + (lvd @ P["lemvec_dep"].T)[None, :, :]
        if meta.get("lemcase") and "lemcase" in P:
            lv_lc = self._lemma_vecs(doc)
            lc_vocab_idx = {tuple(v): i for i, v in enumerate(meta["lemcase_vocab"])}
            lc_bkt = _tr.feat_buckets(doc, "Case", lc_vocab_idx)
            Mlc = np.einsum("hk,lkc->hlc", lv_lc, P["lemcase"], optimize=True)
            combined = combined + Mlc[:, :, lc_bkt].transpose(0, 2, 1)
        if meta.get("lemhash") and "lemhash" in P:
            lh_bkt = _tr.lemma_hash_buckets(doc)
            combined = combined + P["lemhash"].T[lh_bkt][:, None, :]
        if meta.get("lemhashdep") and "lemhashdep" in P:
            lhd_bkt = _tr.lemma_hash_buckets_dep(doc)
            combined = combined + P["lemhashdep"].T[lhd_bkt][None, :, :]
        mask = _window_mask(n, meta["window"]).T
        combined = np.where(mask[:, :, None], combined, NEG)
        S, chosen = combined.max(-1), combined.argmax(-1)
        Sq = np.full((n + 1, n + 1), NEG, dtype="float64")
        Sq[:, 1:] = S
        heads = mst(Sq)[1:]
        label_ids = chosen[heads, np.arange(n)]
        return heads, [meta["labels"][i] for i in label_ids]

    # -- the two lemma-vector lookups read the SEALED table (self._lemvec_idx/_lemvec_V), never a
    # repo path -- see `to_disk`/`from_disk`.
    def _lemma_vecs(self, doc):
        dim = self._lemvec_V.shape[1]
        out = np.zeros((len(doc) + 1, dim), dtype="float32")
        for i, t in enumerate(doc):
            j = self._lemvec_idx.get(t.lemma_ or t.text)
            if j is not None:
                out[i + 1] = self._lemvec_V[j]
        return out

    def _lemma_vecs_dep(self, doc):
        dim = self._lemvec_V.shape[1]
        out = np.zeros((len(doc), dim), dtype="float32")
        for i, t in enumerate(doc):
            j = self._lemvec_idx.get(t.lemma_ or t.text)
            if j is not None:
                out[i] = self._lemvec_V[j]
        return out

    def to_disk(self, path, exclude=()):
        out = pathlib.Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "biaffine.npz", **{k: v for k, v in self.P.items()})
        (out / "encoder.bin").write_bytes(self.encoder.to_bytes())
        (out / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1))
        if getattr(self, "_lemvec_idx", None) is not None:
            keys = [""] * len(self._lemvec_idx)
            for k, i in self._lemvec_idx.items():
                keys[i] = k
            np.savez_compressed(out / "lemvec.npz", keys=np.array(keys, dtype=object),
                                 vectors=self._lemvec_V)

    def from_disk(self, path, exclude=()):
        inp = pathlib.Path(path)
        self.meta = json.loads((inp / "meta.json").read_text())
        self.P = dict(np.load(inp / "biaffine.npz"))
        self.upstream = self.meta.get("upstream", [])
        embed = _tr.build_joint_embed_from_meta(self.meta)
        from thinc.api import chain as _chain
        if self.meta.get("bilstm", True):
            from thinc.api import LSTM, with_padded
            enc = _chain(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
            enc.set_dim("nO", 96)
        else:
            from spacy.util import registry
            enc = _chain(embed, registry.architectures.get("spacy.MaxoutWindowEncoder.v2")(
                width=96, depth=4, window_size=1, maxout_pieces=3))
        probe = Doc(self.vocab, words=["x"])
        enc.initialize(X=[probe])
        enc.from_bytes((inp / "encoder.bin").read_bytes())
        self.encoder = enc
        lv_path = inp / "lemvec.npz"
        if lv_path.exists():
            d = np.load(lv_path, allow_pickle=True)
            keys = [str(k) for k in d["keys"]]
            self._lemvec_idx = {k: i for i, k in enumerate(keys)}
            self._lemvec_V = d["vectors"].astype("float32")
        else:
            self._lemvec_idx = self._lemvec_V = None
        return self


@Language.factory("sud_arcfactored_parser")
def make_arcfactored_parser(nlp, name):
    return ArcFactoredParser(nlp.vocab, name=name)
