#!/usr/bin/env python
"""`lzh_upos_stack`: arbitrate between the morphologiser's UPOS and the tagger's XPOS.

THE IDEA. Two heads predict overlapping things from different targets: the morphologiser emits a
joint (POS, FEATS) label, the tagger a 122-value XPOS of which 97 values map to exactly one UPOS.
They are equally accurate on the hard slice and FAIL DIFFERENTLY, so a selector that decides which
to trust recovers accuracy neither reaches alone. This is not a new feature -- it is arbitration
between two predictions the pipeline already computes.

⚠ THE GAIN DEPENDS ON THE PIPELINE ORDER, AND THE RELEASED ORDER COSTS MOST OF IT.
    [tok2vec, tagger, parser, morphologizer]  tagger independent   +0.493 overall, +1.389 N/V
    [tok2vec, parser, morphologizer, tagger]  tagger reads UPOS    +0.121 overall, +0.247 N/V
Feeding UPOS to the tagger halves the disagreements (3591 -> 1942) and drops its hit-rate within
them (37.4 % -> 27.4 %): it echoes the UPOS it was given, so the decorrelation the selector feeds on
is engineered away. The RELEASED order is shipped because `package_sud.sh` refuses `tagger` before
`morphologizer` (the tagger is meant to read UPOS+FEATS) and because that order preserves the
`upos_mask` editing feature. This component therefore ships the SMALLER, SAFER of the two gains.

⚠ NO sklearn AT INFERENCE. The selector is trained with sklearn and its weights exported to a plain
`.npz`; the forward pass here is four matmuls. A pickled estimator would add a dependency and break
on a version bump.

⚠ ARTEFACTS LIVE IN THE COMPONENT DIRECTORY, written by `to_disk` and read by `from_disk` -- never
a path from the config. A factory argument naming a host file is what made `lzh_sud_kyoto-0.3.0`
die with FileNotFoundError on every machine but the build host (CLAUDE.md standing hazard 4).
"""
from typing import Optional
import numpy as np
from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("lzh_upos_stack", default_config={"enabled": True},
                  requires=["token.pos", "token.tag"], assigns=["token.pos"])
def make_lzh_upos_stack(nlp: Language, name: str, enabled: bool):
    return LzhUposStack(enabled=enabled)


class LzhUposStack:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.W = []           # list of (weight, bias) for the MLP
        self.mean = self.scale = None
        self.x2u = {}         # XPOS -> implied UPOS
        self.xpur = {}        # XPOS -> purity of that mapping
        self.ambn = {}        # form -> how many UPOS values it takes in train
        self.nv = set()       # forms attested as both NOUN and VERB
        self.pairs = []       # (morph UPOS -> tagger UPOS) one-hot vocabulary
        self.upos = []

    # ---- inference ----
    def _forward(self, x):
        for i, (W, b) in enumerate(self.W):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, doc: Doc) -> Doc:
        if not self.enabled or not self.W:
            return doc
        rows, idx, tus = [], [], []
        for t in doc:
            tu = self.x2u.get(t.tag_)
            if tu is None or tu == t.pos_:
                continue
            rows.append(self._feats(doc, t, tu)); idx.append(t.i); tus.append(tu)
        if not rows:
            return doc
        X = (np.asarray(rows, "float32") - self.mean) / self.scale
        p = self._forward(X).ravel()
        for k, i in enumerate(idx):
            if p[k] >= 0.5:
                doc[i].pos_ = tus[k]
        return doc

    def _feats(self, doc, t, tu):
        ui = {u: i for i, u in enumerate(self.upos)}
        pi = {q: i for i, q in enumerate(self.pairs)}
        base = [self.xpur.get(t.tag_, 0.5), float(t.text in self.nv),
                float(np.log1p(self.ambn.get(t.text, 1))),
                float(ui.get(t.pos_, len(self.upos))), float(ui.get(tu, len(self.upos)))]
        oh = [0.0] * len(self.pairs)
        k = pi.get(f"{t.pos_}->{tu}")
        if k is not None:
            oh[k] = 1.0
        return base + oh

    # ---- serialisation: everything inside the component directory ----
    def to_disk(self, path, exclude=tuple()):
        import pathlib, json
        p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "mlp.npz", **{f"W{i}": w for i, (w, _) in enumerate(self.W)},
                 **{f"b{i}": b for i, (_, b) in enumerate(self.W)},
                 mean=self.mean, scale=self.scale)
        (p / "tables.json").write_text(json.dumps(
            {"x2u": self.x2u, "xpur": self.xpur, "ambn": self.ambn,
             "nv": sorted(self.nv), "pairs": self.pairs, "upos": self.upos},
            ensure_ascii=False), encoding="utf-8")

    def from_disk(self, path, exclude=tuple()):
        import pathlib, json
        p = pathlib.Path(path)
        if not (p / "mlp.npz").exists():
            # ⚠ REFUSE rather than silently no-op: a component that loads without its model and
            # quietly does nothing is the failure mode CLAUDE.md hazard 8 was written for.
            raise IOError(f"{p}: lzh_upos_stack has no mlp.npz; the selector cannot run")
        z = np.load(p / "mlp.npz")
        n = sum(1 for k in z.files if k.startswith("W"))
        self.W = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.mean, self.scale = z["mean"], z["scale"]
        d = json.loads((p / "tables.json").read_text(encoding="utf-8"))
        self.x2u, self.xpur = d["x2u"], d["xpur"]
        self.ambn, self.nv = d["ambn"], set(d["nv"])
        self.pairs, self.upos = d["pairs"], d["upos"]
        return self

    def to_bytes(self, exclude=tuple()):
        import io, json
        buf = io.BytesIO()
        np.savez(buf, **{f"W{i}": w for i, (w, _) in enumerate(self.W)},
                 **{f"b{i}": b for i, (_, b) in enumerate(self.W)},
                 mean=self.mean, scale=self.scale)
        meta = json.dumps({"x2u": self.x2u, "xpur": self.xpur, "ambn": self.ambn,
                           "nv": sorted(self.nv), "pairs": self.pairs, "upos": self.upos})
        return json.dumps({"npz": buf.getvalue().hex(), "meta": meta}).encode("utf-8")

    def from_bytes(self, data, exclude=tuple()):
        import io, json
        d = json.loads(data.decode("utf-8"))
        z = np.load(io.BytesIO(bytes.fromhex(d["npz"])))
        n = sum(1 for k in z.files if k.startswith("W"))
        self.W = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.mean, self.scale = z["mean"], z["scale"]
        m = json.loads(d["meta"])
        self.x2u, self.xpur = m["x2u"], m["xpur"]
        self.ambn, self.nv = m["ambn"], set(m["nv"])
        self.pairs, self.upos = m["pairs"], m["upos"]
        return self
